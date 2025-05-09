from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExperimentConfig
from utils import seed_everything

if TYPE_CHECKING:
    from network_env import NetworkEnvironment
    from policy import GNNPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(42)
torch.autograd.set_detect_anomaly(True)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: ExperimentConfig) -> None:
    cfg = OmegaConf.to_object(cfg)

    writer = SummaryWriter(log_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "logs"))

    environment: NetworkEnvironment = instantiate(cfg.network_environment).to(device)
    policy: GNNPolicy = instantiate(cfg.gnn_policy).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.optimizer.lr)

    training_progress_bar = tqdm(total=cfg.training.episodes, desc="Training")
    for episode in range(cfg.training.episodes):
        environment.reset()

        log_probs, entropies = [], []
        total_packets = 0

        episode_progress_bar = tqdm(total=cfg.training.timesteps, desc=f"Episode {episode + 1}", leave=False)
        for _ in range(cfg.training.timesteps):
            ### CLASS ENVIRONMNET ----- sample action and put to queuee ---- move packets across network
            ############### ------------------------ PACKET CREATION STEP -------------------- ###############
            # new_packets: rows = src, columns = dst ----- new_packets: bool
            # traffic_matrix should a table f probabilities between 0 and 1
            # create at most one packet per source.
            # new_packets should be a vector of ints (max int is NUM_NODES) or -1 if no packet is created
            # if timestep < cfg.training.timesteps // 2:
            new_packets = environment.create_packets()
            action_dist: torch.distributions.Categorical = policy(new_packets, environment)
            action_smth = environment.step(new_packets, action_dist)
            log_prob = action_smth["action_log_prob"]
            entropy = action_smth["action_entropy"]
            log_probs.append(log_prob)
            entropies.append(entropy)

            arrived = environment.advance_time()
            arrivals_to_forward = environment.check_arrival(**arrived)
            arrived_destinations = arrivals_to_forward["destinations"]  # final destinations
            arrived_sources = arrivals_to_forward["sources"]  # initial sources
            arrived_times = arrivals_to_forward["times"]  # times

            ############### ---------------- PUT NEWLY ARRIVED PACKETS TO QUEUES STEP ----------------- ###############
            # rerun the initial step (make it a function or something)
            # you need to have a loop over arrived_packets (for new packets in arrived_packets)
            # and for each loop run the iniital step that assigns things to queues
            for final_dst, initial_src, time in zip(arrived_destinations, arrived_sources, arrived_times, strict=True):
                action_dist: torch.distributions.Categorical = policy(final_dst, environment)
                action_smth = environment.step(final_dst, action_dist, initial_src, time)
                log_prob = action_smth["action_log_prob"]
                entropy = action_smth["action_entropy"]
                log_probs.append(log_prob)
                entropies.append(entropy)

            if total_packets != 0 and total_packets == environment.total_arrived_packets.sum():
                break

            total_packets += (new_packets >= 0).sum()
            episode_progress_bar.update(1)
            episode_progress_bar.set_postfix(
                {
                    "new_packets": (new_packets >= 0).sum().item(),
                    "arrived_packets[%]": 100 * environment.total_arrived_packets.sum().item() / total_packets.item()
                    if total_packets.item() != 0
                    else 0,
                    "avg_package_time": environment.total_arrived_packets_time.sum().item()
                    / environment.total_arrived_packets.sum().item()
                    if environment.total_arrived_packets.sum().item() != 0
                    else 0,
                }
            )

        # --- Policy Update ---
        total_arrived_packets = environment.total_arrived_packets.sum()
        total_arrived_packets_time = environment.total_arrived_packets_time.sum()
        # reward = total_arrived_packets * cfg.reward.value - (total_packets - total_arrived_packets) * cfg.reward.value
        reward = (
            -total_arrived_packets_time * cfg.reward.value / total_arrived_packets
            + 200 * total_arrived_packets * cfg.reward.value / total_packets
        ) / 10

        reward = 1 / total_arrived_packets_time / cfg.reward.value * total_arrived_packets

        if total_arrived_packets > 0:
            # rewards = torch.cat(rewards)
            avg_log_prob = torch.cat(log_probs).mean()
            avg_entropy = torch.cat(entropies).mean()

            # --- Compute REINFORCE loss ---
            if reward > 0:
                loss = -avg_log_prob * reward - cfg.reward.entropy_beta * avg_entropy
            else:
                loss = -cfg.reward.entropy_beta * avg_entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_progress_bar.update(1)
        if total_arrived_packets > 0:
            log_params = {
                "loss": loss.item(),
                "avg_log_prob": avg_log_prob.item(),
                "avg_entropy": avg_entropy.item(),
                "reward": reward.item(),
                "arrived_packets[%]": 100 * environment.total_arrived_packets.sum().item() / total_packets.item(),
                "avg_package_time": environment.total_arrived_packets_time.sum().item()
                / environment.total_arrived_packets.sum().item(),
            }
            for tag, scalar_value in log_params.items():
                writer.add_scalar(tag, scalar_value, global_step=episode, walltime=None, new_style=False)
            training_progress_bar.set_postfix(log_params)


if __name__ == "__main__":
    main()
