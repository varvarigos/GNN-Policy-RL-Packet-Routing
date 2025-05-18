# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ipdb

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

        log_probs_list, entropies_list, arrived_packets_list = [], [], []
        arrived_packets_time_list = []
        total_packets_list = []
        total_packets = 0

        episode_progress_bar = tqdm(total=cfg.training.timesteps, desc=f"Episode {episode + 1}", leave=False)
        previous_total_arrived_packets = 0
        previous_total_arrived_packets_time = 0
        for _ in range(cfg.training.timesteps):
            log_probs = []
            entropies = []
            ### CLASS ENVIRONMNET ----- sample action and put to queuee ---- move packets across network
            ############### ------------------------ PACKET CREATION STEP -------------------- ###############
            # new_packets: rows = src, columns = dst ----- new_packets: bool
            # traffic_matrix should a table f probabilities between 0 and 1
            # create at most one packet per source.
            # new_packets should be a vector of ints (max int is NUM_NODES) or -1 if no packet is created
            # if timestep < cfg.training.timesteps // 2:
            new_packets = environment.create_packets()
            action_dist = policy(new_packets, environment)
            action_info = environment.step(new_packets, action_dist)
            log_prob = action_info["action_log_prob"]
            entropy = action_info["action_entropy"]
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
            # and for each loop run the initial step that assigns things to queues
            for final_dst, initial_src, time in zip(arrived_destinations, arrived_sources, arrived_times, strict=True):
                action_dist = policy(final_dst, environment)
                action_info = environment.step(final_dst, action_dist, initial_src, time)

                log_prob = action_info["action_log_prob"]
                entropy = action_info["action_entropy"]

                log_probs.append(log_prob)
                entropies.append(entropy)

            # ipdb.set_trace()

            log_probs_list.append(torch.cat(log_probs).sum())
            entropies_list.append(torch.cat(entropies).sum())

            total_arrived_packets = environment.total_arrived_packets.sum()
            arrived_packets = total_arrived_packets.item() - previous_total_arrived_packets
            previous_total_arrived_packets = total_arrived_packets.item()
            arrived_packets_list.append(arrived_packets)

            total_arrived_packets_time = environment.total_arrived_packets_time.sum()
            arrived_packets_time = total_arrived_packets_time.item() - previous_total_arrived_packets_time
            previous_total_arrived_packets_time = total_arrived_packets_time.item()
            arrived_packets_time_list.append(arrived_packets_time)

            if total_packets != 0 and total_packets == environment.total_arrived_packets.sum():
                break

            total_packets += (new_packets >= 0).sum()
            total_packets_list.append(total_packets.item()) # can remove this HERE
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

        # --- Compute per-timestep reward
        rewards = [
            arrived_packets # - arrived_time / (arrived_packets if arrived_packets > 0 else 1)
            for arrived_packets, arrived_time in zip(arrived_packets_list, arrived_packets_time_list)
        ]

        # --- Compute discounted returns
        gamma = cfg.reward.gamma
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # --- Compute vanilla policy method loss
        policy_loss = -torch.cat([log_prob * ret for log_prob, ret in zip(log_probs, returns)]).mean()

        # --- Compute entropy loss
        entropy_loss = -torch.cat(entropies).mean()

        # --- Compute total loss
        loss = policy_loss + cfg.reward.entropy_beta * entropy_loss

        # ipdb.set_trace()

        # --- Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Log metrics
        total_arrived_packets = environment.total_arrived_packets.sum()
        training_progress_bar.update(1)
        if total_arrived_packets > 0:
            # Compute final metrics
            total_arrived = environment.total_arrived_packets.sum().item()
            total_arrived_time = environment.total_arrived_packets_time.sum().item()
            delivery_ratio = 100 * total_arrived / total_packets.item()
            avg_packet_time = total_arrived_time / total_arrived
            log_params = {
                "total_loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "reward": torch.tensor(rewards).mean().item(),
                "entropy": entropy_loss.item(),
                "delivery_ratio[%]": delivery_ratio,
                "avg_packet_time": avg_packet_time,
            }
            for tag, scalar_value in log_params.items():
                writer.add_scalar(tag, scalar_value, global_step=episode, walltime=None, new_style=False)
            training_progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "policy_loss": f"{policy_loss.item():.4f}",
                    "reward": torch.tensor(rewards).mean().item(),
                    "entropy": f"{entropy_loss.item():.4f}",
                    "delivery[%]": f"{delivery_ratio:.1f}",
                    "avg_time": f"{avg_packet_time:.2f}",
                }
            )


if __name__ == "__main__":
    main()
