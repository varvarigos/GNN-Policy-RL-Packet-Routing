from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class TrainingConfig:
    seed: int = 42
    episodes: int = 500
    timesteps: int = 500
    batch_size: int = 32


@dataclass
class NetworkEnvironmentConfig:
    _target_: str = "network_env.NetworkEnvironment"
    transmission_time: int = 1
    num_of_nodes: int = 10
    packet_creation_prob: float = 0.2


@dataclass
class RewardConfig:
    value: float = 0.1
    gamma: float = 0.99
    entropy_beta: float = 0.001


@dataclass
class OptimizerConfig:
    lr: float = 5e-5


@dataclass
class GNNPolicyConfig:
    _target_: str = "policy.GNNPolicy"
    node_dim: int = 3
    edge_dim: int = 3
    hidden_dim: int = 64
    num_of_nodes: int = 10


@dataclass
class ExperimentConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network_environment: NetworkEnvironmentConfig = field(default_factory=NetworkEnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    gnn_policy: GNNPolicyConfig = field(default_factory=GNNPolicyConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
cs.store(group="training", name="base_training", node=TrainingConfig)
cs.store(group="network_environment", name="base_network_environment", node=NetworkEnvironmentConfig)
cs.store(group="reward", name="base_reward", node=RewardConfig)
cs.store(group="gnn_policy", name="base_gnn_policy", node=GNNPolicyConfig)
cs.store(group="optimizer", name="base_optimizer", node=OptimizerConfig)
