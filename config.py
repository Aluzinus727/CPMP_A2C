from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    """
    Clase que contiene los parámetros necesarios para la ejecución del programa.
    """
    nb_stacks: int = 7
    height: int = 7
    nb_containers: int = 20
    observation_size: int = 42
    action_space_size: int = 20
    encoding_size: int = 42
    representation_layers: List[int] = field(default_factory=list)
    actor_layers: List[int] = field(default_factory=list)
    critic_layers: List[int] = field(default_factory=list)
    lr: float = 1e-3
    seed: int = 0
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = None
    normalize_advantage: bool = False
    batch_size: int = 16
    nb_rollout_steps: int = 500
    training_steps: int = 2e3
    nb_episodes: int = 1e6
    gamma: float = 1
    checkpoint_interval: int = 100
    device: torch.device = torch.device("cuda")