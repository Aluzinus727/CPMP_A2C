from typing import List, Tuple, NamedTuple

from torch import nn, Tensor
from torch.distributions import Categorical

from config import Config


def mlp(
    input_size: int,
    layer_sizes: List[int],
    output_size: int,
    output_activation=nn.Identity,
    activation=nn.ReLU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]

    print(f"Layer: {layers}")
    return nn.Sequential(*layers)


class ModelOutput(NamedTuple):
    policy_logits: Tensor
    value: Tensor


class ActorCriticModel(nn.Module):
    def __init__(self, config: Config):
        super(ActorCriticModel, self).__init__()
        self.action_space_size = config.action_space_size

        self.representation_network = mlp(
            config.observation_size, config.representation_layers, config.encoding_size
        )

        self.actor_network = mlp(
            config.encoding_size, config.actor_layers, config.action_space_size
        )

        self.critic_network = mlp(config.encoding_size, config.critic_layers, 1)

    def forward(self, observation: Tensor) -> ModelOutput:
        x = self.representation_network.forward(observation)
        policy_logits = self.actor_network.forward(x)
        value = self.critic_network.forward(x)
        return ModelOutput(policy_logits, value)

    def evaluate_actions(
        self, observations: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        policy_logits, values = self.forward(observations)
        dist = Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy