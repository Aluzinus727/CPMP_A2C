import numpy as np
from config import Config


class Buffer:
    def __init__(self, config: Config):
        self.config = config
        self.observations = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.returns = []
        self.advantages = []

    def add(
        self, state: np.ndarray, reward: float, action: int, mask: float, value: float
    ):
        """
        Se guardan los datos en las colecciones que correspondan.
        """
        
        self.observations.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.masks.append(mask)
        self.values.append(value)

    def compute_returns_and_advantage(self, last_value: float, gamma: float) -> None:
        """
        Calcula los advantages obtenidos durante la ejecución.
        """

        delta = last_value

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * delta * self.masks[step]
            self.returns.insert(0, delta)

        self.advantages = np.array(self.returns) - np.array(self.values)

    def reset(self) -> None:
        """
        Vacía todas las listas del buffer.
        """

        self.observations = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []
        self.returns = []
        self.advantages = []

