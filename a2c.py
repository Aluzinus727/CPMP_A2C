import os
import datetime
import numpy as np

import torch
from torch import Tensor
from torch.functional import F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import Tuple
from collections import deque

from buffer import Buffer
from config import Config
from greedy import greedy_solve
from utils import process_state
from model import ActorCriticModel
from environment import Enviroment

DATE = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")


class A2CTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = ActorCriticModel(config).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr)

        self.env = Enviroment(
            config=config
        )

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.buffer = Buffer(self.config)
        self.log_dir = os.path.join("results", "CPMP_A2C", DATE)
        self._state = self.env.reset()

    def sample_action(self, policy: Tensor):
        """
        Selecciona una acción a aplicar al Enviroment.
        """
        dist = Categorical(logits=policy)
        return dist.sample().item()

    def collect_rollouts_greedy(self):
        """
        Recorre el Enviroment y recolecta la información necesaria para el training.
        Actua según un algoritmo greedy.
        """
        self.buffer.reset()

        step = 0
        total_steps = self.config.batch_size * self.config.nb_rollout_steps

        while step < total_steps:
            done = False
            self.env.reset()
            moves = greedy_solve(self.env.layout)

            for move in moves:
                with torch.no_grad():
                    observation = (
                        torch.tensor(np.array(process_state(self._state)))
                        .float()
                        .to(self.config.device)
                    )
                    output = self.model.forward(observation)
                    
                action = self.env.mapped_act_space[move]
                next_state, reward, done = self.env.step(action)

                self.ep_reward += reward
                self.ep_length += 1
                step += 1
                
                self.buffer.add(
                    process_state(self._state), reward, action, 1.0 - done, output.value.item()
                )

                self._state = next_state

                if done:
                    self.ep_rewards.append(self.ep_reward)
                    self.ep_lengths.append(self.ep_length)
                    self.ep_length = 0
                    self.ep_reward = 0.0

                    self._state = self.env.reset()

                    self.current_ep += 1

                    if self.current_ep >= self.config.nb_episodes:
                        break

                if step >= total_steps:
                    break

        with torch.no_grad():
            observation = (
                torch.tensor(np.array(process_state(self._state))).float().to(self.config.device)
            )
            output = self.model.forward(observation)
            self.buffer.compute_returns_and_advantage(
                output.value.item(), self.config.gamma
            )

    def collect_rollouts(self):
        """
        Recorre el Enviroment y recolecta la información necesaria para el training.
        Actua en base a lo que prediga el modelo.
        """
        self.buffer.reset()

        step = 0
        total_steps = self.config.batch_size * self.config.nb_rollout_steps

        while step < total_steps:
            done = False
            while not done:
                with torch.no_grad():
                    observation = (
                        torch.tensor(np.array(process_state(self._state)))
                        .float()
                        .to(self.config.device)
                    )
                    output = self.model.forward(observation)
                    action = self.sample_action(output.policy_logits)

                next_state, reward, done = self.env.step(action)

                self.ep_reward += reward
                self.ep_length += 1
                step += 1

                self.buffer.add(
                    process_state(self._state), reward, action, 1.0 - done, output.value.item()
                )

                self._state = next_state

                if done:
                    self.ep_rewards.append(self.ep_reward)
                    self.ep_lengths.append(self.ep_length)
                    self.ep_length = 0
                    self.ep_reward = 0.0

                    self._state = self.env.reset()

                    self.current_ep += 1

                    if self.current_ep >= self.config.nb_episodes:
                        break

                if step >= total_steps:
                    break

        with torch.no_grad():
            observation = (
                torch.tensor(np.array(process_state(self._state))).float().to(self.config.device)
            )
            output = self.model.forward(observation)
            self.buffer.compute_returns_and_advantage(
                output.value.item(), self.config.gamma
            )

    def log_step(
        self, loss: float, actor_loss: float, critic_loss: float, entropy_loss: float
    ):
        """
        Guarda la información para visualizarla en TensorBoard.
        """
        self.writer.add_scalar(
            "1.Total_reward/1.Total_reward",
            np.mean(self.ep_rewards),
            self.current_step,
        )
        self.writer.add_scalar(
            "1.Total_reward/2.Episode_length",
            np.mean(self.ep_lengths),
            self.current_step,
        )

        self.writer.add_scalar(
            "1.Total_reward/3.Episodes", self.current_ep, self.current_step,
        )

        self.writer.add_scalar("2.Loss/1.Total_weighted_loss", loss, self.current_step)
        self.writer.add_scalar("2.Loss/2.Actor_loss", actor_loss, self.current_step)
        self.writer.add_scalar("2.Loss/3.Critic_loss", critic_loss, self.current_step)
        self.writer.add_scalar("2.Loss/4.Entropy_loss", entropy_loss, self.current_step)

    def check_end_condition(self):
        """
        Se encarga de ver la condicion de termino del training.
        """
        while (
            self.current_step < self.config.training_steps
            and self.current_ep < self.config.nb_episodes
        ):
            yield

    def save_checkpoint(self):
        """
        Se encarga de guardar la información del modelo.
        """
        checkpoint = {
            "training_step": self.current_step,
            "episodes": self.current_ep,
            "weights": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        path = os.path.join(self.log_dir, "model.checkpoint")
        torch.save(checkpoint, path)

    def load_model(self, checkpoint_path=None):
        """
        Se carga el checkpoint.

        Args:
            checkpoint_path: Path de la ubicacion del checkpoint
        """
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                if checkpoint["training_step"] is not None:
                    self.current_step = checkpoint["training_step"]
                    print(f"Current step: {self.current_step}")
                if checkpoint["weights"] is not None:
                    pass
                    self.model.load_state_dict(checkpoint["weights"])
                if checkpoint["optimizer_state"] is not None:
                    pass
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                if checkpoint["episodes"] is not None:
                    self.current_ep = checkpoint["episodes"]
                    print(f"Current episode: {self.current_ep}")

    def optimizer_step(self, loss: Tensor):
        """
        Entrena al modelo en base a la perdida obtenida.
        """
        self.optimizer.zero_grad()

        loss.backward()

        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()


    def loss_function(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calcula la pérdida en base a las observaciones y recompensas obtenidas.
        """
        observations_target = (
            torch.tensor(np.array(self.buffer.observations))
            .float()
            .to(self.config.device)
        )
        actions_target = torch.tensor(self.buffer.actions).long().to(self.config.device)
        returns_target = (
            torch.tensor(self.buffer.returns).float().to(self.config.device)
        )
        advantage_target = (
            torch.tensor(self.buffer.advantages).float().to(self.config.device)
        )

        log_probs, values, entropy = self.model.evaluate_actions(
            observations_target, actions_target
        )
        values = values.flatten()

        # Actor
        if self.config.normalize_advantage:
            advantage_target = (advantage_target - advantage_target.mean()) / (
                advantage_target.std() + 1e-8
            )

        actor_loss = -(log_probs * advantage_target.detach()).mean()

        # Critic
        critic_loss = F.mse_loss(returns_target, values).mean() * self.config.vf_coef

        # Entropy
        if entropy is None:
            entropy_loss = -torch.mean(-log_probs)
        else:
            entropy_loss = -torch.mean(entropy)

        entropy_loss *= self.config.ent_coef

        # Loss
        loss = actor_loss + critic_loss + entropy_loss

        return (loss, actor_loss, critic_loss, entropy_loss)

    def reset_variables(self):
        """
        Inicializa o resetea las variables de training.
        """
        self.current_step = 0
        self.current_ep = 0

        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)

        self.ep_length = 0
        self.ep_reward = 0.0

    def train(self, greedy=False, checkpoint_path=None):
        """
        Loop principal de training del modelo.

        Args:
            greedy: Boolean que indica si el modelo debe recolectar la información siguiendo al algoritmo greedy.
            checkpoint_path: Path de la ubicacion del checkpoint
        """
        self.load_model(checkpoint_path)
        self.reset_variables()

        self.writer = SummaryWriter(self.log_dir)
        pbar = tqdm(self.check_end_condition(), desc="Steps")

        for _ in pbar:
            self.model.eval()

            if not greedy:
                self.collect_rollouts()
            else:
                self.collect_rollouts_greedy()

            self.model.train()

            (loss, actor_loss, critic_loss, entropy_loss) = self.loss_function()
            
            self.optimizer_step(loss)

            self.log_step(
                loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()
            )

            self.current_step += 1

            if self.current_step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            self.buffer.reset()

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy_loss": entropy_loss.item(),
                    "avg_reward": np.mean(self.ep_rewards),
                    "episodes": self.current_ep,
                }
            )

        self.writer.close()