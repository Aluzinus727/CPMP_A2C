import copy
import numpy as np

from utils import prepare, compute_unsorted_elements

class A2C:
    def __init__(
        self, 
        actor, 
        critic,
        actor_file,
        critic_file
    ) -> None:
        self.actor = actor
        self.critic = critic

        self.actor_file = actor_file
        self.critic_file = critic_file


    def one_hot_encoding(self, action, n_actions):
        encoded = np.zeros(n_actions, np.float32)
        encoded[action] = 1
        
        return encoded

    def _train_actor_greedy(self, samples, discount_factor=1):
        observations_prepared = []
        encoded_actions = []

        observations, new_observations, actions, rewards, dones, values = samples

        for i in range(len(observations)):
            observation = observations[i]
            action = actions[i]
            action = self.one_hot_encoding(action, 20)

            observation = prepare(observation, 7)
            observations_prepared.append(observation)

            encoded_actions.append(action)

        observations_prepared = np.array(observations_prepared)
        encoded_actions = np.array(encoded_actions)

        history = self.actor.fit(observations_prepared, encoded_actions, epochs=5, batch_size=256, verbose=False)
        self.actor.save(self.actor_file)

        print(f"Actor loss: {history.history['loss'][0]}")

    def _train_critic_greedy(self, samples, discount_factor=1):
        observations_prepared = []
        TD_targets = []

        observations, new_observations, _, rewards, dones, values = samples

        for i in range(len(observations)):
            observation = observations[i]
            reward = rewards[i] 
            done = dones[i]

            if i == len(observations) - 1:
                value_next = 0
            else:
                value_next = values[i + 1]

            TD_target = reward + (1 - done) * discount_factor * value_next
            TD_targets.append(TD_target)

            observation = prepare(observation, 7)
            observations_prepared.append(observation)


        observations_prepared = np.array(observations_prepared)
        TD_targets = np.array(TD_targets)

        history = self.critic.fit(observations_prepared, TD_targets, epochs=2, batch_size=256, verbose=False)
        self.critic.save(self.critic_file)

        print(f"Critic loss: {history.history['loss'][0]}")

    def train_actor(self, observation, new_observation, action, reward, done, discount_factor=1):
        observation.append(compute_unsorted_elements(observation))
        observation = prepare(observation, 7)
        observation = np.array([observation])

        new_observation.append(compute_unsorted_elements(new_observation))
        new_observation = prepare(new_observation, 7)
        new_observation = np.array([new_observation])

        encoded_action = self.one_hot_encoding(action, 20)
        action_probs = self.actor.predict(observation)
        
        value_curr = self.critic.predict(observation)
        next_value = self.critic.predict(new_observation)
        
        TD_target = reward + (1 - done) * discount_factor * next_value
        advantage = TD_target - value_curr
        advantage_reshaped = np.vstack([advantage])

        gradient = encoded_action - action_probs
        gradient_with_advantage = .0001 * gradient * advantage_reshaped + action_probs
        gradient_with_advantage = np.array(gradient_with_advantage)

        history = self.actor.fit(observation, gradient_with_advantage, epochs=1, batch_size=256, verbose=False)
        self.actor.save(self.actor_file)

        print(f"Actor loss: {history.history['loss'][0]}")
        return history.history['loss'][0]

    def train_critic(self, observation, new_observation, reward, done, discount_factor=1):
        observation = copy.deepcopy(observation)
        new_observation = copy.deepcopy(new_observation)

        new_observation.append(compute_unsorted_elements(new_observation))
        new_observation_reshaped = np.array([prepare(new_observation, 7)])
        next_value = self.critic.predict(new_observation_reshaped)
        
        TD_target = reward + (1 - done) * discount_factor * next_value
        
        observation.append(compute_unsorted_elements(observation))
        observation = np.array([prepare(observation, 7)])
        TD_target = np.array([TD_target])
        
        history = self.critic.fit(observation, TD_target, epochs=1, batch_size=256, verbose=False)
        self.critic.save(self.critic_file)

        print(f"Critic loss: {history.history['loss'][0]}")
        return history.history['loss'][0]

    def train_greedy(self, samples):
        self._train_critic_greedy(samples)
        self._train_actor_greedy(samples)

    def train(self, observation, new_observation, action, reward, done):
        critic_loss = self.train_critic(observation, new_observation, reward, done)
        actor_loss = self.train_actor(observation, new_observation, action, reward, done)

        return critic_loss, actor_loss