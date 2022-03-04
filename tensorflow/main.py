import random
import numpy as np
import utils
import copy
from layout import Layout
from greedy import greedy_solve
from a2c import A2C
from model import *


def compute_unsorted_elements(stacks):
    unsorted_elements = []

    for stack in stacks:
        if len(stack) == 0: 
            unsorted_elements.append(0)
            continue
        
        sorted_elements = 1
        while(sorted_elements<len(stack) and stack[sorted_elements] <= stack[sorted_elements-1]):
            sorted_elements +=1
        
        unsorted_elements.append(len(stack) - sorted_elements)
    
    return unsorted_elements

def create_layout(n_stacks, height, n):
    stacks = []
    for i in range(n_stacks):
        stacks.append([])
    
    for j in range(n):
        s=random.randint(0,n_stacks-1)
        while len(stacks[s])==height: s=s=random.randint(0,n_stacks-1)
        stacks[s].append(random.randint(1,n))
    
    return Layout(stacks, height)


def generate_layout_info(nb_stacks=7, height=7, containers=20, n_layouts=12):
    observation_memory = []
    state_values_memory = []
    new_observation_memory = []
    action_memory = []
    reward_memory = []
    done_memory = []

    actions = {
        (0, 1): 0,
        (0, 2): 1,
        (0, 3): 2,
        (0, 4): 3,
        (1, 0): 4,
        (1, 2): 5,
        (1, 3): 6,
        (1, 4): 7,
        (2, 0): 8,
        (2, 1): 9,
        (2, 3): 10,
        (2, 4): 11,
        (3, 0): 12,
        (3, 1): 13,
        (3, 2): 14,
        (3, 4): 15,
        (4, 0): 16,
        (4, 1): 17,
        (4, 2): 18,
        (4, 3): 19
    }

    for _ in range(n_layouts):
        layout = create_layout(nb_stacks, height, containers)
        layout_original = copy.deepcopy(layout)
        moves = greedy_solve(layout)
        max_moves = len(moves)

        for i, move in enumerate(moves):
            source, destination = move

            state_value = max_moves
            state_values_memory.append(-(state_value))
            max_moves -= 1

            observation = copy.deepcopy(layout_original.stacks)
            observation.append(compute_unsorted_elements(observation))
        
            observation_memory.append(observation)

            action_memory.append(actions[move])
            reward_memory.append(-1)

            layout_original.move(source, destination)

            new_observation = copy.deepcopy(layout_original.stacks)
            new_observation.append(compute_unsorted_elements(new_observation))
            new_observation_memory.append(new_observation)

            done_memory.append(0)

    return observation_memory, new_observation_memory, action_memory, reward_memory, done_memory, state_values_memory


if __name__ == '__main__':    
    actor_file = 'tensorflow/models/actor-model.h5'
    critic_file = 'tensorflow/models/critic-model.h5'

    # Creación de modelos.
    # Si se desean cargar en vez de crear, cambiar la documentación de lineas de crear a cargar.
    # critic = keras.models.load_model(critic_file)
    # actor = keras.models.load_model(actor_file)
    critic = create_critic_model() 
    actor = create_actor_model()

    critic.summary()
    actor.summary()
    
    trainer = A2C(
        actor, 
        critic,
        actor_file,
        critic_file
    )

    for _ in range(5):
        samples = generate_layout_info(
            nb_stacks=5,
            height=7,
            containers=20,
            n_layouts=25000
        )

        trainer.train_greedy(samples)
