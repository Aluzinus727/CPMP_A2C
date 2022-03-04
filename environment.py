import random
from copy import deepcopy

from config import Config
from layout import Layout
from greedy import get_action_type


class Enviroment():
    def __init__(self, config: Config) -> None:
        self.S = config.nb_stacks
        self.H = config.height

        self.action_space = []
        self.mapped_act_space = {}
        act_space_elements = 0

        for i in range(config.nb_stacks):
            for j in range(config.nb_stacks):
                if i != j:
                    self.action_space.append((i, j))
                    self.mapped_act_space[(i, j)] = act_space_elements
                    act_space_elements += 1

    def reset(self, N: int = 20):
        """
        Se encarga de instanciar un nuevo Layout.
        """
        stacks = []
        for _ in range(self.S):
            stacks.append([])
        
        for _ in range(N):
            s=random.randint(0,self.S-1)
            while len(stacks[s])==self.H: s=s=random.randint(0,self.S-1)
            stacks[s].append(random.randint(1,N))
        
        self.layout = Layout(stacks, self.H)
        return self.layout.stacks

    def step(self, action):
        """
        Se encarga de realizar un paso en el Enviroment.
        Varía su recompensa en función del tipo de movimiento a realizar.
        """
        action = self.action_space[action]

        action_type = get_action_type(self.layout.stacks, action)
        if action_type == 'BG':
            reward = -1
        elif action_type == 'GG':
            reward = -2
        else: 
            reward = -3

        stack_source, stack_destination = action

        if not self.layout.verify(stack_source, stack_destination):
            return self.layout.stacks, -5, 0
        
        self.layout.move(stack_source, stack_destination)

        new_state = deepcopy(self.layout.stacks)
        done = 1

        for i in range(len(self.layout.sorted_stack)):
            if self.layout.sorted_stack[i] is False:
                done = 0
                break
                
        return new_state, reward, done

    # ========================================================================= #
    #                                Debug                                      #
    # ========================================================================= #
    
    def show_state(self):
        """
        Representación gráfica de un estado.
        """
        lay = [ fila + [0]*(self.H-len(fila)) for fila in self.layout.stacks ]

        for i in range(self.H-1, -1, -1):
            for j in range(len(lay)):
                print(lay[j][i], end=' ')
            print()
        print()