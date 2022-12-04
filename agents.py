from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from gym.spaces import Space
from gym.spaces.utils import flatdim
from typing import List, Dict, DefaultDict

import collections

#import pudb

class Agent(ABC):

    def __init__(
        self,
        action_space: Space,
        gamma: float,
        epsilon: float,
        **kwargs
    ):

        self.action_space = action_space
        self.n_acts = flatdim(action_space)
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.q_table: DefaultDict = defaultdict(lambda: 0)

    def choose_action(self, obs, lstActs, blnTraining=True) -> int:

        ##find action with max value
        act_vals = [self.q_table[(obs, act)] for act in lstActs]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
        
        if(blnTraining == False):
            #pudb.set_trace()
            print("Testing -> Agent")
        if random.random() < self.epsilon and blnTraining==True:
            return random.choice(lstActs)
        else:
            return random.choice(max_acts)
        return -1


    @abstractmethod
    def update_q_table(self):
        ...


class QLearningAgent(Agent):

    def __init__(self, alpha: float, **kwargs):

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def update_q_table(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        
        ##find action with max value
        act_vals = [self.q_table[(n_obs, act)] for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]

        ##calculate target
        target_value = reward + self.gamma * (1 - done) * self.q_table[(n_obs, random.choice(max_acts))]

        ##update q-table
        self.q_table[(obs, action)] += self.alpha * (
                target_value - self.q_table[(obs, action)]
        )

        return self.q_table[(obs, action)]


