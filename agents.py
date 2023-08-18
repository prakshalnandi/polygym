from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import random
from gym.spaces import Space
from gym.spaces.utils import flatdim
from typing import List, Dict, DefaultDict

from sklearn.preprocessing import normalize
import collections

#import pudb

class Agent(ABC):

    def __init__(
        self,
        action_space: Space,
        gamma: float,
        epsilon: float,
        num_weights: int,
        **kwargs
    ):

        self.action_space = action_space
        self.n_acts = flatdim(action_space)
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.q_table: DefaultDict = defaultdict(lambda: 0)
        #self.weights = np.zeros((4,6), dtype=float)
        self.weights = np.zeros((num_weights,6), dtype=float)


    @abstractmethod
    def choose_action(self):
        ...

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        self.epsilon = 1.0 - (min(1.0, timestep / (0.9 * max_timestep))) * 0.95
        print("epsilon value for current iteration :" , self.epsilon)
        #self.epsilon = 0.5
    #@abstractmethod
    #def update_q_table(self):
    #    ...


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

    def choose_action(self, obs, lstActs, blnTraining=True) -> int:

        ##find action with max value
        act_vals = [self.q_table[(obs, act)] for act in lstActs]
        max_val = max(act_vals)
        max_acts = [lstActs[idx] for idx, act_val in enumerate(act_vals) if act_val == max_val]

        if random.random() <= self.epsilon and blnTraining==True:
            return random.choice(lstActs)
        else:
            return random.choice(max_acts)
        return -1

class QLApproxAgent(Agent):

    def __init__(self, alpha: float, **kwargs):

        super().__init__(**kwargs)
        self.alpha: float = alpha

    def choose_action(self, obs, lstActs, blnTraining=True, blnExplore=False) -> int:

        #features = self.extract_features(obs)
        features = obs

        ##find action with max value
        act_vals = [self.get_q_value(obs, act) for act in lstActs]
        if not act_vals:
            max_val = 0
            max_acts = lstActs[0]
            print("error empty list: ", act_vals)
            print("obs: ", obs)
            print("features: ", features)
        else:
            max_val = max(act_vals)
            max_acts = [lstActs[idx] for idx, act_val in enumerate(act_vals) if act_val == max_val]

        randomNum = random.random()
        if randomNum  <= self.epsilon and blnTraining==True:
            return random.choice(lstActs)
        else:
            if max_acts == []:
                max_acts = [lstActs[0]]
            act_n = random.choice(max_acts)
            return act_n
        return -1

    
    def update_weights(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        
        #features = self.extract_features(obs)
        #n_features = self.extract_features(n_obs)
        features = obs
        n_features = n_obs
        ##find action with max value
        act_vals = [self.get_q_value(n_features, act) for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
        ##calculate target
        target_value = reward + self.gamma * (1 - done) * self.get_q_value(n_features, random.choice(max_acts))
        ##update q-table
        for i in range(len(features)):
            #if np.isnan(self.weights[i][action]) == 0:
            if  features[i] != None:
                diff = self.alpha * (target_value - self.get_q_value(features, action)) * features[i]
                if  np.isnan(diff) == False and np.isinf(diff) == False:
                    self.weights[(i, action)] += diff
                
        return self.q_table[(obs, action)]


    def get_q_value(
        self, obs: np.ndarray, action: int) -> float:
        features = obs
        q_val = 0.0
        for i in range(len(features)):
            if  features[i] != None:
                diff = features[i] * self.weights[i][action]
                if  np.isnan(diff) == False and np.isinf(diff) == False:
                    q_val = q_val + diff

        return q_val
    
    def extract_features(self, obs : np.ndarray):

        MAX_DEP = 41
        MAX_DIM = 10
        limits = np.array([MAX_DIM, MAX_DEP, MAX_DEP, MAX_DEP])
        return tuple(np.array(obs)/limits)
enumerate
