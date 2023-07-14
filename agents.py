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

##    def choose_action(self, obs, lstActs, blnTraining=True) -> int:
##
##       ##find action with max value
##        act_vals = [self.q_table[(obs, act)] for act in lstActs]
##        #print("act_vals while choosing actions", act_vals)
##        #print("choice act_vals: ", act_vals)
##        max_val = max(act_vals)
##        #print("max_val: ", max_val)
##        max_acts = [lstActs[idx] for idx, act_val in enumerate(act_vals) if act_val == max_val]
##        #print("max_acts: ", max_acts)
##
##        if(blnTraining == False):
##            #pudb.set_trace()
##
##            self.epsilon = 1
##        else:
##            self.epsilon = 0.5
##        if random.random() >= self.epsilon and blnTraining==True:
##            #print("in random: ", lstActs)
##            return random.choice(lstActs)
##        else:
##            #print("in max_axts: ", max_acts)
##            return random.choice(max_acts)
##        return -1

    @abstractmethod
    def choose_action(self):
        ...

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        self.epsilon = 1.0 - (min(1.0, timestep / (0.9 * max_timestep))) * 0.95
        #print("epsilon value for current iteration :" , self.epsilon)
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

        #if(blnTraining == False):
        #    #pudb.set_trace()
        #    self.epsilon = 1
        #else:
        #    self.epsilon = 0.5
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

        #print("Choose act features: ", features)
        #print("Choose act weights: ", self.weights)
        ##find action with max value
        act_vals = [self.get_q_value(obs, act) for act in lstActs]
        if not act_vals:
            max_val = 0
            if blnExplore:
                max_acts = [3]
                lstActs = [3]
            else:
                max_acts = [1]
                lstActs = [1]
            #max_acts = lstActs[0]
            print("error empty list: ", act_vals)
            print("obs: ", obs)
            print("features: ", features)
        else:
            max_val = max(act_vals)
            max_acts = [lstActs[idx] for idx, act_val in enumerate(act_vals) if act_val == max_val]

        #if(blnTraining == False):
        #    #pudb.set_trace()
        #    self.epsilon = 1
        #else:
        #    self.epsilon = 0.5
        if random.random() <= self.epsilon and blnTraining==True:
            if(blnExplore):
                #print("lstActs: ", lstActs)
                p = [0.7, 0.15, 0.15]
                return random.choices(lstActs, weights=p)[0]
            else:
                return random.choice(lstActs)
            #return random.choice(lstActs)
        else:
            if max_acts == []:
                max_acts = [lstActs[0]]
                #print("act_vals: ", act_vals)
                #print("lstActs :", lstActs)
            #print("max_acts : ", max_acts)
            act_n = random.choice(max_acts)
            if blnTraining == False:
                print("act_vals_out: ", act_vals)
                print("chosen action : ",act_n)
            return act_n
        return -1

    
    def update_weights(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        
        #features = self.extract_features(obs)
        #n_features = self.extract_features(n_obs)
        #print("Inside update_weights : ")
        features = obs
        n_features = n_obs
        ##find action with max value
        act_vals = [self.get_q_value(n_features, act) for act in range(self.n_acts)]
        max_val = max(act_vals)
        max_acts = [idx for idx, act_val in enumerate(act_vals) if act_val == max_val]
        #print("max_acts : ", max_acts)
        ##calculate target
        target_value = reward + self.gamma * (1 - done) * self.get_q_value(n_features, random.choice(max_acts))
        #print("target_value : ", target_value)
        ##update q-table
        for i in range(len(features)):
            #if np.isnan(self.weights[i][action]) == 0:
            if  features[i] != None:
                #print("weight for ", i, "is ",self.weights[i])
                diff = self.alpha * (target_value - self.get_q_value(features, action)) * features[i]
                if  np.isnan(diff) == False and np.isinf(diff) == False:
                    self.weights[(i, action)] += diff
                
                ##self.weights[(i, action)] += self.alpha * (
                ##        target_value - self.get_q_value(features, action)
                ##) * features[i]

        #print("weights: ", self.weights)
        #print("features: ", features)
        #self.weights = normalize(self.weights, axis=1)
        return self.q_table[(obs, action)]

    #def normalise_weights(self):
    #    for action in self.n_acts:
            

    def get_q_value(
        self, obs: np.ndarray, action: int) -> float:
        #print("Inside get_q_value : ")
        #features = self.extract_features(obs)
        features = obs
        #print("q features: ", features)
        #print("weights: ", self.weights)
        q_val = 0.0
        for i in range(len(features)):
            #print("i : ", i)
            #print("action : ", action)
            #print("feature : ",features[i])
            if  features[i] != None:
                diff = features[i] * self.weights[i][action]
                if  np.isnan(diff) == False and np.isinf(diff) == False:
                #print("feature : ", features[i])
                #print("q_val : ", q_val)
                #print("weight :",self.weights[i][action])
                ##q_val = q_val + features[i] * self.weights[i][action]
                    q_val = q_val + diff

        return q_val
    
    def extract_features(self, obs : np.ndarray):

        MAX_DEP = 41
        MAX_DIM = 10
        limits = np.array([MAX_DIM, MAX_DEP, MAX_DEP, MAX_DEP])
        #print("obs ex : ", np.array(obs))
        #print("limits ex : ", limits)
        return tuple(np.array(obs)/limits)

