from abc import ABC, abstractmethod, abstractstaticmethod
from copy import deepcopy
from itertools import accumulate

import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from deep_networks import FCNetwork
import torch.nn.functional as F
from torch.nn.modules.activation import LogSoftmax

import random
from gym.spaces.utils import flatdim

class Agent(ABC):
    #Base class


    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        **kwargs,
    ):

        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.size

        # ######################################### #
        # ######################################### #
        # self.critics_net = FCNetwork(
        #     (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.ReLU
        # )

        self.critics_net = FCNetwork(
             (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )

        # ############################################# #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.epsilon = 0.65
        # ######################################### #

        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        self.epsilon = 1.0 - (min(1.0, timestep / (0.9 * max_timestep))) * 0.95
        #self.epsilon = 0.5
        #raise NotImplementedError("Needed for Q3")

    def act(self, obs: np.ndarray, lstActs, explore: bool):

        with torch.no_grad():
            arrAllActs = [0, 1, 2]
            arrAllowedAct = np.array(lstActs)
            arrNetwork = self.critics_target(torch.from_numpy(obs).float()).numpy()
            setDiff = np.setdiff1d(arrAllActs, arrAllowedAct)
            arrNetwork[setDiff] = np.nan

            if(explore):
                if random.random() <= self.epsilon:
                    return random.choice(lstActs)
                else:
                    #return np.argmax(self.critics_target(torch.from_numpy(obs).float())).numpy()
                    return np.nanargmax(arrNetwork)
            else:
                #return np.argmax(self.critics_target(torch.from_numpy(obs).float())).numpy()
                return np.nanargmax(arrNetwork)


    def update(self, batch, reward) -> Dict[str, float]:

        self.critics_optim.zero_grad()
        ## calculate q prediction
        predicted_q = torch.gather(self.critics_net(torch.tensor(batch.states, dtype=torch.float32)), 1, torch.tensor(batch.actions))
        with torch.no_grad():
            ##calculate target
            #cal_target = batch.rewards.squeeze().detach() + self.gamma * (self.critics_target(batch.next_states).max(dim=1)[0]) * (1-batch.done).squeeze()
            cal_target = reward + self.gamma * (self.critics_target(torch.tensor(batch.next_states, dtype=torch.float32)).max(dim=1)[0]) * (1-batch.done).squeeze()

        ## loss mean
        q_loss = torch.mean(((cal_target - predicted_q.squeeze())**2))
        q_loss.backward()

        ##update network
        self.critics_optim.step()

        ##update target network
        self.update_counter += 1
        if self.update_counter >= self.target_update_freq :
            self.critics_target.hard_update(self.critics_net)
            self.update_counter = 0

        return {"q_loss": q_loss.detach().numpy()}



