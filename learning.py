#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Reinforcement Learning Playground

This module contains the model builder and the 'Experience Replay' class.
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

import numpy as np
import os.path


# ---------------------------------------------------------------------------
# Experience Replay class
# ---------------------------------------------------------------------------

class ExperienceReplay(object):
    """This class represents the 'memory' of the game (past movements,
    rewards and board situations)."""

    def __init__(self, max_memory=500, discount=.9):
        """Initialization"""
        self.max_memory = max_memory
        self.discount = discount

        # The memory is just a list
        self.memory = list()

    def remember(self, states, game_is_over):
        """Remember a movement."""
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_is_over])

        # If the list is "full", forget the first memory
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        """Build a memory batch: it takes 'batch_size' elements from the
        memory (randomly chosen) in order to train the model."""

        # Sizes
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]

        # Initialize the inputs and rewards
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        # Fill the batch with randomly selected memories
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):

            # Retrieve the memory
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            # Sets the batch element state (input)
            inputs[i:i + 1] = state_t

            # "There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken deep"
            targets[i] = model.predict(state_t)[0]

            # If the game is over: just take the reward
            if game_over:
                targets[i, action_t] = reward_t

            # else: the reward should be "estimated"
            else:
                # Predict the reward of the next state
                q_sa = np.max(model.predict(state_tp1)[0])

                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * q_sa

        return inputs, targets


# ---------------------------------------------------------------------------
# Model Builder class
# ---------------------------------------------------------------------------

class ModelBuilder(object):
    """This class hides the model build."""

    def __init__(self, width=4, height=4, weights_directory="model_weights"):
        """Initialization: properties of the model"""

        self.epsilon = .1  # exploration
        self.num_actions = 4  # [up, right, down, left]
        self.hidden_size = 100

        self.model = Sequential()
        self.model.add(Dense(self.hidden_size,
                             input_shape=(width * height,),
                             activation='relu'))
        self.model.add(Dense(self.hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=.2), "mse")

        self.model_file = os.path.join(weights_directory,
                                       f"model_weights-{height}x{width}.h5")

    def load_weights(self):
        """Load the model weights if the file exists"""

        # Look for the weights file
        if os.path.exists(self.model_file):
            self.model.load_weights(self.model_file)

    def save_weights(self):
        """Save the model weights to a file"""
        self.model.save_weights(self.model_file)

    def get_real_model(self):
        """Returns the model"""
        return self.model

    def next_action(self, input_tm1):
        """Calculate the next action."""

        # Once every... "epsilon", we will chose a random movement
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.num_actions - 1, size=1)

        # For the rest: just predict the reward probability of each movement
        # and choose the one with the higher value
        else:
            q = self.model.predict(input_tm1)
            action = np.argmax(q[0] - 1)

        return action

    def train_on_batch(self, inputs, targets):
        """Just run the model's 'train_on_batch'..."""

        return self.model.train_on_batch(inputs, targets)
