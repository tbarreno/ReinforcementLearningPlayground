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
import time

import gaming


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

    def __init__(self, name="", width=4, height=4, weights_directory="model_weights"):
        """Initialization: properties of the model"""

        self.epsilon = .1  # exploration
        self.num_actions = 4  # [up, right, down, left]
        self.hidden_size = 100
        self.name=name

        self.model = Sequential()
        self.model.add(Dense(self.hidden_size,
                             input_shape=(width * height,),
                             activation='relu'))
        self.model.add(Dense(self.hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=.2), "mse")

        self.weights_directory = weights_directory

        self.model_file = os.path.join(weights_directory,
                                       f"model_weights{name}.h5")

    def load_weights(self):
        """Load the model weights if the file exists"""

        # Look for the weights file
        if os.path.exists(self.model_file):
            self.model.load_weights(self.model_file)

    def save_weights(self):
        """Save the model weights to a file"""

        # Directory creation
        if not os.path.exists(self.weights_directory):
            os.mkdir(self.weights_directory)

        # Save the model weights
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


# ---------------------------------------------------------------------------
# Episode method
# ---------------------------------------------------------------------------

def run_episode(name="", epochs=100, board_height=4, board_width=4,
                verbose=False):
    """Run an episode (several epochs)"""

    # Episode parameters
    batch_size = 50

    # Stats directory
    stats_directory = "stats"

    # Build the game
    the_game = gaming.SeekGame(board_height, board_width)
    the_game.reset()

    # Build the model
    print("Building the model...")
    model = ModelBuilder(name, the_game.width, the_game.height)
    model.load_weights()
    print("Model ready.")

    # Build the memory
    exp_replay = ExperienceReplay()

    # Reset the winning count
    win_cnt = 0

    # Start time
    start_time = time.time()

    # Train loop (over the epochs)
    for e in range(epochs):
        loss = 0.0
        the_game.reset()
        game_over = False

        if verbose:
            print("--- Board representation:")
            x = the_game.get_board()
            print(f"{x}")

        # Get initial input
        input_t = the_game.get_board_vector()

        while not game_over:
            # print("--- Board ---")
            # x = the_game.get_board()
            # print("{}".format(x))

            input_tm1 = input_t

            # Get the next action
            action = model.next_action(input_tm1)

            # apply action, get rewards and new state
            # Move the player on the board
            the_game.move(action)

            # Get the resulting board, reward and if the game is over
            input_t = the_game.get_board_vector()
            reward = the_game.get_reward()
            game_over = the_game.is_over()

            # Have we won?
            if reward >= 1.0:
                win_cnt += 1

            # Remember this movement
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # Adapt model: train a memory batch
            inputs, targets = exp_replay.get_batch(model.get_real_model(),
                                                   batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)

        # End time
        end_time = time.time()
        elapsed_time = end_time - start_time

        has_won = the_game.get_reward() >= 1.0

        # Just for testing...
        if verbose:
            if has_won:
                print(f" WIN  ({the_game.get_reward()})")
            else:
                print(f" LOSE ({the_game.get_reward()})")

        # Episode summary
        print(f"Epoch {(e+1):03d}/{epochs:03d} | Loss {loss:.4f} |"
              f" win={has_won:d} | Win count {win_cnt}")

    print("----")
    print(f"Elapsed time {elapsed_time:.3f}")

    # Save the model
    model.save_weights()

    # Directory
    if not os.path.exists(stats_directory):
        os.mkdir(stats_directory)

    statistics_file = os.path.join(stats_directory, f"stats{name}.csv")

    # Load previous data (or create an empty dataset)
    if os.path.exists(statistics_file):
        with open(statistics_file) as s_f:
            data = s_f.readlines()
    else:
        data = [f"Episode,Epochs,WinCount,WinPct,Loss,Time\n"]

    # Creates a new stat entry
    data.append(f"{len(data):d},{epochs:d},{win_cnt:d},"
                f"{(win_cnt / epochs):.3f},{loss:.6f},{elapsed_time:.3f}\n")

    # Save the data
    with open(statistics_file, "w") as s_f:
        s_f.writelines(data)

