#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Playground: train-model.py

This script runs a training session for the model.
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

import os.path
import datetime
import numpy as np
import pandas as pd
import random


# ---------------------------------------------------------------------------
# Seek Game class
# ---------------------------------------------------------------------------

class SeekGame(object):
    """This class represents the environment (the board and the target and
    player positions)"""

    def __init__(self, width=4, height=4, tries=50, verbose=False):
        """Initialize all the board values."""

        # Board characteristics
        self.width = width
        self.height = height
        self.tries = tries
        self.left_tries = 0
        self.verbose = verbose

        # Values of the target and player possition over the board
        self.target_id = -1.0
        self.player_id = 1.0

        # Possitions
        self.player_x = 0
        self.player_y = 0
        self.target_x = 0
        self.target_y = 0

        # A cummulative penalty for hitting the walls
        self.soft_penalty = 0.0

    def reset(self):
        """Clean penalty, reset tries, and set the player and target on
        random possition"""

        # Reset the left tries and clean the penalty
        self.left_tries = self.tries
        self.soft_penalty = 0.0

        # A random point over the board
        self.target_x = random.randint(0, self.width - 1)
        self.target_y = random.randint(0, self.height - 1)

        # Look for a free point (avoiding the target possition)
        found_point = False
        while not found_point:
            x_point = random.randint(0, self.width - 1)
            y_point = random.randint(0, self.height - 1)
            if self.target_x != x_point and self.target_y != y_point:
                found_point = True

        self.player_x = x_point
        self.player_y = y_point

    def move(self, direction):
        """Move the player."""

        # Each movement is a new try
        self.left_tries -= 1
        dir_icon = "?"

        if direction == 0:
            # Up
            dir_icon = "^"
            if self.player_x >= 0:
                self.player_x -= 1
            else:
                # Penalizamos un poco si se choca con los bordes
                self.soft_penalty -= 0.2
        elif direction == 1:
            # Right
            dir_icon = ">"
            if self.player_y < self.width - 1:
                self.player_y += 1
            else:
                # Penalizamos un poco si se choca con los bordes
                self.soft_penalty -= 0.2
        elif direction == 2:
            # Down
            dir_icon = "v"
            if self.player_x < self.height - 1:
                self.player_x += 1
            else:
                # Penalizamos un poco si se choca con los bordes
                self.soft_penalty -= 0.2
        elif direction == 3:
            # Left
            dir_icon = "<"
            if self.player_y >= 0:
                self.player_y -= 1
            else:
                # Penalizamos un poco si se choca con los bordes
                self.soft_penalty -= 0.2

        if verbose:
            print(f"{dir_icon}", end="")

    def get_board(self):
        """Returns the board matrix: empty spaces with a 0.0 value and
        the player and target with specific values."""

        # The empty board
        board = np.zeros((self.width, self.height))

        # The target
        board[self.target_x, self.target_y] = self.target_id

        # And the player
        board[self.player_x, self.player_y] = self.player_id

        return board

    def get_board_vector(self):
        """Returns the board in a vertor form."""
        return self.get_board().reshape((1, -1))

    def get_reward(self):
        """This method returns the current reward: '1.0' if the target is
        reached, '-1.0' if there's no more left tries or '0.0' (plus the
        off-board penalties) in any other case."""

        # No more left tries
        if self.left_tries == 0:
            return -1.0

        # Target found
        if self.player_x == self.target_x and self.player_y == self.target_y:
            return 1.0

        # ...else... the soft-penalty
        return self.soft_penalty

    def is_over(self):
        """Returns 'True' if the game is over."""

        # No more tries or target found
        if self.get_reward() >= 1.0 or self.left_tries == 0:
            return True

        # ...too much penalties...
        elif self.get_reward() <= -1.0:
            return True

        # Else: the game is running
        else:
            return False


class ExperienceReplay(object):
    """This class represents the 'memory' of the game (past movements,
    rewards and board situations."""

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


class ModelBuilder(object):
    """This class hides the model build."""

    def __init__(self, width=4, height=4, file_name="model_weights.h5"):
        """Initialization: properties of the model"""

        self.epsilon = .1  # exploration
        self.num_actions = 4  # [up, right, down, left]
        self.hidden_size = 100
        self.model_filename = file_name

        self.model = Sequential()
        self.model.add(Dense(self.hidden_size,
                             input_shape=(width * height,),
                             activation='relu'))
        self.model.add(Dense(self.hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions))
        self.model.compile(sgd(lr=.2), "mse")

    def load_weights(self):
        """Load the model weights if the file exists"""

        # Look for the weights file
        if os.path.exists(self.model_filename):
            self.model.load_weights(self.model_filename)

    def save_weights(self):
        """Save the model weights to a file"""

        self.model.save_weights(self.model_filename)

    def get_real_model(self):
        """Returns the model"""
        return self.model

    def next_action(self):
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
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Session parameters
    epoch = 100
    batch_size = 50
    verbose = True
    statistics_file = "stats.csv"

    # Build the game
    the_game = SeekGame(verbose=verbose)
    the_game.reset()

    # Build the model
    print("Building the model...")
    model = ModelBuilder(the_game.width, the_game.height)
    model.load_weights()
    print("Model ready.")

    # Build the memory
    exp_replay = ExperienceReplay()

    # Reset the winning count
    win_cnt = 0

    # Train loop (over the epochs)
    for e in range(epoch):
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
            # print("--- Tablero inicial:")
            # x = the_game.get_board()
            # print("{}".format(x))

            input_tm1 = input_t

            # Get the next action
            action = model.next_action()

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

        has_won = the_game.get_reward() >= 1.0
        if verbose:
            if has_won:
                print(f" WIN  ({the_game.get_reward()})")
            else:
                print(f" LOSE ({the_game.get_reward()})")
        print(
            f"Epoch {e:03d}/{epoch:03d} | Loss {loss:.4f} | win={has_won:d} | Win count {win_cnt}")

    print("----")

    # Save the model
    model.save_weights()

    # And the statistics
    if os.path.exists(statistics_file):
        data = pd.read_csv(statistics_file)
    else:
        data = pd.DataFrame(columns=["Index", "WinCount", "WinPct", "Loss"])

    # Creates a new stat entry
    entry = {
        "Index": data.shape[0],
        "WinCount": win_cnt,
        "WinPct": win_cnt/epoch,
        "Loss": loss
    }

    # Add the entry
    data = data.append(entry, ignore_index=True)

    # Save the data
    data.to_csv(statistics_file, index=False)
