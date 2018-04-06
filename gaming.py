#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Reinforcement Learning Playground

This module contains the gaming related classes.
"""

import numpy as np
import random


# ---------------------------------------------------------------------------
# Seek Game class
# ---------------------------------------------------------------------------

class SeekGame(object):
    """This class represents the environment (the board with the target and
    player positions) of a 'seek' game."""

    def __init__(self, width=4, height=4, tries=50):
        """Initialize all the board values."""

        # Board characteristics
        self.width = width
        self.height = height
        self.tries = tries
        self.left_tries = 0

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

        # Player movements (for representation)
        self.movements = []

    def reset(self):
        """Clean penalty, reset tries, and set the player and target on
        random possition"""

        # Reset the left tries and clean the penalty
        self.left_tries = self.tries
        self.soft_penalty = 0.0
        self.movements = []

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
        """Move the player. A 'soft penalty' is applied when hitting the
        borders."""

        # Each movement is a new try
        self.left_tries -= 1
        dir_icon = "?"

        self.movements.append(direction)

        if direction == 0:
            # Up
            if self.player_x >= 0:
                self.player_x -= 1
            else:
                # Border hit
                self.soft_penalty -= 0.2
        elif direction == 1:
            # Right
            if self.player_y < self.width - 1:
                self.player_y += 1
            else:
                # Border hit
                self.soft_penalty -= 0.2
        elif direction == 2:
            # Down
            if self.player_x < self.height - 1:
                self.player_x += 1
            else:
                # Border hit
                self.soft_penalty -= 0.2
        elif direction == 3:
            # Left
            if self.player_y >= 0:
                self.player_y -= 1
            else:
                # Border hit
                self.soft_penalty -= 0.2

    def get_movements(self):
        """Retuns the payer's past movements for representation."""
        return self.movements

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
