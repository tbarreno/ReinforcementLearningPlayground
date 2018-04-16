#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Reinforcement Learning Playground

This script trains the models (with diferent board sizes) over 10 episodes
(each one with several epochs).
"""

import os.path

import learning

# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Train the model with 10 episodes and 100 epochs each
    episodes = 10
    epochs = 100

    # Board size
    # for board_size in range(4, 8):
    for board_size in range(5, 6):
        for episode in range(1, episodes+1):
            print(f"=== Size {board_size}x{board_size} | "
                  f"Episode {episode:03d}/{episodes:03d} ===")

            episode_name=f"{board_size}x{board_size}"

            learning.run_episode(name=episode_name,
                                 epochs=epochs,
                                 board_height=board_size,
                                 board_width=board_size)
