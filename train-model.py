#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Reinforcement Learning Playground

This script runs a several training episodes with different board sizes.
"""

import os.path

import learning

# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Launch 4 runs with 10 episodes and 100 epochs each
    runs = 4
    episodes = 10
    epochs = 10

    # Board size
    for board_size in range(4, 8):
        for run in range(1, runs+1):
            for episode in range(1, episodes+1):
                print(f"=== Size {board_size}x{board_size} | Run {run} | "
                      f"Episode {episode:03d}/{episodes:03d} ===")

                episode_name=f"{board_size}x{board_size}-r{run:02d}"

                learning.run_episode(name=episode_name,
                                     epochs=epochs,
                                     board_height=board_size,
                                     board_width=board_size)
