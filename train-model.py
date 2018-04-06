#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Reinforcement Learning Playground

This script runs a single training episode (session).
"""

import os.path

import gaming
import learning

# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Episode parameters
    epochs = 100
    batch_size = 50
    verbose = False

    # Board size
    board_height = 4
    board_width = 4

    # Build the game
    the_game = gaming.SeekGame(board_height, board_width)
    the_game.reset()

    # Build the model
    print("Building the model...")
    model = learning.ModelBuilder(the_game.width, the_game.height)
    model.load_weights()
    print("Model ready.")

    # Build the memory
    exp_replay = learning.ExperienceReplay()

    # Reset the winning count
    win_cnt = 0

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
            # print("--- Tablero inicial:")
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

        has_won = the_game.get_reward() >= 1.0
        if verbose:
            if has_won:
                print(f" WIN  ({the_game.get_reward()})")
            else:
                print(f" LOSE ({the_game.get_reward()})")
        print(f"Epoch {(e+1):03d}/{epochs:03d} | Loss {loss:.4f} |"
              f" win={has_won:d} | Win count {win_cnt}")

    print("----")

    # Save the model
    model.save_weights()

    # And the statistics
    statistics_file = os.path.join("stats",
                                   f"stats-{board_height}x{board_width}.csv")

    if os.path.exists(statistics_file):
        with open(statistics_file) as s_f:
            data = s_f.readlines()
    else:
        data = [f"Episode,Epochs,WinCount,WinPct,Loss\n"]

    # Creates a new stat entry
    data.append(f"{len(data):d},{epochs:d},{win_cnt:d},"
                f"{(win_cnt / epochs):.3f},{loss:.6f}\n")

    # Save the data
    with open(statistics_file, "w") as s_f:
        s_f.writelines(data)
