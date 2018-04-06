# Reinforcement-Learning Playground project

## Introduction

This project contains few tests and examples about **Reinforcement Learning**
using the **Keras** library.

It's heavily based on the
[Keras plays Catch](http://edersantana.github.io/articles/keras_rl/)
([GitHub repository](https://github.com/EderSantana/KerasPlaysCatch)) project
from [Eder Santana](http://edersantana.github.io/). This project is
simply enough for understanding **Q-Learning** and try some variations.

## The project structure

The `gaming.py` module contains just a class that represents the board with the
player and target position. This class also is responsible for calculating the
reward for each movement.

The game creation only requires the size of the board (a 4x4 board as default).

Having the game in a independent class allows us to change the gaming behaviour
independently from the learning part.

The `learning.py` module have the model build and the *experience replay*
classes.

Finally, `train-model.py` is an executable script that builds the module and
run a 100 epochs episode. It saves the trained model and loads it before
training if it exists. It also saves a CSV file with the training statistics.

## Seek: the game

In [Keras plays Catch](http://edersantana.github.io/articles/keras_rl/), the
game consists in catching "*falling fruit*" with a basket that can move along
the floor (just left and right).

I've changed the game to an "*Seek game*": there's a "target" over the board
and the player must move toward it (moving up, down, left or right).

The player is rewarded when it achieves the target and is *punished* if it
tries to leave the board.

## References and interesting links

  - [Demystifying Deep Reinforcement Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
  - [Artificial Intelligence: Q-learning](http://artint.info/html/ArtInt_265.html)

