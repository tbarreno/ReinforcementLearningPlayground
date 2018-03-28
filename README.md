# Q-Learning Playground project

## Introduction

This project contains few tests and examples about **Reinforcement Learning**
using the **Keras** library.

It's heavily based on the
[Keras plays Catch](http://edersantana.github.io/articles/keras_rl/)
([GitHub repository](https://github.com/EderSantana/KerasPlaysCatch)) project
from [Eder Santana](http://edersantana.github.io/). This project is
simply enough for understanding **Q-Learning** and try some variations.

## First step: changing the game

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

