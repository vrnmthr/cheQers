#!/usr/bin/python

from game import Game
from human_player import make_human_player

game = Game(make_human_player("1"), make_human_player("2"))
game.run()
print("Game finished")
