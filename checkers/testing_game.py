#!/usr/bin/python

from game import Game
from human_player import make_human_player
from random_player import make_random_player

player1 = make_random_player("1") #make_human_player("1")
player2 = make_random_player("2") #make_human_player("2")

for i in range(10):
    game = Game(player1, player2)
    winner = game.run()

    if winner == 0:
        print("The game was a draw")
    elif winner == 1:
        print("Player 1 won!")
    elif winner == 2:
        print("Player 2 won!")

    print("Game %s finished" % i)
