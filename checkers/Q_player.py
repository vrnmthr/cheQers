import os
from game import Game
from checkers_learner import CheQer
from human_player import make_human_player

checkers_ai = CheQer(.7, .1, [35, 15])

train = True
player1 = checkers_ai.step if train else make_human_player("Human")
player2 = checkers_ai.step

try:
    for i in range(1000000):
            print("Winner of game ", i, ": ", Game(player1, player2).run())
            checkers_ai.print_info()

except KeyboardInterrupt:
    pass

finally:
    checkers_ai.__del__()
    print("Exited")
