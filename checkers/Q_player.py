import os
from game import Game
from checkers_learner import CheQer
from human_player import make_human_player
import win_unicode_console as wuc

wuc.enable()

checkers_ai = CheQer(.7, .1, [35, 15], .000001)

train = True
player1 = checkers_ai.step if train else make_human_player("Human")
player2 = checkers_ai.step

try:
    for i in range(1000000):
        checkers_ai.print_info()
        print("Winner of game ", i, ": ", Game(player1, player2).run())
        print("Current save step: ", checkers_ai.train_step % checkers_ai.SAVE_STEP_NUM)


except KeyboardInterrupt:
    pass

finally:
    checkers_ai.__del__()
    print("Exited")
