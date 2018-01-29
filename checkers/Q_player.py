import os
from game import Game
from checkers_learner import CheQer
from human_player import make_human_player

checkers_ai = CheQer(.7, .1, [64, 35, 15])

train = False
player1 = checkers_ai.step if train else make_human_player("Human")
player2 = checkers_ai.step

for i in range(2):
    # try:
    print("Winner of game ", i, ": ", Game(player1, player2).run())
    # checkers_ai.print_info()
    #
    # except KeyboardInterrupt:
    #     pass
    #
    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)
    #
    # finally:
    #     checkers_ai.__del__()
    #     print("Exited")
    #     break