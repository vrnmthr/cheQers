from game import Game
from checkers_learner import CheQer
from human_player import make_human_player

checkers_ai = CheQer(.7, .1, 64)

train = True
player1 = checkers_ai.step if train else make_human_player("Human")
player2 = checkers_ai.step

for i in range(1000000):
    try:
        print("Winner of game ", i, ": ", Game(player1, player2).run())
        checkers_ai.print_info()
    except KeyboardInterrupt:
        checkers_ai.__del__()
        print("Exited")
        break