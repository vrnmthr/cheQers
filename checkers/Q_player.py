from game import Game
from checkers_learner import CheQer

checkers_ai = CheQer(.7, .1, 64)

for i in range(1000):
    print("Winner of game ", i, ": ", Game(checkers_ai.step, checkers_ai.step).run())