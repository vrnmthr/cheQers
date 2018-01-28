import random

def make_random_player(name):
    def move(board, move_choices):
        print("\n\nPlayer %s" % name)
        print("The current state of the board is:")
        print(board)
        print("Available moves are:")
        for i in range(len(move_choices)):
            print(move_choices[i])

        index = random.randint(0, len(move_choices) - 1)
        print("Chose index %d" % index)
        return index

    return move