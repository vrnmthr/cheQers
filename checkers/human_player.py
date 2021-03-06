from game import Game

def make_human_player(name):
    def move(board, move_choices):
        Game.clear_screen()
        print("\n\nPlayer %s" % name)
        print("The current state of the board is:")
        print(board.board_str(move_choices))
        print("Your available moves are:")
        for i in range(len(move_choices)):
            print("%d: %s" % (i, move_choices[i]))
        raw_input = input("Enter the index of the move you wish to make: ")

        index = int(raw_input)

        return index
    return move
