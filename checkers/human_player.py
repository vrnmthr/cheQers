def move(self, board, move_choices):
    print("The current state of the board is:")
    print(board)
    print("Your available moves are:")
    for move in move_choices:
        print(move)
    raw_input = input("Enter the index of the move you wish to make.")

    index = int(raw_input)

    return index