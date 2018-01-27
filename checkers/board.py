#!/usr/bin/python
from enum import Enum

class Piece(Enum):
    BLACK_K =   -2
    BLACK =     -1
    NONE =      0
    WHITE =     1
    WHITE_K =   2

class Board:
    def __init__(self, size):
        assert size == 8
        self.size = size
        self.board_arr = []
        # setup the board / starting positions
        self.setup_board()

    def setup_board(self):
        """ Fills the board with pieces in their starting positions.
            Adds WHITE pieces at the top to start (so white should move first)"""
        self.board_arr = [[0 for x in range(w)] for y in range(h)]
        for y in range(size):
            for x in range(size):
                if y < 3 and self.is_checkboard_space(x, y):
                    # add white pieces to the top (in a checkerboard pattern of black spaces - not on white spaces)
                    self.board_arr[y][x] = Piece.WHITE

                elif y >= size - 3 and is_checkboard_space(x, y):
                    # ... and black pieces to the bottom in the opposite pattern
                    self.board_arr[y][x] = Piece.BLACK

    def apply_move(self, move):
        """ Using the given move and piece, move the piece on the board and apply it to self board. """
        # NOTE: at self point, the starting position of the move (move.getStartingPosition) will not neccesarily
        # be equal to the piece's location, because jumping moves have no understanding of the root move
        # and therefore can only think back one jump. WE ARE PRESUMING that the piece given to self function
        # is the one which the move SHOULD be applied to, but due to self issue we can't test self.

        move_start = move.get_start()
        move_end = move.get_end()

        # find any pieces we've jumped in the process, and remove them as well
        jumped_pieces = move.get_jumped_pieces(self)
        if jumped_pieces != None:
            # loop over all jumped pieces and remove them
            for coords in jumped_pieces:
                if coords != None: # TODO: why necessary?
                    self.remove_piece(coords)

        # and, move self piece (WE PRESUME that it's self piece) from its old spot (both on board and with the piece itself)
        self.move_piece(move_start, move_end)
    }

    ##### PIECE MANIPULATION #####
    def get_piece_at(self, coords):
        return self.board_arr[coords[1]][coords[0]]

    def set_piece_at(self, coords, value):
        self.board_arr[coords[1]][coords[0]] = value

    def is_white(self, coords):
        return self.get_piece_at(coords) > 0

    def remove_piece(self, coords):
        """ Removes the piece at the given coords from the board. """
        self.set_piece_at(coords, Piece.NONE)

    def move_piece(self, start, end):
        """ Moves the piece at start to the end location (does not validate the move) """
        piece = self.get_piece_at(start)
        assert piece != Piece.NONE
        self.set_piece_at(end, piece)
        self.set_piece_at(start, Piece.NONE)
        self.check_if_should_be_king(end)

    def set_king(self, coords):
        self.set_piece_at(coords, is_white(piece) ? Piece.WHITE_K : Piece.BLACK_K)

    def check_if_should_be_king(self, coords):
        """ Makes the piece at the given location if it should be right now """
        white = is_white(coords)
        if (white and coords[1] == board.size - 1) or (!white and coords[1] == 0)
            self.set_king(coords)

    ##### END PIECE MANIPULATION #####
    #
    # /**
    #  * Get's the Piece object at self location, but using a single number,
    #  * which progresses from 0 at the top left to the square of the size at the bottom right
    #  * @param position self number, zero indexed at top left
    #  * @return The Piece here. (may be null).
    #  */
    # public Piece get_piece_at(int position)
    # {
    #     int[] coords = getCoordinatesFromPosition(position); # convert position to coordinates and use that
    #     return self.get_piece_at(coords[0], coords[1])
    # }

    # /**
    #  * Converts a single position value to x and y coordinates.
    #  * @param position The single position value, zero indexed at top left.
    #  * @return A two part int array where [0] is the x coordinate and [1] is the y.
    #  */
    # public int[] getCoordinatesFromPosition(int position)
    # {
    #     int[] coords = new int[2]
    #
    #     # get and use x and y by finding low and high frequency categories
    #     coords[0] = position % self.size; # x is low frequency
    #     coords[1] = position / self.size; # y is high frequency
    #     return coords
    # }
    #
    # /**
    #  * Converts from x and y coordinates to a single position value,
    #  * which progresses from 0 at the top left to the square of the size minus one at the bottom right
    #  * @param x The x coordinate
    #  * @param y The y coordinate
    #  * @return The single position value.
    #  */
    # public int getPositionFromCoordinates(int x, int y)
    # {
    #     # sum all row for y, and add low frequency x
    #     return self.size*y + x
    # }

    def is_checkboard_space(int x, int y):
        """ Returns true if the given position on the board represents
            a "BLACK" square on the checkboard.
            (The checkerboard in self case starts with a "white"
            space in the upper left hand corner """
        # self is a checkerboard space if x is even in an even row or x is odd in an odd row
        return x % 2 == y % 2

    def is_over_edge(x, y):
        """ Returns true if the given coordinates are over the edge the board """
        return x < 0 or x >= self.size or y < 0 or y >= self.size

    # /**
    #  * @return Returns true if the given position is over the edge the board
    #  * @param position The given 0-indexed position value
    #  */
    # public boolean is_over_edge(int position)
    # {
    #      int[] coords = getCoordinatesFromPosition(position); # convert position to coordinates and use that
    #     return self.is_over_edge(coords[0], coords[1])
    # }

    def get_flipped_board():
        # TODO:
        """ Flips the board coordinates so that the other pieces are on top, etc. """
        # copy self Board, as the basis for a new, flipped one
        newBoard = Board(self)

        # switch every piece to the one in the opposite corner
        for (int y = 0; y < newBoard.size; y++)
        {
            for (int x = 0; x < newBoard.size; x++)
            {
                # get piece in opposite corner...
                Piece oldPiece = self.get_piece_at(self.size - 1 - x, self.size - 1 - y)

                if (oldPiece != null)
                {
                    # ...and transfer color and position to a new generated piece if it exists
                    newBoard.setValueAt(x, y, new Piece(x, y, oldPiece.isWhite))
                }
                else
                {
                    # otherwise just add an empty space
                    newBoard.setValueAt(x, y, null)
                }
            }
        }

        return newBoard
    }
}
