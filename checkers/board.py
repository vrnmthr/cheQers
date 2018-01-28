#!/usr/bin/python
import numpy as np
import copy
from piece import Piece

class Board:
    """ A board, to be used as if always from the perspective of the white player
        (you should call board.flip() in order to player as the other player) """

    def __init__(self, size, board_arr=None, start_white_player=0):
        assert size == 8
        self.size = size
        self.cur_white_player = start_white_player
        if board_arr is not None:
            assert board_arr.shape == (size, size)
            self.board_arr = board_arr
        else:
            self.board_arr = np.zeros((self.size, self.size))
            # setup the board / starting positions
            self.setup_board()

    def __copy__(self):
        return Board(self.size, copy.copy(self.board_arr), self.cur_white_player)

    def __deepcopy__(self, memodict={}):
        return Board(self.size, copy.deepcopy(self.board_arr, memodict), self.cur_white_player)

    def setup_board(self):
        """ Fills the board with pieces in their starting positions.
            Adds WHITE pieces at the top to start (so white should move first)"""
        for y in range(self.size):
            for x in range(self.size):
                if y < 3 and self.is_checkboard_space(x, y):
                    # add white pieces to the top (in a checkerboard pattern of black spaces - not on white spaces)
                    self.board_arr[y][x] = Piece.WHITE

                elif y >= self.size - 3 and self.is_checkboard_space(x, y):
                    # ... and black pieces to the bottom in the opposite pattern
                    self.board_arr[y][x] = Piece.BLACK

    def apply_white_move(self, move):
        """ Using the given move and piece, move the piece on the board and apply it to self board. """
        # NOTE: at self point, the starting position of the move (move.getStartingPosition) will not neccesarily
        # be equal to the piece's location, because jumping moves have no understanding of the root move
        # and therefore can only think back one jump. WE ARE PRESUMING that the piece given to self function
        # is the one which the move SHOULD be applied to, but due to self issue we can't test self.

        move_start = move.get_start()
        move_end = move.get_end()

        # find any pieces we've jumped in the process, and remove them
        jumped_pieces = move.get_jumped_pieces(self)
        for coords in jumped_pieces:
            self.remove_piece(coords)

        # and, move self piece (WE PRESUME that it's self piece) from its old spot
        # (both on board and with the piece itself)
        self.move_piece(move_start, move_end)

    def available_white_moves(self):
        """ Returns a list of all move objects for all white pieces on the board """
        moves = []
        # TODO: could speed up by only doing checkboard spaces
        for y in range(self.size):
            for x in range(self.size):
                coords = [x, y]
                piece = self.get_piece_at(coords)
                if piece != Piece.NONE and Piece.is_white_val(piece):
                    moves.extend(Piece.get_all_possible_moves(self, coords))

        return moves

    def won(self):
        return self.winner() == -1

    def winner(self):
        """ Returns the winner of the game (player 1 or 2),
        0 for a stalemate, and -1 if the game is not finished """

        movable_nums = [0, 0]
        for y in range(self.size):
            for x in range(self.size):
                # make sure the piece exists, and if so sum movable pieces for each color)
                coords = [x, y]
                if self.get_piece_at(coords) != Piece.NONE:
                    # only consider piece if it has possible moves
                    if len(Piece.get_all_possible_moves(self, coords)) > 0:
                        if Piece.is_white(self, coords):
                            movable_nums[self.cur_white_player] += 1
                        else:
                            movable_nums[(self.cur_white_player + 1) % 2] += 1

        # determine if anyone won (or if no one had any moves left)
        if movable_nums[0] + movable_nums[1] == 0:
            return 0
        elif movable_nums[1] == 0:
            return 1 # player 1 wins if player 2 has NO moves
        elif movable_nums[0] == 0:
            return 2 # player 2 wins if player 1 has NO moves
        else:
            return -1

    def cur_player_won(self):
        """ Returns 1 if the current player has won, 2 if the other has,
        0 for a stalemate, and -1 if the game is not finished """
        winner = self.winner()
        if winner < 1:
            return winner
        else:
            return 1 if self.cur_white_player == winner else 2

    def set_white_player(self, player_num):
        """ Sets the current "white" player by flipping the board if
            the current board doesn't match the provided player
            (note that the game starts with player 0 as white)"""
        assert 0 <= player_num < 2
        if self.cur_white_player % 2 != player_num:
            self.__flip()
        self.cur_white_player = player_num

    def __flip(self):
        """ Flips the board coordinates so that the other pieces are on top, AND
            converts all black players to white to completely swap perspective"""
        self.board_arr = np.rot90(self.board_arr, 2)
        self.board_arr *= -1  # converts all black to white + v.v.

    ##### PIECE MANIPULATION #####
    def get_piece_at(self, coords):
        return self.board_arr[coords[1]][coords[0]]

    def set_piece_at(self, coords, value):
        self.board_arr[coords[1]][coords[0]] = value

    def remove_piece(self, coords):
        """ Removes the piece at the given coords from the board. """
        self.set_piece_at(coords, Piece.NONE)

    def move_piece(self, start, end):
        """ Moves the piece at start to the end location (does not validate the move) """
        piece = self.get_piece_at(start)
        assert piece != Piece.NONE
        self.set_piece_at(end, piece)
        self.set_piece_at(start, Piece.NONE)
        Piece.check_if_should_be_king(self, end)

    ##### END PIECE MANIPULATION #####

    def is_checkboard_space(self, x, y):
        """ Returns true if the given position on the board represents
            a "BLACK" square on the checkboard.
            (The checkerboard in self case starts with a "white"
            space in the upper left hand corner """
        # self is a checkerboard space if x is even in an even row or x is odd in an odd row
        return x % 2 == y % 2

    def is_over_edge(self, x, y):
        """ Returns true if the given coordinates are over the edge the board """
        return x < 0 or x >= self.size or y < 0 or y >= self.size

    def __str__(self):
        buf = []
        for y in range(self.size):
            for x in range(self.size):
                buf.append(Piece.to_str(self.get_piece_at([x, y])))
            buf.append("\n")
        return "".join(buf)