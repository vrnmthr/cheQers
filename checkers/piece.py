
#!/usr/bin/python
from enum import IntEnum
from move import Move
import logging

class Piece(IntEnum):
    BLACK_K =   -2
    BLACK =     -1
    NONE =      0
    WHITE =     1
    WHITE_K =   2

    ## static methods
    @staticmethod
    def to_str(val):
        if val == Piece.BLACK_K:  return "BK"
        if val == Piece.BLACK:    return "B "
        if val == Piece.NONE:     return "_ "
        if val == Piece.WHITE:    return "W "
        if val == Piece.WHITE_K:  return "WK"

    @staticmethod
    def is_white(board, coords):
        return board.get_piece_at(coords) > 0

    @staticmethod
    def is_white_val(val):
        return val > 0

    @staticmethod
    def set_king(board, coords):
        board.set_piece_at(coords, Piece.WHITE_K if Piece.is_white(board, coords) else Piece.BLACK_K)

    @staticmethod
    def is_king(board, coords):
        piece_val = board.get_piece_at(coords)
        return piece_val == Piece.WHITE_K or piece_val == Piece.BLACK_K

    @staticmethod
    def check_if_should_be_king(board, coords):
        """ Makes the piece at the given location a king if it should be right now """
        white = Piece.is_white(board, coords)
        if (white and coords[1] == board.size - 1) or (not white and coords[1] == 0):
            Piece.set_king(board, coords)

    @staticmethod
    def get_all_possible_moves(board, coords):
        """ Generates all physically possible moves of the given piece.
        (Only actually generates the non-jumping moves -
        jumps are done recusively in getAllPossibleJumps)

        @return Returns a list of all the moves (including recusively found jumps), including each individual one involved in every jump.

        @param board The board to work with - assumed to be flipped to correspond to this piece's color. """

        moves = []

        # use kingness to determine number of rows to check
        rows_to_check = 2 if Piece.is_king(board, coords) else 1
        piece_x, piece_y = coords
        is_white = Piece.is_white(board, coords)
        is_king = Piece.is_king(board, coords)

        # if at far edge as normal piece,
        # stop looking for moves (we'll be kinged, so the turn is over)
        if not is_king and piece_y == board.size - 1:
            return []

        # change y endpoints based on color=direction of movement
        if is_white:
            # if it's white, we move from further down the board backwards to possible king position
            starting_y = piece_y + 1
            y_increment = -2
        else:
            # if it's black, we move from further up the board forward to possible king position
            starting_y = piece_y - 1
            y_increment = 2

        # iterate over the four spaces where normal (non-jumping) moves are possible
        for x in [piece_x - 1, piece_x + 1]:
            # go over the rows (or row) (we iterate the number of times determined by the kingess above)
            # )add this so we can add the normal increment before the boundary checks in the loop)
            y = starting_y - y_increment
            for i in range(rows_to_check):
                # increment y if we need to (this will have no effect if we only run one iteration)
                y += y_increment

                # check for going off end of board, in which case just skip this iteration
                # (we may do this twice if at a corner)
                if board.is_over_edge(x, y):
                    continue

                # add a move here if there's not a piece
                if board.get_piece_at([x, y]) == Piece.NONE:
                    # this is not jump move in any case, and is always the first move
                    moves.append(Move([piece_x, piece_y], [x, y], None, False))

        # after we've checked all normal moves, look for and add all possible jumps
        # (recursively as well - I mean ALL jumps)
        possible_jumps = Piece.get_all_possible_jumps(board, coords, None, [], is_white, is_king)
        moves.extend(possible_jumps)
        return moves

    @staticmethod
    def get_all_possible_jumps(board, coords, preceding_move, jumped_coords, is_white, is_king):
        """  Finds all jumping moves originating from this piece.
        Does this recursively; for each move a new imaginary piece will be generated,
        and this function will then be called on that piece to find all possible subsequent moves.

        @param board The board to work with - assumed to be flipped to correspond to this piece's color.

        @param preceding_move The moves preceding the call to search for moves off this piece - only used in recursion, should be set to null at first call. (if it's not, it means this piece is imaginary).

        Other args are passed in instead of found on board because we must call
        this with imaginary pieces. """

        moves = []

        # use kingness to determine number of rows to check
        rows_to_check = 2 if is_king else 1
        piece_x, piece_y = coords

        # if at far edge as normal piece,
        # stop looking for moves (we'll be kinged, so the turn is over)
        if not is_king and piece_y == board.size - 1:
            return []

        # change y endpoints based on color=direction of movement
        if is_white:
            # if it's white, we move from further down the board backwards to possible king position
            starting_y = piece_y + 2
            y_increment = -4
        else:
            # if it's black, we move from further up the board forward to possible king position
            starting_y = piece_y - 2
            y_increment = 4

        # iterate over the four spaces where normal (non-jumping) moves are possible
        for x in [piece_x - 2, piece_x + 2]:
            # go over the rows (or row) (we iterate the number of times determined by the kingess above)
            # )add this so we can add the normal increment before the boundary checks in the loop)
            y = starting_y - y_increment
            for _ in range(rows_to_check):
                # increment y if we need to (this will have no effect if we only run one iteration)
                y += y_increment

                # check for going off end of board, in which case just skip this iteration
                # (we may do this twice if at a corner)
                if board.is_over_edge(x, y):
                    continue

                # don't try to go backward to our old move start so we don't get in infinite recursion loops
                if preceding_move is not None \
                    and x == preceding_move.get_last_hop()[0] \
                    and y == preceding_move.get_last_hop()[1]:
                    continue

                # test if there is a different-colored piece between us
                # (at the average of our position) and the starting point
                # AND that there's no piece in the planned landing space (meaning we can possible jump there)
                between_coords = [int((piece_x + x)/2), int((piece_y + y)/2)]
                between_piece = board.get_piece_at(between_coords)
                if between_piece != Piece.NONE \
                    and Piece.is_white_val(between_piece) != is_white \
                    and board.get_piece_at([x, y]) == Piece.NONE:

                    # check if the targeted piece has already been jumped in this jump sequence
                    b_break = False
                    for j_coords in jumped_coords:
                        if between_coords == j_coords:
                            b_break = True
                            break
                    if b_break:
                        logging.info("Infinite Loop")
                        break

                    # in which case, add a move here, and note that it is a jump (we may be following some other jumps)
                    # (origin points are absolute origin (ORIGINAL piece))
                    jumping_move = Move([piece_x, piece_y], [x, y], preceding_move, True)
                    moves.append(jumping_move)

                    # after jumping, create an imaginary piece as if it was there to look for more jumps
                    imaginary_piece = [x, y]
                    # add the jumped piece to jumped_pieces
                    jumped_coords.append(between_coords)
                    # find possible subsequent moves recursively
                    subsequent_moves = Piece.get_all_possible_jumps(board,
                        imaginary_piece, jumping_move, jumped_coords, is_white, is_king)

                    # add these moves to our list if they exist, otherwise just move on to other possibilities
                    moves.extend(subsequent_moves)

        return moves
