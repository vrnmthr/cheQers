#!/usr/bin/python

class Move:
    """ A single hop (the last known hop) of a linked list of moves and jumps. """
    def __init__(self, last_hop, end, preceding_move, is_jump):
        self.last_hop = last_hop
        self.end = end
        self.is_jump = is_jump
        self.preceding_move = preceding_move
        self.start = self.__find_start()

    def __find_start(self):
        # if this move is a jump with a preceding move,
        # then our start is different from our last hop
        if self.is_jump and self.preceding_move is not None:
            move = self.preceding_move
            while move.preceding_move is not None:
                move = move.preceding_move
            return move.start  # move for which preceding_move was None
        else:
            return self.last_hop

    def get_start(self):
        return self.start

    def get_last_hop(self):
        return self.last_hop

    def get_end(self):
        return self.end

    def get_jumped_pieces(self, board):
        """ Finds the pieces jumped in this move.
            (Get's inbetween jumps using recursion) """
        pieces = []

        # if this move wasn't a jump, it didn't jump a piece!
        if self.is_jump:
            # the piece this move is jumping should be between the start
            # and end of this move (the average of those two positions)
            piece_x = (self.last_hop[0] + self.end[0]) / 2
            piece_y = (self.last_hop[1] + self.end[1]) / 2

            # add this most recent jump...
            pieces.append([piece_x, piece_y])

            # ...but also go back to get the inbetween ones (if we're not the first move)
            if self.preceding_move is not None:
                prev_jumped = self.preceding_move.get_jumped_pieces(board)
                pieces.extend(prev_jumped)

            # something is wrong (a preceding move isn't a jump) if this returns null, so let the error be thrown

        return pieces

    def __str__(self):
        # build up chain from end
        if self.preceding_move is not None:
            return "%s ^  %s" % (str(self.preceding_move), self.end)
        else:
            return "%s %s %s" % (str(self.start), "^ " if self.is_jump else "->", str(self.end))
