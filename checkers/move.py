#!/usr/bin/python

class Move:
    def __init__(self, start, end, preceding_move, is_jump):
        self.start = start
        self.end = end
        self.preceding_move = preceding_move
        self.is_jump = is_jump

    def get_start(self):
        return self.start

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
            piece_x = (self.start[0] + self.end[0]) / 2;
            piece_y = (self.start[1] + self.end[1]) / 2;

            # add this most recent jump...
            pieces.append([piece_x, piece_y]);

            # ...but also go back to get the inbetween ones (if we're not the first move)
            if self.preceding_move is not None:
                prev_jumped = self.preceding_move.get_jumped_pieces(board)
                pieces.extend(prev_jumped);

                # something is wrong (a preceding move isn't a jump) if this returns null, so let the error be thrown

        return pieces

    def __str__(self):
        return "%s -> %s" % (self.start, self.end)
