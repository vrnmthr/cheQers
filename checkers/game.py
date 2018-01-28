#!/usr/bin/python

import copy
from board import Board

class Game:
    def __init__(self, player1move, player2move):
        self.players = [player1move, player2move]

        self.board = Board(8)
        self.done = False

    def run(self):
        """ Returns which player won the game (1 or 2), or 0 if a stalemate """
        player_num = 0
        while not self.board.won():
            self.get_player_move(player_num)
            # alternate players
            player_num = (player_num + 1) % 2

        return self.board.winner()

    def get_player_move(self, player_num):
        # flip the board so they current player is "white"
        self.board.set_white_player(player_num)
        while True:
            # do a deep copy to allow for user manipulation
            available_moves = self.board.available_white_moves()
            move_i = self.players[player_num](copy.deepcopy(self.board), available_moves)

            if 0 <= move_i or move_i < len(available_moves):
                self.board.apply_white_move(available_moves[move_i])
                return
