#!/usr/bin/python

class Game:
    def __init__(self, player1move, player2move):
        self.players = [player1move, player2move]

        self.board = Board()
        self.done = False

    def run(self):
        """ Returns which player won the game (1 or 2), or 0 if a stalemate """
        player = 0
        while not board.won():
            self.get_player_move(player)
            # alternate players
            player = (player + 1) % 2

        return board.winner()

    def get_player_move(self, player_num):
        while True:
            player_board = self.board.player_board(player_num)
            available_moves = self.board.available_moves(player_num)
            move_i = self.players[player_num](player_board, available_moves)

            if 0 <= move_i or move_i < len(available_moves):
                self.board.apply_move(available_moves[move_i])
                return
