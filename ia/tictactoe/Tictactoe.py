# -*- coding: utf-8 -*-

class Board:
    '''Classe permettant de jouer au morpion. Elle semble assez longue mais elle illustre
       le fonctionnement classique des jeux de plateau, en fournissant des méthodes que
       l'on retrouvera plus tard avec le jeu des échecs'''
    _X = 'X'
    _O = 'O'
    _E = '.'  # empty

    def __init__(self):
        self._nextPlayer = self._X

        self._board = []
        for x in range(3):
            self._board.append([self._E] * 3)

        self._alignments = []
        for x in range(3):
            a = []
            amirror = []
            for y in range(3):
                a.append((x, y))
                amirror.append((y, x))
            self._alignments.append(a)
            self._alignments.append(amirror)
        self._alignments.append([(0, 0), (1, 1), (2, 2)])
        self._alignments.append([(2, 0), (1, 1), (0, 2)])

        self._stack = []  # Used to keep track of push/pop moves

    def _get_an_alignment(self):
        for a in self._alignments:
            if (self._board[a[0][0]][a[0][1]] != self._E) and (self._board[a[0][0]][a[0][1]] == self._board[a[1][0]][a[1][1]]) and (self._board[a[0][0]][a[0][1]] == self._board[a[2][0]][a[2][1]]):
                return self._board[a[0][0]][a[0][1]]
        return None

    def _has_an_alignment(self):
        return self._get_an_alignment() is not None

    def _at_least_one_empty_cell(self):
        for x in range(3):
            for y in range(3):
                if self._board[x][y] == self._E:
                    return True
        return False

    def is_game_over(self):
        '''Test si le jeu est terminé'''
        if self._has_an_alignment():
            return True
        if self._at_least_one_empty_cell():
            return False
        return True

    def result(self):
        '''Retourne le vainqueur du jeu'''
        return self._get_an_alignment()

    def push(self, move):
        '''Permet d'empiler un coup pour pouvoir le déjouer ensuite.'''
        [player, x, y] = move
        assert player == self._nextPlayer
        self._stack.append(move)
        self._board[x][y] = player
        if self._nextPlayer == self._X:
            self._nextPlayer = self._O
        else:
            self._nextPlayer = self._X

    def pop(self):
        '''Désempile un coup qui a été joué. Permet de retrouver le plateau dans
           l'état dans lequel il était avant de jouer.'''
        move = self._stack.pop()
        [player, x, y] = move
        self._nextPlayer = player
        self._board[x][y] = self._E

    def legal_moves(self):
        '''Une fonction importante : elle permet de retourner tous les coups possibles
           pour le plateau de jeu courant'''
        moves = []
        for x in range(3):
            for y in range(3):
                if self._board[x][y] == self._E:
                    moves.append([self._nextPlayer, x, y])
        return moves

    def _piece2str(self, c):
        if c == self._O:
            return 'O'
        elif c == self._X:
            return 'X'
        else:
            return '.'

    def __str__(self):
        toreturn = ""
        for l in self._board:
            for c in l:
                toreturn += self._piece2str(c)
            toreturn += "\n"
        toreturn += "Next player: " + \
            ("X" if self._nextPlayer == self._X else "O") + "\n"
        return toreturn

    __repr__ = __str__
