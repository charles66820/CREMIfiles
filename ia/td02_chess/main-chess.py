#!/usr/bin/python3.8
import sys
import time
import chess
import click
from random import randint, choice
from IPython.display import display, clear_output

def run_from_iPython():
  try:
    __IPYTHON__
    return True
  except NameError:
    return False

def printBoard(b):
  if (run_from_iPython()):
    clear_output(wait=True)
    display(b)
  else:
    print("\033[8A%s" % b)

## Partie 2
# 1.
def exhaustiveSearch(b, d, limiteTime=None):
  def aux(b, d):
    if d == 0:
      return 1
    if limiteTime is not None and time.time() > limiteTime:
      raise TimeoutError
    if b.is_game_over():
      print("Resultat : ", b.result())
      return 1
    nbNodes = 1
    for m in b.generate_legal_moves():
      b.push(m)
      nbNodes += aux(b, d - 1)
      b.pop()
    return nbNodes

  start_time = time.time()

  print("==================")
  print("Depth :", d)
  nbNodes = aux(b, d)
  print("Nb nodes :", nbNodes)
  print("In %0.2f seconds" % (time.time() - start_time))

# 2.

# king, queen, rook, bishop, knight, pawn
pawnsValues = {'k': 200, 'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1}

def Shanon(b):
  score = 0
  nbPawns = {'k': 0, 'q': 0, 'r': 0, 'b': 0, 'n': 0,
             'p': 0, 'K': 0, 'Q': 0, 'R': 0, 'B': 0, 'N': 0, 'P': 0}

  for pos, pieces in b.piece_map().items():
    nbPawns[pieces.symbol()] += 1
    # chess.square_rank(63)

  # TODO: Ajoutez un moyen d’exprimer qu’il est préférable d’avancer ses pions pour les mener éventuellement à la Reine

  for key in pawnsValues:
    score += (pawnsValues[key] * (nbPawns[key] - nbPawns[key.upper()]))

  return score

# 3.
def miniMax(b, d, a, limiteTime=None):
  if d < 0:
    d = 0

  d += 1

  def aux(b, d):
    if d == 0 or b.is_game_over():
      return 1, Shanon(b), None

    if limiteTime is not None and time.time() > limiteTime:
      raise TimeoutError

    countNodes = 1
    bestScore = None
    bestMove = None
    for move in b.generate_legal_moves():
      b.push(move)
      nbNodes, score, _ = aux(b, d - 1)
      b.pop()

      countNodes += nbNodes

      if bestScore == None:
        bestScore = score
        bestMove = [move]
      else:
        newScore = b.turn == a if max(bestScore, score) else min(bestScore, score)

        if newScore == bestScore:
          bestMove.append(move)
        else:
          bestMove = [move]
          bestScore = newScore
    return countNodes, bestScore, bestMove[randint(0, len(bestMove) - 1)]
  return aux(b, d)

def MiniMax(b, d, a, limiteTime=None):
  return miniMax(b, d, a, limiteTime)[2]

# Iterative Deepening
def miniMaxID(b, d, a, seconds, debug=False):
  move = None
  try:
    for d_bis in range(d):
      if debug:
        print("==================")
        print("Depth : %d" % d_bis)
      start_time = time.time()
      nbNodes, score, newMove = miniMax(b, d_bis, a, time.time() + seconds)
      if debug:
        print("Nb nodes : %d" % nbNodes)
        print("Move : %s" % newMove)
        print("In %0.2f seconds" % (time.time() - start_time))
      move = newMove
  except TimeoutError:
    if debug:
      print("Time is over")
  return move

# 4.
def randomMove(b):
    return choice([m for m in b.generate_legal_moves()])

def match1(b, debug):
  if debug and not run_from_iPython(): print(b)
  while not b.is_game_over():
    if debug: printBoard(b)

    if b.turn == chess.WHITE:
      nextMove = randomMove(b)
    else:
      nextMove = MiniMax(b, 2, chess.BLACK)

    b.push(nextMove)
  print("Result : ", b.result())

def match2(b, debug):
  if debug and not run_from_iPython(): print(b)
  while not b.is_game_over():
    if debug: printBoard(b)

    if b.turn == chess.WHITE:
      nextMove = MiniMax(b, 0, chess.WHITE)
    else:
      nextMove = MiniMax(b, 2, chess.BLACK)

    b.push(nextMove)
  print("Result : ", b.result())

## Partie 3
# 1.
def alphaBeta(b, d, a, limiteTime=None):
  if d < 0:
    d = 0

  d += 1

  # maxScore = alpha
  # minScore = beta
  def aux(b, d, alpha, beta):
    if d == 0 or b.is_game_over():
      return 1, Shanon(b), None

    if limiteTime is not None and time.time() > limiteTime:
      raise TimeoutError

    countNodes = 1
    bestMove = []
    for move in b.generate_legal_moves():
      b.push(move)
      nbNodes, score, _ = aux(b, d - 1, alpha, beta)
      b.pop()

      countNodes += nbNodes

      if b.turn == a:
        newAlpha = max(alpha, score)
        if newAlpha >= beta:
          return countNodes, beta, move

        if newAlpha == alpha:
          bestMove.append(move)
        else:
          bestMove = [move]
          alpha = newAlpha
      else:
        newBeta = min(beta, score)
        if alpha >= newBeta:
          return countNodes, alpha, move

        if newBeta == beta:
          bestMove.append(move)
        else:
          bestMove = [move]
          beta = newBeta

    return countNodes, b.turn == a if alpha else beta, bestMove[randint(0, len(bestMove) - 1)]
  return aux(b, d, -sys.maxsize - 1, sys.maxsize)

def match3(b, debug):
  if debug and not run_from_iPython(): print(b)
  while not b.is_game_over():
    if debug: printBoard(b)

    if b.turn == chess.WHITE:
      nextMove = alphaBeta(b, 2, b.turn)[2]
    else:
      nextMove = MiniMax(b, 2, chess.BLACK)

    b.push(nextMove)
  print("Result : ", b.result())

# 2.
# Iterative Deepening
def alphaBetaID(b, d, a, seconds, debug=False):
  move = None
  try:
    for d_bis in range(d):
      if debug:
        print("==================")
        print("Depth : %d" % d_bis)
      start_time = time.time()
      nbNodes, score, newMove = alphaBeta(b, d_bis, a, time.time() + seconds)
      if debug:
        print("Nb nodes : %d" % nbNodes)
        print("Move : %s" % newMove)
        print("In %0.2f seconds" % (time.time() - start_time))
      move = newMove
  except TimeoutError:
    if debug:
      print("Time is over")
  return move

# 3.
def humanMove(b):
  while True:
    userUCI = input("Your move (ex: a2a4) :")

    try:
      move = chess.Move.from_uci(userUCI)
      if any(str(move) == str(m) for m in b.generate_legal_moves()):
        return move
      else:
        print("Bad move")
        print("Possible move :")
        for m in b.generate_legal_moves():
          print(m, end=', ')
        print("")
    except:
      print("Bad instruction")

def matchHuman(b):
  if click.confirm("You want to start?", default=True):
    playerColor = chess.WHITE
  else:
    playerColor = chess.BLACK

  if not run_from_iPython(): print(b)
  while not b.is_game_over():
    printBoard(b)

    if b.turn == playerColor:
      nextMove = humanMove(b)
    else:
      nextMove = alphaBetaID(b, 2, b.turn, 10)

    b.push(nextMove)
  print("Result : ", b.result())

def main():
  ## Partie 2

  print("Lancement de exhaustiveSearch :")
  board = chess.Board()
  # try:
  #   for i in range(10):
  #     exhaustiveSearch(board, i, time.time() + 30)
  # except TimeoutError:
  #   print("Time is over")

  # 1. En moins de 30 seconds on peut aller a une profondeur de `4`. On s'arrête a la 5ème.

  # | Profondeur :       | 1  |  2  |  3   |   4    |
  # |:------------------:|:--:|:---:|:----:|:------:|
  # | Nombre de noeuds : | 21 | 421 | 9323 | 206604 |

  # 2.
  print("\nTest Shanon :", Shanon(board))

  # 3.
  print("\nLancement de miniMaxID :")
  # miniMaxID(chess.Board(), 8, chess.WHITE, 30, True)

  print("\nTest niveau 1, 2 et 3 :")
  board = chess.Board()
  print("level 1 :", MiniMax(board, 0, chess.WHITE))
  print("level 2 :", MiniMax(board, 1, chess.WHITE))
  print("level 3 :", MiniMax(board, 2, chess.WHITE))

  # 4.
  print("\nMatch Joueur Aléatoire contre Minimax niveau 3")
  # match1(chess.Board(), True)

  print("\nMatch Minimax niveau 1 contre Minimax niveau 3")
  # match2(chess.Board(), True)

  ## Partie 3
  # 1. J'ai coder l'`Iterative Deepening` (`alphaBetaID()`) avant de faire la comparaison. La comparaison est donc plus loin.
  print("\nMatch Minimax contre α − β")
  # match3(chess.Board(), True)

  # 2.
  print("\nLancement de alphaBetaID :")
  # alphaBetaID(board, 8, chess.WHITE, 10, True)

  # Comparaison de la question 1 :
  print("\nAvec une partie partie sans aucun coup jouer")
  board = chess.Board()
  print(board)
  printBoard(board)

  print("\nMiniMax :")
  # miniMaxID(board, 8, chess.WHITE, 30, True)

  print("\nAlpha-beta :")
  # alphaBetaID(board, 8, chess.WHITE, 10, True)

  print("\nAvec une partie partie avec des coups jouer")
  board = chess.Board()
  board.push(chess.Move.from_uci("d2d4"))
  board.push(chess.Move.from_uci("c7c5"))
  board.push(chess.Move.from_uci("d1d3"))
  board.push(chess.Move.from_uci("g8f6"))
  print(board)
  printBoard(board)

  print("\nMiniMax :")
  # miniMaxID(board, 8, chess.WHITE, 30, True)

  print("\nAlpha-beta :")
  # alphaBetaID(board, 8, chess.WHITE, 10, True)


  # On voit que pour uns profondeur de `0` et `1` il y a peu de différence de nœuds explorés.
  # Par contre pour une profondeur plus grand, on voit que `alpha-beta` parcoure beaucoup moins de nœuds.
  # Et `alpha-beta` et nettement plus rapide.
  # Sur une partie déjà commencée, on peut voir que `alpha-beta` est bien plus efficace que `miniMax`.

  print("\nMatch contre l'ia :")
  matchHuman(chess.Board())

  return 0

if __name__ == '__main__':
  sys.exit(main())
