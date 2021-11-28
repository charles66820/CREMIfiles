#!/usr/bin/python3.8
import sys
import time
import chess
from random import choice
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
  print("==================")
  return

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
def MiniMax(b, d, a):
  if d < 0:
    d = 0

  d += 1

  def aux(b, d, a):
    if d == 0:
      return Shanon(b), None

    if b.is_game_over():
      return Shanon(b), None

    bestScore = None
    bestMove = None
    for move in b.generate_legal_moves():
      b.push(move)

      score, _ = aux(b, d - 1, a)

      if bestScore == None:
        bestScore = score
        bestMove = move
      else:
        newScore = b.turn == a if max(bestScore, score) else min(bestScore, score)

        if newScore != bestScore:
          bestMove = move
          bestScore = newScore

      b.pop()
    return bestScore, bestMove
  return aux(b, d, a)[1]

# 4.
def randomMove(b):
    return choice([m for m in b.generate_legal_moves()])

def match1(b, debug):
  if debug and not run_from_iPython(): print(b)
  while True:
    if debug: printBoard(b)

    if b.is_game_over():
      print("Result : ", b.result())
      break

    if b.turn == chess.WHITE:
      nextMove = randomMove(b)
    else:
      nextMove = MiniMax(b, 2, chess.BLACK)

    b.push(nextMove)

def match2(b, debug):
  if debug and not run_from_iPython(): print(b)
  while True:
    if debug: printBoard(b)

    if b.is_game_over():
      print("Result : ", b.result())
      break

    if b.turn == chess.WHITE:
      nextMove = MiniMax(b, 0, chess.WHITE)
    else:
      nextMove = MiniMax(b, 2, chess.BLACK)

    b.push(nextMove)

def main():
  ## Partie 2

  print(run_from_iPython())

  board = chess.Board()

  try:
    for i in range(10):
      exhaustiveSearch(board, i, time.time() + 30)
  except TimeoutError:
    print("Time is over")

  # 1. En moins de 30 seconds on peut aller a une profondeur de `4`. On s'arrête a la 5ème.

  # | Profondeur :       | 1  |  2  |  3   |   4    |
  # |:------------------:|:--:|:---:|:----:|:------:|
  # | Nombre de noeuds : | 21 | 421 | 9323 | 206604 |

  # 2.
  print("Test Shanon :", Shanon(board))

  # 3.
  print(MiniMax(board, 0, chess.WHITE)) # level 1
  print(MiniMax(board, 1, chess.WHITE)) # level 2
  print(MiniMax(board, 2, chess.WHITE)) # level 3

  # 4.
  print("Match Joueur Aléatoire contre Minimax niveau 3")
  match1(chess.Board(), True)

  print("Match Minimax niveau 1 contre Minimax niveau 3")
  match2(chess.Board(), True)

  return 0

if __name__ == '__main__':
  sys.exit(main())
