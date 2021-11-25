#!/usr/bin/python3.8
import sys
import time
import chess


def exhaustiveSearch(b, d, limiteTime = None):
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
      nbNodes += aux(b, d -1)
      b.pop()
    return nbNodes

  start_time = time.time()
  nbNodes = aux(b, d)
  print("Nb nodes :", nbNodes)
  print("Explorer en %0.2f seconds" % (time.time() - start_time))
  return


def main():
  board = chess.Board()
  # print(board)
  exhaustiveSearch(board, 4, time.time() + 30)

  return 0


if __name__ == '__main__':
  sys.exit(main())
