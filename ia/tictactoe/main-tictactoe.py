#!/usr/bin/python3.8
import sys
import time
import Tictactoe

def getscore(b):
  if b.result() == b._X:
    return 1
  elif b.result() == b._O:
    return -1
  return 0

def brutForce(b):
  def aux(b):
    countLeaf = 0
    countNodes = 1
    for move in b.legal_moves():
      b.push(move)
      if b.is_game_over():
        countLeaf += 1
        countNodes += 1
      else:
        rLeaf, rNodes = aux(b)
        countLeaf += rLeaf
        countNodes += rNodes
      b.pop()
    return (countLeaf, countNodes)
  start_time = time.time()
  countGame, countNodes = aux(b)
  print("BrutForce nombre de partie :", countGame, "Nombre de nœuds :", countNodes)
  print("Explorer en %0.2f seconds" % (time.time() - start_time))
  return

def miniMax(b):
  def aux(b, d):
    countLeaf = 0
    countNodes = 1
    bestScore = None
    for move in b.legal_moves():
      b.push(move)
      if b.is_game_over():
        countLeaf += 1
        countNodes += 1
        bestScore = getscore(b)
      else:
        rLeaf, rNodes, score = aux(b, d + 1)
        countLeaf += rLeaf
        countNodes += rNodes
        if bestScore == None:
          bestScore = score
        else:
          bestScore = d % 2 == 0 if max(bestScore, score) else min(bestScore, score)
      b.pop()
    return (countLeaf, countNodes, bestScore)
  start_time = time.time()
  countGame, countNodes, bestScore = aux(b, 0)
  print("MiniMax nombre de partie :", countGame, "Nombre de nœuds :", countNodes, "Score :", bestScore)
  print("Explorer en %0.2f seconds" % (time.time() - start_time))
  return

def miniMaxCut(b):
  # maxScore = alpha
  # minScore = beta
  def aux(b, d, maxScore, minScore):
    countLeaf = 0
    countNodes = 1
    if b.is_game_over():
      countLeaf += 1
      countNodes += 1
      score = getscore(b)
      return (countLeaf, countNodes, score)
    for move in b.legal_moves():
      b.push(move)
      rLeaf, rNodes, rScore = aux(b, d + 1, maxScore, minScore)
      b.pop()
      countLeaf += rLeaf
      countNodes += rNodes
      if d % 2 == 0:
        maxScore = max(maxScore, rScore)
        if maxScore >= minScore:
          return (countLeaf, countNodes, minScore)
      else:
        minScore = min(minScore, rScore)
        if maxScore >= minScore:
          return (countLeaf, countNodes, maxScore)
    return (countLeaf, countNodes, d % 2 == 0 if maxScore else minScore)
  start_time = time.time()
  countGame, countNodes, bestScore = aux(b, 0, -2, 2)
  print("miniMaxCut nombre de partie :", countGame, "Nombre de nœuds :", countNodes, "Score :", bestScore)
  print("Explorer en %0.2f seconds" % (time.time() - start_time))
  return

def main():
  board = Tictactoe.Board()

  brutForce(board)

  miniMax(board)

  miniMaxCut(board)

  return 0

if __name__ == '__main__':
  sys.exit(main())
