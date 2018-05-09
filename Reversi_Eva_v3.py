# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:45:37 2018

@author: ck
"""

'''
update:
    估值函数为权值表+行动力+稳定子
    对于较顶层节点使用一步预搜索获得估值排序后的前maxN个最优操作，限制搜索宽度并尽量提前剪枝

test:
    潜在行动力无助于提升性能
    
'''

import json
import numpy

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量

# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

def getmoves(board, color):
    moves = []
    ValidBoardList = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                newBoard = board.copy()
                if place(newBoard, i, j, color):
                    moves.append((i, j))
                    ValidBoardList.append(newBoard)
    return moves, ValidBoardList

def getbound(board, color):
    bound1 = 0
    bound2 = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                flag1 = 0
                flag2 = 0
                for d in range(8):
                    ii = i + DIR[d][0]
                    jj = j + DIR[d][1]
                    if 0 <= ii and ii < 8 and 0 <= jj and jj < 8:
                        if flag1 == 0 and board[ii][jj] == -color:
                            flag1 = 1
                            bound1 += 1
                        if flag2 == 0 and board[ii][jj] == color:
                            flag2 = 1
                            bound2 += 1
    return bound1, bound2

def getstable(board, color):
    stable = [0,0,0]
    # 角, 边, 八个方向都无空格
    cind1 = [0,0,7,7]
    cind2 = [0,7,7,0]
    inc1 = [0,1,0,-1]
    inc2 = [1,0,-1,0]
    stop = [0,0,0,0]
    for i in range(4):
        if board[cind1[i]][cind2[i]] == color:
            stop[i] = 1
            stable[0] += 1
            for j in range(1,7):
                if board[cind1[i]+inc1[i]*j][cind2[i]+inc2[i]*j] != color:
                    break
                else:
                    stop[i] = j + 1
                    stable[1] += 1
    for i in range(4):
        if board[cind1[i]][cind2[i]] == color:
            for j in range(1,7-stop[i-1]):
                if board[cind1[i]-inc1[i-1]*j][cind2[i]-inc2[i-1]*j] != color:
                    break
                else:
                    stable[1] += 1
    colfull = numpy.zeros((8, 8), dtype=numpy.int)
    colfull[:,numpy.sum(abs(board), axis = 0) == 8] = True
    rowfull = numpy.zeros((8, 8), dtype=numpy.int)
    rowfull[numpy.sum(abs(board), axis = 1) == 8,:] = True
    diag1full = numpy.zeros((8, 8), dtype=numpy.int)
    for i in range(15):
        diagsum = 0
        if i <= 7:
            sind1 = i
            sind2 = 0
            jrange = i+1
        else:
            sind1 = 7
            sind2 = i-7
            jrange = 15-i
        for j in range(jrange):
            diagsum += abs(board[sind1-j][sind2+j])
        if diagsum == jrange:
            for k in range(jrange):
                diag1full[sind1-j][sind2+j] = True
    diag2full = numpy.zeros((8, 8), dtype=numpy.int)
    for i in range(15):
        diagsum = 0
        if i <= 7:
            sind1 = i
            sind2 = 7
            jrange = i+1
        else:
            sind1 = 7
            sind2 = 14-i
            jrange = 15-i
        for j in range(jrange):
            diagsum += abs(board[sind1-j][sind2-j])
        if diagsum == jrange:
            for k in range(jrange):
                diag2full[sind1-j][sind2-j] = True
    stable[2] = sum(sum(numpy.logical_and(numpy.logical_and(numpy.logical_and(colfull, rowfull), diag1full), diag2full)))
    return stable

Vmap = numpy.array([[500,-25,10,5,5,10,-25,500],
                    [-25,-45,1,1,1,1,-45,-25],
                    [10,1,3,2,2,3,1,10],
                    [5,1,2,1,1,2,1,5],
                    [5,1,2,1,1,2,1,5],
                    [10,1,3,2,2,3,1,10],
                    [-25,-45,1,1,1,1,-45,-25],
                    [500,-25,10,5,5,10,-25,500]])

    
def mapweightsum(board, mycolor):
    return sum(sum(board*Vmap))*mycolor

def evaluation(moves, board, mycolor):
    moves_, ValidBoardList_ = getmoves(board, -mycolor)
    stable = getstable(board, mycolor)
	value = mapweightsum(board, mycolor) + 15*(len(moves)-len(moves_)) + 10*sum(stable)
    return int(value)

def onestepplace(board, mycolor):
    stage = sum(sum(abs(board)))
    if stage <= 9:
        depth = 5
    elif stage >= 50:
        depth = 6
    else:
        depth = 4
    value, bestmove = alphabetav2(board, depth, -10000, 10000, mycolor, mycolor, depth)
    return bestmove

def alphabetav2(board, depth, alpha, beta, actcolor, mycolor, maxdepth):
    moves, ValidBoardList = getmoves(board, actcolor)
    if len(moves) == 0:
        return evaluation(moves, board, mycolor), (-1, -1)
    if depth == 0:
        return evaluation(moves, board, mycolor), []
    
    if depth == maxdepth:
        for i in range(len(moves)):
            if Vmap[moves[i][0]][moves[i][1]] == Vmap[0][0] and actcolor == mycolor:
                return 1000, moves[i]
                
    #对于较顶层节点使用一步预搜索获得估值排序后的前maxN个最优操作，限制搜索宽度并尽量提前剪枝
    if depth >= 4:
        Vmoves = []
        for i in range(len(ValidBoardList)):
            value, bestmove = alphabetav2(ValidBoardList[i], 1, -10000, 10000, -actcolor, mycolor, maxdepth)
            Vmoves.append(value)
        ind = numpy.argsort(Vmoves)
        maxN = 6
        moves = [moves[i] for i in ind[0:maxN]]
        ValidBoardList = [ValidBoardList[i] for i in ind[0:maxN]]
    
    bestmove = []
    bestscore = -10000
    for i in range(len(ValidBoardList)):
        score, childmove = alphabetav2(ValidBoardList[i], depth-1, -beta, -max(alpha, bestscore), -actcolor, mycolor, maxdepth)
        score = -score
        if score > bestscore:
            bestscore = score
            bestmove = moves[i]
            if bestscore > beta:
                return bestscore, bestmove
    return bestscore, bestmove

# 处理输入，还原棋盘
def initBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = numpy.zeros((8, 8), dtype=numpy.int)
    board[3][4] = board[4][3] = 1 #白
    board[3][3] = board[4][4] = -1 #黑
    myColor = 1
    if requests[0]["x"] >= 0:
        myColor = -1
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor

board, myColor = initBoard()

move = onestepplace(board, myColor)
x, y = move
print(json.dumps({"response": {"x": x, "y": y}}))