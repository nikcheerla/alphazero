import random


def drawBoard(board):
    # This function prints out the board that it was passed.

    # "board" is a list of 10 strings representing the board (ignore index 0)
    print('   |   |')
    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])
    print('   |   |')

move_stack = []
def makeMove(board, letter, move):
    global prevMove

    board[move] = letter
    move_stack.append(move)

def undoMove(board):
    board[move_stack.pop()] = ' '

def isWinner(bo, le):
    # Given a board and a player's letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don't have to type as much.
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
    (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
    (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
    (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
    (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
    (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal

def isSpaceFree(board, move):
    # Return true if the passed move is free on the passed board.
    return board[move] == ' '

def validMoves(board):
    return [i for i in range(1, 10) if isSpaceFree(board, i)]

def isGameOver(board):
    # Return True if every space on the board has been taken. Otherwise return False.
    if isWinner(board, 'X') or isWinner(board, 'O'): return True
    for i in range(1, 10):
        if isSpaceFree(board, i):
            return False
    return True

def getScore(board, letter):
    status = 0
    if isWinner(board, letter):
        status = 1
    elif isWinner(board, turn(letter)):
        status = -1
    return status

def turn(letter):
    return 'X' if letter == 'O' else 'O'
