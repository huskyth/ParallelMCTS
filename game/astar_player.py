import copy

from game.chess.common import BLACK, WHITE, get_neighbours, shiftOutChessman, DISTANCE

# difficulty of the game
DEEPEST_LEVEL = 3


def getScore(pointStatus):
    score = 0
    scoreLevel = [1, 2, 4, 6]
    black = [x for x in DISTANCE if x == BLACK]
    # if chessman was eaten, sub 8 score for each one
    score -= 8 * (6 - len(black))
    for chessman, color in enumerate(pointStatus):
        advantg = 0
        disadvtg = 0
        neighboors = get_neighbours(chessman, DISTANCE)
        for eachNeighboor in neighboors:
            # computer use black chessman as default
            if pointStatus[eachNeighboor] == BLACK and color == WHITE:
                advantg += 1
                score += scoreLevel[advantg - 1]
            elif pointStatus[eachNeighboor] == WHITE and color == BLACK:
                disadvtg += 1
                score -= scoreLevel[disadvtg - 1]
            else:
                pass
            # unnecessary
            '''
            elif color == data.WHITE:
                if pointStatus[eachNeighboor] == data.BLACK:
                    score += 2
                elif pointStatus[eachNeighboor] == data.WHITE:
                    score -= 2
            '''
    return score

def computerMove(pointStatus, level):
    move = []
    maxScore = -48
    bestMove = None
    # for convenient, set color = computer color (black) when enter the
    # function firstly
    if level % 2 == 1:
        selfColor = BLACK
        opponentColor = WHITE
    else:
        selfColor = WHITE
        opponentColor = WHITE
    # In the deepest level, the best move is itself, replace it with None
    if level > DEEPEST_LEVEL:
        score = getScore(pointStatus)
        return [], score
    else:
        for chessman, color in enumerate(pointStatus):
            if color == selfColor:
                for neighboorChessman in get_neighbours(chessman, DISTANCE):
                    if pointStatus[neighboorChessman] == 0:
                        move.append((chessman, neighboorChessman))
        if not move:
            return [], -49
        bakPointStatus = copy.deepcopy(pointStatus)
        for eachMove in move:
            pointStatus[eachMove[1]] = selfColor
            pointStatus[eachMove[0]] = 0
            pointStatus = shiftOutChessman(pointStatus, DISTANCE)
            # newMove is useless, just for return the best move in the first
            # level
            newMove, score = computerMove(pointStatus, DISTANCE, level + 1)
            if score > maxScore:
                maxScore = score
                bestMove = eachMove
            # revoke the change
            pointStatus = copy.deepcopy(bakPointStatus)
        return bestMove, maxScore


class AStarPlayer:

    @staticmethod
    def select(pont):
        return computerMove(pont, 1)