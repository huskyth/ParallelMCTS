import copy
import json



def check(chessman, distance, pointStatus, checkedChessmen):
    checkedChessmen.append(chessman)
    dead = True
    neighboorChessmen = get_neighbours(chessman, distance)
    for neighboorChessman in neighboorChessmen:
        if neighboorChessman not in checkedChessmen:
            # if the neighboor is the same color, check the neighboor to find a
            # empty neighboor
            if pointStatus[neighboorChessman] == pointStatus[chessman]:
                dead = check(neighboorChessman, distance,
                             pointStatus, checkedChessmen)
                if dead == False:
                    return dead
            elif pointStatus[neighboorChessman] == 0:
                dead = False
                return dead
            else:
                pass
    return dead


DISTANCE = [[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]]
def shiftOutChessman(pointStatus, distance):
    deadChessmen = []
    bakPointStatus = copy.deepcopy(pointStatus)
    for chessman, color in enumerate(pointStatus):
        checkedChessmen = []
        dead = True
        if color != 0:
            # pdb.set_trace()
            dead = check(chessman, distance, pointStatus, checkedChessmen)
        else:
            pass
        if dead:
            deadChessmen.append(chessman)
        pointStatus = bakPointStatus
    for eachDeadChessman in deadChessmen:
        pointStatus[eachDeadChessman] = 0

    return pointStatus


def get_neighbours(chessman, distance):
    neighbour_chessmen = []
    for eachChessman, eachDistance in enumerate(distance[chessman]):
        if eachDistance == 1:
            neighbour_chessmen.append(eachChessman)
    return neighbour_chessmen

BLACK = 1
BLACK_COLOR = (0, 0, 0)
WHITE = -1

# difficulty of the game
DEEPEST_LEVEL = 4


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
        opponentColor = BLACK
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
            newMove, score = computerMove(pointStatus, level + 1)
            if score > maxScore:
                maxScore = score
                bestMove = eachMove
            # revoke the change
            pointStatus = copy.deepcopy(bakPointStatus)
        return bestMove, maxScore



if __name__ == '__main__':
    p = [0, 0, 0, 0, 0., -1, 0, 0, 0, 1, 0, 0., -1, 0, 0., -1, 1.,
  1, 1, 1, 1.]
    print(len(p))
    print(computerMove(p, 2))