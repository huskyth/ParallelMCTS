import copy
import json
import os

import cv2
import numpy
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import torch

MAX_HISTORY_STEPS = 8


def create_directory(path):
    if not os.path.exists(str(path)):
        os.mkdir(str(path))


def get_neighbours(chessman, distance):
    neighbour_chessmen = []
    for eachChessman, eachDistance in enumerate(distance[chessman]):
        if eachDistance == 1:
            neighbour_chessmen.append(eachChessman)
    return neighbour_chessmen


def get_distance():
    try:
        f = open(DISTANCE_PATH, 'rb')
        distance = json.loads(f.read())
        return distance
    except Exception as e:
        print(f'file open error {e}')
        return None
    finally:
        f.close()


def get_map():
    try:
        f = open(MAP_PATH, 'rb')
        point_pos = json.loads(f.read())
        return point_pos
    except Exception as e:
        print(f'file open error {e}')
        return None
    finally:
        f.close()


# TODO://对于7x7的矩阵映射关系
"""
key表示21长度数组的索引，也就是棋盘的索引，value表示tensor的row,col
"""
ARRAY_TO_IMAGE = {
    0: (0, 3), 15: (6, 3), 6: (3, 0), 10: (3, 6),
    1: (0, 2), 3: (0, 4), 2: (1, 3),
    4: (2, 0), 7: (4, 0), 5: (3, 1),
    8: (2, 6), 9: (3, 5), 11: (4, 6),
    12: (5, 3), 13: (6, 2), 14: (6, 4),
    20: (3, 3),
    16: (2, 3), 17: (3, 2), 18: (3, 4), 19: (4, 3)
}
IMAGE_TO_ARRAY = {v: k for k, v in ARRAY_TO_IMAGE.items()}

def from_torch_to_array(tensor):
    ret = [0] * 21
    for i in range(7):
        for j in range(7):
            if tensor[i][j] != 0:
                ret[IMAGE_TO_ARRAY[(i, j)]] = tensor[i][j]
    return ret



def from_array_to_input_tensor(point_status, current_player, last_action_list):
    """
        :param point_status:
        :param current_player:
        :return: 返回(7, 7, 3)的张量，第三个维度的第一个维度为执棋方
                                    第三个维度的第二个为对手
                                    第三个维度的第二个指示
                                    第三个维度的第二个为棋手
    """
    is_cuda = True if torch.cuda.is_available() else False
    if current_player not in [-1, 1]:
        raise Exception('current_player must be -1 or 1')

    if not isinstance(point_status, numpy.ndarray) and not isinstance(point_status, list):
        raise Exception(f'point_status must be list or numpy.ndarray, type {type(point_status)}')

    if len(point_status) != 21:
        raise Exception('point_status length must be 21')

    v = list(ARRAY_TO_IMAGE.values())
    v_ind = np.stack(v)
    state_not = np.ones((7, 7)) * 2
    state_not[v_ind[:, 0], v_ind[:, 1]] = 0

    input_tensor = torch.zeros((7, 7, 3 + MAX_HISTORY_STEPS))

    input_tensor[:, :, 0][state_not == 2] = 2
    input_tensor[:, :, 1][state_not == 2] = 2
    input_tensor[:, :, 2] = 1 if 1 == current_player else 0
    for i, chessman in enumerate(point_status):
        row, column = ARRAY_TO_IMAGE[i]
        if chessman == current_player:
            assert input_tensor[row, column, 0].item() != 2
            input_tensor[row, column, 0] = 1
        elif chessman == -current_player:
            assert input_tensor[row, column, 1].item() != 2
            input_tensor[row, column, 1] = 1
        else:
            assert chessman == 0

    for i, last_action in enumerate(last_action_list):
        if last_action != (-1, -1):
            _, to = last_action
            row, column = ARRAY_TO_IMAGE[to]
            input_tensor[row, column, i + 3] = 1

    return input_tensor.cuda() if is_cuda else input_tensor


def write_image(name, image):
    name = str(name)
    cv2.imwrite(f"{name}.png", image)


def read_image(path):
    return cv2.imread(path)


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


def write_video(frame_list, file_name, fps=0.5):
    if len(frame_list) == 0:
        return
    image = frame_list[0]
    assert isinstance(image, np.ndarray)
    result = cv2.VideoWriter(f"{file_name}.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps,
                             (int(image.shape[1]), int(image.shape[0])))

    for i, frame in enumerate(frame_list):
        result.write(frame)
    result.release()


def write_msg(msg, path, is_append=True):
    mode = 'a' if is_append else 'w'
    with open(path, mode) as file:
        file.write(msg + "\n")


def bar_show(x, y, is_show=False, name="test.png"):
    plt.grid()
    plt.xticks(x)
    plt.bar(x, y)
    plt.savefig(name)
    if is_show:
        plt.show()
    plt.close()


def serialize(path, value):
    torch.save(value, path)


def deserialize(path):
    value = torch.load(path)
    return value


ROOT_PATH = Path(os.path.abspath(__file__)).parent.parent

BLACK = 1
BLACK_COLOR = (0, 0, 0)
WHITE = -1
WHITE_COLOR = (255, 255, 255)

LENGTH_OF_BOARD = 21

DISTANCE_PATH = str(ROOT_PATH / 'chess/data/distance.txt')
MAP_PATH = str(ROOT_PATH / 'chess/data/pointPos.txt')

DISTANCE = get_distance()
GAME_MAP = get_map()
MOVE_TO_INDEX_DICT = {}
INDEX_TO_MOVE_DICT = {}
MOVE_LIST = []
# MOVE_LIST从小到大排列
for from_point in range(21):
    to_point_list = get_neighbours(from_point, DISTANCE)
    to_point_list = sorted(to_point_list)
    for to_point in to_point_list:
        MOVE_LIST.append((from_point, to_point))
for idx, move_tuple in enumerate(MOVE_LIST):
    MOVE_TO_INDEX_DICT[move_tuple] = idx
    INDEX_TO_MOVE_DICT[idx] = move_tuple

if __name__ == '__main__':
    # from pprint import pprint
    #
    # ary = 21 * [0]
    # ary[10] = 1
    # data = from_array_to_input_tensor(ary, -1).data
    #
    # pprint(data[:, :, 0])
    # pprint(data[:, :, 1])
    import numpy as np
    import torch

    v = list(ARRAY_TO_IMAGE.values())
    v_ind = np.stack(v)
    state = np.ones((7, 7))
    state[v_ind[:, 0], v_ind[:, 1]] = 0
    print(state)
