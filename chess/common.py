import copy
import json
import os

import cv2
import numpy
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
import torch


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
ARRAY_TO_IMAGE = {
    0: (0, 3), 15: (6, 3), 6: (3, 0), 10: (3, 6),
    1: (0, 2), 3: (0, 4), 2: (1, 3),
    4: (2, 0), 7: (4, 0), 5: (3, 1),
    8: (2, 6), 9: (3, 5), 11: (4, 6),
    12: (5, 3), 13: (6, 2), 14: (6, 4),
    20: (3, 3),
    16: (2, 3), 17: (3, 2), 18: (3, 4), 19: (4, 3)
}


def from_array_to_input_tensor(numpy_array):
    assert len(numpy_array) == 21
    assert isinstance(numpy_array, numpy.ndarray)
    input_tensor = np.zeros((7, 7))
    for i, chessman in enumerate(numpy_array):
        row, column = ARRAY_TO_IMAGE[i]
        input_tensor[row, column] = chessman
    return input_tensor


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
MODEL_SAVE_PATH = str(ROOT_PATH / "temp")

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
    n = len(MOVE_TO_INDEX_DICT)
    data = np.random.normal(0, 1, n)
    x = np.arange(0, 72, 1)
    bar_show(x, data)
    max_action = np.argmax(data)
    print(max_action, data[max_action])
    print(INDEX_TO_MOVE_DICT[max_action])
