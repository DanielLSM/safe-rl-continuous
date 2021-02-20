import numpy as np
import pickle as pkl
from datetime import datetime


def save_pickle(obj, name):
    with open("data/" + name + ".pkl", 'wb') as handle:
        pkl.dump(obj, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open("data/" + name + ".pkl", 'rb') as handle:
        return pkl.load(handle)


def shift_interval(from_a, from_b, to_c, to_d, t):
    shift = to_c + ((to_d - to_c) / (from_b - from_a)) * (t - from_a)
    shift = max(shift, to_c)
    shift = min(shift, to_d)
    return shift


def return_date():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    return dt_string


if __name__ == "__main__":
    # a = [1, 2, 3, 3, 4]
    # b = {"data": [1, 2, 3, 4]}
    # save_pickle(a, "test_list")
    # save_pickle(b, "test_dict")

    # test_list = load_pickle("test_list")
    # test_dict = load_pickle("test_dict")

    # print(test_list)
    # print(test_dict)
    # print(load_pickle("default"))

    # print(shift_interval(0, 1, 0.9, 1, 0.5))

    # print(shift_interval(0.9, 1, 0, 1, 0.5))

    action = np.array([0.95, 0])
    reward = 0
    if np.abs(action[0]) > 0.9:
        penalty_ratio = shift_interval(0.9, 1, 0, 1, np.abs(action[0]))
        reward = reward - 20 * penalty_ratio
        warning_state = penalty_ratio

    print(penalty_ratio)
    print(reward)