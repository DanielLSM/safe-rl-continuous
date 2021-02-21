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


def moving_average(rewards, last_N):
    scores = []
    averages = []
    stds = []
    for _ in range(len(rewards)):
        scores.append(rewards[_])
        average = sum(scores[-last_N:]) / len(scores[-last_N:])
        averages.append(average)
        # import ipdb
        # ipdb.set_trace()
        x = scores[-last_N:]
        x_mean = averages[-last_N:]
        diff = []
        for _ in range(len(x)):
            diff.append((abs(x[_] - x_mean[_]))**(1 / 2))
        diff_sum = sum(diff)

        stds.append(diff_sum / last_N)

    return averages, stds


# def moving_average(rewards, last_N):
#     cumsum, moving_aves = [0], []

#     for i, x in enumerate(rewards, 1):
#         cumsum.append(cumsum[i - 1] + x)
#         if i >= last_N:
#             moving_ave = (cumsum[i] - cumsum[i - last_N]) / last_N
#             #can do stuff with moving_ave here
#             moving_aves.append(moving_ave)

#     return moving_aves

# def running_mean(x, N):
#     cumsum = np.cumsum(np.insert(x, 0, 0))
#     return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == "__main__":

    mo, stds = moving_average(rewards=[1, 2, 1, 2, 2, 2], last_N=2)
    print(mo, stds)

    # ma = running_mean([1, 2, 3, 4, 4, 5], 4)
    # print(ma)

    # ma = moving_average(rewards=[1, 2, 1, 2, 2, 2], last_N=3)
    # print(ma)

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
    # ========================================================
    # action = np.array([0.95, 0])
    # reward = 0
    # if np.abs(action[0]) > 0.9:
    #     penalty_ratio = shift_interval(0.9, 1, 0, 1, np.abs(action[0]))
    #     reward = reward - 20 * penalty_ratio
    #     warning_state = penalty_ratio

    # print(penalty_ratio)
    # print(reward)

    n = np.array([0, 1])
    n = None
    if n is not None:
        print("hello")
