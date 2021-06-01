import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from utils import load_pickle, moving_average

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Latin Modern Math'
matplotlib.rcParams['font.size'] = 10

# plt.figure(figsize=(3.5, HEIGHT)) one-column
# figsize=(7., HEIGHT) double-colum


class Plotter:
    def __init__(self):
        plt.figure(figsize=(3.5, 2.5))

    def plot3(self, file_name1, file_name2, file_name3, last_N=100):

        self.plot1(file_name=file_name1, last_N=100, color='red')
        self.plot1(file_name=file_name2, last_N=100, color='green')
        self.plot1(file_name=file_name3, last_N=100, color='blue')

    def plot2(self,
              file_name1="safe_lunar_env_without_shield",
              file_name2="safe_lunar_env_shield_yes"):

        self.plot1(file_name=file_name1, last_N=100, color='red')
        self.plot1(file_name=file_name2, last_N=100, color='green')

    def plot1(self, file_name, last_N=100, color='blue'):
        # plt.rc('font', family='sans-serif')
        metadata = load_pickle(file_name)
        score = metadata['return']
        episodes = range(len(score))
        mean, std = moving_average(score, last_N=last_N)
        lower_bound = [a_i - 2 * b_i for a_i, b_i in zip(mean, std)]
        upper_bound = [a_i + 2 * b_i for a_i, b_i in zip(mean, std)]
        # plt.plot(episodes, score)
        plt.fill_between(episodes,
                         lower_bound,
                         upper_bound,
                         facecolor=color,
                         alpha=0.5)
        plt.plot(episodes, mean, color=color)
        plt.xlabel("episodes")
        plt.ylabel("average mean")

    def plot_general(self,
                     file_name,
                     param_name,
                     last_N=100,
                     color='blue',
                     limit_x=None,
                     range_y=None,
                     y_ticks=None):
        # plt.rc('font', family='sans-serif')
        metadata = load_pickle(file_name)
        # import ipdb
        # ipdb.set_trace()
        score = metadata[param_name]
        print(score[-1])
        mean, std = moving_average(score, last_N=last_N)
        if limit_x is not None:
            episodes = range(limit_x)
            mean = mean[:limit_x]
            std = std[:limit_x]
        else:
            episodes = range(len(score))
            mean, std = moving_average(score, last_N=last_N)

        lower_bound = [a_i - 1 * b_i for a_i, b_i in zip(mean, std)]
        upper_bound = [a_i + 1 * b_i for a_i, b_i in zip(mean, std)]
        # plt.plot(episodes, score)
        plt.fill_between(episodes,
                         lower_bound,
                         upper_bound,
                         facecolor=color,
                         alpha=0.5)
        plt.plot(episodes, mean, color=color)
        if range_y is not None:
            plt.ylim(range_y)
        if y_ticks is not None:
            plt.yticks(np.arange(range_y[0], range_y[1] + 2 * y_ticks,
                                 y_ticks))
        plt.xlabel("episodes")
        plt.ylabel(param_name)

    def show(self):
        plt.legend(loc='best')
        matplotlib.rcParams['font.size'] = 10
        plt.show()


if __name__ == '__main__':
    pp = Plotter()

    # pp.plot3("without_shield_seems_okay",
    #          "shield_yes_seems_good",
    #          "shield_updates_latest",
    #          last_N=400)

    # shield_updates_latest21-02-2021 18:34:53
    # without_shield_21-02-2021 19:37:54
    # perfect_shield_21-02-2021 19:38:06

    # 'score'
    # 'avg_return'
    # 'shield_means'
    # 'number_of_bad_engine_uses'
    # 'landed_inside'

    pp.plot_general(
        file_name='shield_updates_latest_27-02-2021 14:26:18',
        param_name='shield_means',
        last_N=1,
        color='red',
        # limit_x=5000,
        range_y=None,
        y_ticks=None)

    # pp.plot_general(file_name='shield_updates_latest21-02-2021 18:34:53',
    #                 param_name='number_of_bad_engine_uses',
    #                 last_N=1,
    #                 color='green',
    #                 limit_x=5000,
    #                 range_y=None,
    #                 y_ticks=None)

    # pp.plot_general(file_name='shield_updates_latest21-02-2021 18:34:53',
    #                 param_name='shield_means',
    #                 last_N=1,
    #                 limit_x=1000,
    #                 range_y=[0.4, 1],
    #                 y_ticks=0.1)

    # def plot_mv_avgs(self, *files, last_N=50):
    #     metadata = []
    #     for _ in range(len(files)):
    #         metadata.append(load_pickle(files[_]))

    #     scores = []
    #     episodes = []
    #     avg_rewards = []

    #     for _ in range(len(metadata)):
    #         scores.append(metadata[_]['return'])
    #         avg_rewards.append(moving_average(rewards=scores[_],
    #                                           last_N=last_N))
    #         episodes.append(range(len(scores[_])))

    #     for _ in range(len(metadata)):
    #         plt.
    pp.show()