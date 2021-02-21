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

    def plot3(self, file_name1, file_name2, file_name3, last_N=100):

        self.plot1(file_name=file_name1, last_N=100, color='red')
        self.plot1(file_name=file_name2, last_N=100, color='green')
        self.plot1(file_name=file_name3, last_N=100, color='blue')

    def plot2(self,
              file_name1="safe_lunar_env_without_shield",
              file_name2="safe_lunar_env_shield_yes"):
        metadata_1 = load_pickle(file_name1)
        metadata_2 = load_pickle(file_name2)
        score_1 = metadata_1['return']
        avg_reward = metadata_1['avg_return']

        score_2 = metadata_2['return']
        avg_reward2 = metadata_2['avg_return']

        episodes_1 = range(len(score_1))
        episodes_2 = range(len(score_2))

        plt.plot(episodes_1, avg_reward, color='red')
        plt.plot(episodes_2, avg_reward2, color='green')

        # title('something')
        plt.xlabel("episodes")
        plt.ylabel("return")
        plt.legend(loc='best')
        plt.show()

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
        # title('something')

    def show(self):
        plt.xlabel("episodes")
        plt.ylabel("average mean")
        plt.legend(loc='best')
        matplotlib.rcParams['font.size'] = 10
        plt.show()


if __name__ == '__main__':
    pp = Plotter()

    # pp.demo(file_name="default")
    # pp.plot1(file_name="shield_yes_seems_good")
    # pp.plot2("without_shield_seems_okay", "shield_yes_seems_good")

    # pp.plot3("without_shield_seems_okay", "shield_yes_seems_good",
    #          "updates_seem_yasuo")

    pp.plot3("without_shield_seems_okay",
             "shield_yes_seems_good",
             "shield_updates_latest",
             last_N=400)

    # pp.plot1(file_name="without_shield_seems_okay")
    pp.show()