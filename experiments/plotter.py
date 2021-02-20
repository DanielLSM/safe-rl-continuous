import matplotlib
import matplotlib.pyplot as plt

from utils import load_pickle

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Latin Modern Math'
matplotlib.rcParams['font.size'] = 10

# plt.figure(figsize=(3.5, HEIGHT)) one-column
# figsize=(7., HEIGHT) double-colum


class Plotter:
    def __init__(self):
        pass

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

    def plot1(self, file_name):
        # plt.rc('font', family='sans-serif')
        metadata = load_pickle(file_name)
        score = metadata['return']
        avg_reward = metadata['avg_return']
        episodes = range(len(score))
        plt.plot(episodes, score)
        plt.plot(episodes, avg_reward)
        # title('something')
        plt.xlabel("episodes")
        plt.ylabel("return")
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    pp = Plotter()
    # pp.demo(file_name="bipedal")
    # pp.demo(file_name="default")
    # pp.plot1(file_name="shield_yes_seems_good")
    pp.plot2("without_shield_seems_okay", "shield_yes_seems_good")