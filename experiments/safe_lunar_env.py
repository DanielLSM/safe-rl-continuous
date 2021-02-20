# Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
# Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
# Engine can't work with less than 50% power.
# Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
import gym
import numpy as np
from shield import Shield
from conjugate_prior import NormalNormalKnownVar
from utils import shift_interval


class SafeLunarEnv(gym.Wrapper):
    def __init__(self, env, shield=None):
        super().__init__(env)
        self.env = env
        if shield != None:
            self.shield = shield
        else:
            self.shield = None
        # self.exploded = 0
        self.steps_to_explosion = 20
        self.warning_state = 0
        self.warning_state_fract = 1 / self.steps_to_explosion
        self.action_size = self.action_space.shape[0]
        self.state_size = self.observation_space.shape
        stat_size = list(self.state_size)
        stat_size[0] += 1
        self.state_size = tuple(stat_size)  #its the warning state

        # self.observation_space.shape[0] = env.observation_space.shape[0]

    def step(self, action):
        if self.shield:
            action = self.shield.shield_action(action)
            # print("this never happens")
        next_state, reward, done, info = self.env.step(action)
        # print(next_state)
        # done_explosion, reward_explosion = self.check_explosion(*action)
        # import ipdb
        # ipdb.set_trace()

        if np.abs(action[0]) > 0.7:
            #inverse idea give penalty from 0 to 0.5, but not above
            print("eat shit")
            penalty_ratio = shift_interval(0.7, 1, 0, 1, np.abs(action[0]))
            reward = reward - (2 * penalty_ratio)
            self.warning_state = penalty_ratio
            print(reward)
            #make a negative reward proportional to the power used
        else:
            self.warning_state = -1

        next_state = np.append(next_state, self.warning_state)
        # print(next_state)
        # done = done or done_explosion
        # reward = reward + reward_explosion
        # print(self.steps_to_explosion)
        return next_state, reward, done, info

    def reset(self):
        self.steps_to_explosion = 20
        self.warning_state = -1
        first_state = self.env.reset()
        first_state = np.append(first_state, self.warning_state)
        return first_state

    def check_explosion(self, *action):
        if np.abs(action[0]) > 0.9:
            self.steps_to_explosion -= 1
        if self.steps_to_explosion == 0:
            return True, -100
        return False, 0


class UserFeedbackShield:
    def __init__(self):
        # https://stats.stackexchange.com/questions/237037/bayesian-updating-with-new-data
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        self.shield_distribution_main_engine = NormalNormalKnownVar(
            1, prior_mean=1, prior_var=0.01)
        self.shield_distribution_left_engine = NormalNormalKnownVar(
            1, prior_mean=-1, prior_var=0.01)
        self.shield_distribution_right_engine = NormalNormalKnownVar(
            1, prior_mean=1, prior_var=0.01)

        self.oracle_main_engine = self.shield_distribution_right_engine = NormalNormalKnownVar(
            1, prior_mean=1, prior_var=0.001)
        self.oracle_left_engine = self.shield_distribution_right_engine = NormalNormalKnownVar(
            1, prior_mean=-1, prior_var=0.001)
        self.oracle_right_engine = self.shield_distribution_right_engine = NormalNormalKnownVar(
            1, prior_mean=1, prior_var=0.001)

    def get_current_shield(self):
        return Shield(thresholds_main_engine=self.
                      shield_distribution_main_engine.sample(),
                      thresholds_left_engine=self.
                      shield_distribution_left_engine.sample(),
                      thresholds_right_engine=self.
                      shield_distribution_right_engine.sample())

    def update_oracle_with_last_action(self, last_action, mode='all'):
        modes = ['left', 'left_right', 'all']
        assert mode in modes

        if np.abs(last_action[1]) < -0.8:
            self.oracle_left_engine = NormalNormalKnownVar(
                0.01,
                prior_mean=(self.oracle_left_engine.mean + 0.05),
                prior_var=0.01)
            self.update_shield_left_from_oracle()

        if np.abs(last_action[1]) > 0.8 and (mode == 'left_right'
                                             or mode == 'all'):
            self.oracle_left_engine = NormalNormalKnownVar(
                0.01,
                prior_mean=(self.oracle_right_engine.mean - 0.05),
                prior_var=0.01)
            self.update_shield_right_from_oracle()

        if np.abs(last_action[0]) > 0.9 and mode == 'all':
            self.oracle_left_engine = NormalNormalKnownVar(
                0.01,
                prior_mean=(self.oracle_main_engine.mean - 0.05),
                prior_var=0.01)
            self.update_shield_main_from_oracle()

    def update_shield_left_from_oracle(self):
        self.shield_distribution_left_engine = self.shield_distribution_left_engine.update(
            [self.oracle_left_engine.sample()])

    def update_shield_right_from_oracle(self):
        self.shield_distribution_right_engine = self.shield_distribution_right_engine.update(
            [self.oracle_right_engine.sample()])

    def update_shield_main_from_oracle(self):
        self.shield_distribution_main_engine = self.shield_distribution_main_engine.update(
            [self.oracle_main_engine.sample()])

    # def create_oracle

    def demo(self):
        import numpy as np
        from matplotlib import pyplot as plt

        from conjugate_prior import NormalNormalKnownVar
        model = NormalNormalKnownVar(1)
        model.plot(-5, 5)
        plt.show()
        new_model = model

        for _ in range(10):
            new_model = NormalNormalKnownVar(0.01,
                                             prior_mean=(new_model.mean +
                                                         0.05),
                                             prior_var=0.01)
            model = model.update([new_model.sample()])
            model.plot(-5, 5)
        print(model.sample())
        plt.show()
