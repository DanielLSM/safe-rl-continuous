# Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
# Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power.
# Engine can't work with less than 50% power.
# Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.
import math
import random
import gym
import numpy as np
from shield import Shield
from conjugate_prior import NormalNormalKnownVar
from utils import shift_interval, average, kl_divergence


class SafeLunarEnvUpdates(gym.Wrapper):
    def __init__(self, env, shield=None, update_shield_every_n_episode=100):
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
        self.bad_action = None
        self.internal_episode = -1
        self.update_shield_every_n_episode = update_shield_every_n_episode
        self.user_feedback_shield = UserFeedbackShield()
        self.number_of_bad_engine = 0
        self.landed_inside = 0
        self.crashed = 0
        self.weird = 0
        self.n_step = 0
        # info_to_send = {'number_of_bad_engine_uses': self.number_of_bad_engine}
        # self.observation_space.shape[0] = env.observation_space.shape[0]

    def step(self, action):
        self.n_step += 1
        if self.shield:
            # print("never happens")
            action = self.shield.shield_action(action)
            # print("this never happens")
        next_state, reward, done, info = self.env.step(action)
        if done:
            if reward == 100:
                self.landed_inside += 1
            elif reward == -100:
                self.crashed += 1
            else:
                self.weird += 1

        if np.abs(action[0]) > 0.7:
            # if self.shield.mean < 0.7:
            #     import ipdb
            #     ipdb.set_trace()
            #inverse idea give penalty from 0 to 0.5, but not above
            self.number_of_bad_engine += 1
            self.bad_action = action
            penalty_ratio = shift_interval(0.7, 1, 0, 1, np.abs(action[0]))
            reward = reward - (2 * penalty_ratio)
            self.warning_state = penalty_ratio
            # print("bad reward {} and penalty ratio {}".format(
            #     reward, penalty_ratio))
            if penalty_ratio < 0:
                raise "what the fuck"
            #make a negative reward proportional to the power used
        else:
            self.warning_state = -1

        next_state = np.append(next_state, self.warning_state)

        if done:
            info_to_send = {
                'number_of_bad_engine_uses':
                self.number_of_bad_engine,
                'number_lands_inside':
                self.landed_inside,
                'shield_mean':
                self.shield.thresholds_main_engine if self.shield else None,
                'number_of_crashes':
                self.crashed,
                'number_of_weird':
                self.weird
            }
        else:
            info_to_send = info
        return next_state, reward, done, info_to_send

    def reset(self):
        print("steps {} number_of_bad_actions {}".format(
            self.n_step, self.number_of_bad_engine))
        self.number_of_bad_engine = 0
        self.n_step = 0
        self.warning_state = -1
        self.internal_episode += 1

        if (self.bad_action is not None) and self.shield is not None:
            result = self.user_feedback_shield.update_oracle_with_last_action(
                self.bad_action)
            if result:
                print("shield converged after {} episodes with last action {}".
                      format(self.internal_episode, self.bad_action))
            self.shield = self.user_feedback_shield.get_current_shield()

        self.bad_action = None

        first_state = self.env.reset()
        first_state = np.append(first_state, self.warning_state)
        return first_state


class UserFeedbackShield:
    def __init__(self):
        # https://stats.stackexchange.com/questions/237037/bayesian-updating-with-new-data
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        # self.shield_distribution_main_engine = NormalNormalKnownVar(
        #     0.00001, prior_mean=1, prior_var=0.0001)

        self.shield_distribution_main_engine = NormalNormalKnownVar(
            0.001, prior_mean=1, prior_var=0.001)

        self.oracle_main_engine = NormalNormalKnownVar(0.0001,
                                                       prior_mean=1,
                                                       prior_var=0.00001)
        self.KL_div = []

    def get_current_shield(self):
        return Shield(thresholds_main_engine=self.
                      shield_distribution_main_engine.sample())

    def update_oracle_with_last_action(self, last_action):
        if np.abs(last_action[0]) > 0.7:
            self.oracle_main_engine = NormalNormalKnownVar(
                0.0001,
                prior_mean=(self.oracle_main_engine.mean - 0.015),
                prior_var=0.0001)
            self.update_shield_main_from_oracle()
            return 0
        else:
            return 1

    def update_shield_main_from_oracle(self, baysian_updates=1):
        samples = []
        for _ in range(baysian_updates):
            samples.append(self.oracle_main_engine.sample())
        # self.shield_distribution_main_engine = self.shield_distribution_main_engine.update(
        #     [self.oracle_main_engine.sample()])
        for _ in range(baysian_updates):
            self.shield_distribution_main_engine = self.shield_distribution_main_engine.update(
                samples)
        #TODO: somehow, the updates are slow lol
        # print(_, self.shield_distribution_main_engine.mean, samples)

        # print(
        #     "{} mean of main shield distrib and {} mean of main oracle distrib and a samples is like {}"
        #     .format(self.shield_distribution_main_engine.mean,
        #             self.oracle_main_engine.mean, samples))

    def update_shield(self, last_action):
        result = self.update_oracle_with_last_action(last_action)
        if result:
            print("shield converged after {} episodes with last action {}".
                  format(_, last_action))
        return result

    def demo_updates(self):
        for _ in range(1000):
            last_action = [self.shield_distribution_main_engine.sample()]
            result = self.update_oracle_with_last_action(last_action)
            if result:
                print("shield converged after {} episodes with last action {}".
                      format(_, last_action))
                break

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

    def demo_with_people(self):
        from matplotlib import pyplot as plt
        for _ in range(1000):
            last_action = [self.shield_distribution_main_engine.sample()]
            result = self.update_with_people(last_action)
            if result:
                print("shield converged after {} episodes with last action {}".
                      format(_, last_action))
                break
            self.shield_distribution_main_engine.plot(0.5, 1)

        plt.title(
            "shield_distrib, 10 oracles, 20 coin flippers converged after 156 loops"
        )
        plt.show()

        #TODO:KL
        # plt.title("KL_div")
        # print(self.KL_div[1:])
        # plt.plot(self.KL_div[1:])
        # plt.show()

    def update_with_people(self,
                           last_action,
                           number_of_oracles=10,
                           n_coin_flippers=20,
                           size_of_possibilities=3):
        self.previous_oracle = self.oracle_main_engine

        mean = self.oracle_main_engine.mean
        sig_squared = self.oracle_main_engine.var
        mapping_func = [
            mean - 3 / 2 * math.sqrt(sig_squared), mean,
            mean + 3 / 2 * math.sqrt(sig_squared)
        ]
        # mapping_func = [
        #     mean - (3 * sig_squared) / 2, mean, mean + (3 * sig_squared) / 2
        # ]

        scores = []

        if np.abs(last_action[0]) > 0.7:
            for _ in range(number_of_oracles):
                scores.append(mapping_func[0])
            for _ in range(n_coin_flippers):
                scores.append(random.choice(mapping_func))

            # import ipdb
            # ipdb.set_trace()
            new_mean = average(scores)

            print(new_mean)
            self.oracle_main_engine.mean
            self.oracle_main_engine = NormalNormalKnownVar(
                0.0001, prior_mean=(new_mean), prior_var=0.0001)

            u1 = self.previous_oracle.mean
            sig1 = math.sqrt(self.previous_oracle.var)
            u2 = self.oracle_main_engine.mean
            sig2 = math.sqrt(self.oracle_main_engine.var)
            print(u1, sig1, u2, sig2)
            kl_div = kl_divergence(u1, sig1, u2, sig2)
            print(u1, u2)
            self.KL_div.append(kl_div)

            self.update_shield_main_from_oracle()

            return 0
        else:
            # u1 = self.previous_oracle.mean
            # sig1 = math.sqrt(self.previous_oracle.var)
            # u2 = self.oracle_main_engine.mean
            # sig2 = math.sqrt(self.oracle_main_engine.var)
            # kl_div = kl_divergence(u1, sig1, u2, sig2)
            # print(kl_div)
            # self.KL_div.append(kl_div)

            return 1


if __name__ == '__main__':
    ufs = UserFeedbackShield()
    # ufs.demo_updates()
    ufs.demo_with_people()
    # ufs.demo()