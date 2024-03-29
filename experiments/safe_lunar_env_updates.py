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
import pickle as pkl


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
        self.action = None
        # info_to_send = {'number_of_bad_engine_uses': self.number_of_bad_engine}
        # self.observation_space.shape[0] = env.observation_space.shape[0]

    def step(self, action):
        self.n_step += 1
        if self.shield:
            # print("never happens")
            action = self.shield.shield_action(action)
            # print("this never happens")
        self.action = action
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
                self.weird,
                'kl_div':
                self.user_feedback_shield.kl_div_current
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

        if ((self.crashed or self.bad_action is not None) or
                self.user_feedback_shield.shield_distribution_main_engine.mean
                < 0.7) and self.shield is not None:
            result = self.user_feedback_shield.update_shield(self.action)
            # if result:
            #     print("shield converged after {} episodes with last action {}".
            #           format(self.internal_episode, self.bad_action))
            if np.greater(self.user_feedback_shield.kl_div_current,
                          self.user_feedback_shield.minimum_kl_div):
                self.shield = self.user_feedback_shield.get_current_shield()

        self.bad_action = None
        self.action = None

        first_state = self.env.reset()
        first_state = np.append(first_state, self.warning_state)
        return first_state

    # def reset(self):
    #     print("steps {} number_of_bad_actions {}".format(
    #         self.n_step, self.number_of_bad_engine))
    #     self.number_of_bad_engine = 0
    #     self.n_step = 0
    #     self.warning_state = -1
    #     self.internal_episode += 1

    #     if (self.bad_action is not None) and self.shield is not None:
    #         result = self.user_feedback_shield.update_shield(self.bad_action)
    #         if result:
    #             print("shield converged after {} episodes with last action {}".
    #                   format(self.internal_episode, self.bad_action))
    #         self.shield = self.user_feedback_shield.get_current_shield()

    #     self.bad_action = None

    #     first_state = self.env.reset()
    #     first_state = np.append(first_state, self.warning_state)
    #     return first_state


class UserFeedbackShield:
    def __init__(self, number_of_oracles=10, n_coin_flippers=0):
        # https://stats.stackexchange.com/questions/237037/bayesian-updating-with-new-data
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        # self.shield_distribution_main_engine = NormalNormalKnownVar(
        #     0.00001, prior_mean=1, prior_var=0.0001)
        self.peeps_to_power = 1

        self.engine_start_var = 0.00005
        self.engine_known_var = 0.00005
        self.oracle_start_var = 0.00001
        self.oracle_known_var = 0.00001
        self.minimum_var = 0.00001
        self.minimum_kl_div = 0.005
        # self.kl_div_current = 1

        self.KL_div = []
        self.means_of_shield_distrib = [1]
        self.variences_of_shield_distrib = [self.engine_start_var]

        self.number_of_oracles = number_of_oracles
        self.n_coin_flippers = n_coin_flippers

        self.shield_distribution_main_engine = NormalNormalKnownVar(
            self.engine_known_var,
            prior_mean=1,
            prior_var=self.engine_start_var)

        self.oracle_main_engine = NormalNormalKnownVar(
            self.oracle_known_var,
            prior_mean=0.95,
            prior_var=self.oracle_known_var)

        self.kl_div_current = self.compute_kl_div(
            self.oracle_main_engine, self.shield_distribution_main_engine)

        # self.shield_distribution_main_engine = NormalNormalKnownVar(
        #     self.minimum_var, prior_mean=1, prior_var=self.minimum_var)

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

    def update_shield(self, last_action):
        if self.peeps_to_power:
            result = self.update_with_people(last_action)
        else:
            result = self.update_oracle_with_last_action(last_action)
        # if result:
        #     print("shield converged after {} episodes with last action {}".
        #           format(_, last_action))
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

    def update_shield_main_from_oracle(self, baysian_updates=1):
        samples = []
        for _ in range(baysian_updates):
            samples.append(self.oracle_main_engine.sample())
        # self.shield_distribution_main_engine = self.shield_distribution_main_engine.update(
        #     [self.oracle_main_engine.sample()])
        for _ in range(baysian_updates):
            self.shield_distribution_main_engine = self.shield_distribution_main_engine.update(
                samples)
            self.means_of_shield_distrib.append(
                self.shield_distribution_main_engine.mean)
            self.variences_of_shield_distrib.append(
                self.shield_distribution_main_engine.var)
            current_var = self.shield_distribution_main_engine.var
            self.shield_distribution_main_engine.var = max(
                current_var, self.minimum_var)
        #TODO: somehow, the updates are slow lol
        # print(_, self.shield_distribution_main_engine.mean, samples)

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
            # print(self.shield_distribution_main_engine.var)

        # plt.title(
        #     "shield_distrib, {} oracles, {} coin flippers converged after 156 loops"
        # )
        plt.show()

        # #TODO:KL
        # plt.title("KL_div")
        # print(self.KL_div[1:])
        # plt.plot(self.KL_div[1:])
        # plt.show()

    def compute_kl_div(self, dist1, dist2):
        u1 = dist1.mean
        sig1 = math.sqrt(dist1.var)
        u2 = dist2.mean
        sig2 = math.sqrt(dist2.var)
        kl_div = kl_divergence(u1, sig1, u2, sig2)
        print(u1, sig1, u2, sig2)
        print(kl_div)
        self.KL_div.append(kl_div)
        return kl_div

    def build_population_quiz(self, number_of_oracles, n_coin_flippers,
                              mapping_func, last_action):
        scores = []
        for _ in range(number_of_oracles):

            mean = self.oracle_main_engine.mean
            if mean > 0.7:
                scores.append(mapping_func[0])
            else:
                scores.append(mapping_func[1])
        for _ in range(n_coin_flippers):
            scores.append(random.choice(mapping_func))

        return scores

    def update_with_people(self,
                           last_action,
                           number_of_oracles=0,
                           n_coin_flippers=0,
                           size_of_possibilities=3):

        number_of_oracles = self.number_of_oracles
        n_coin_flippers = self.n_coin_flippers

        mean = self.oracle_main_engine.mean
        sig_squared = self.oracle_main_engine.var
        mapping_func = [
            mean - 3 / 2 * math.sqrt(sig_squared), mean,
            mean + 3 / 2 * math.sqrt(sig_squared)
        ]

        scores = self.build_population_quiz(number_of_oracles, n_coin_flippers,
                                            mapping_func, last_action)

        if np.abs(last_action[0]) > 0.7 or np.greater(self.kl_div_current,
                                                      self.minimum_kl_div):

            new_mean = average(scores)

            self.oracle_main_engine = NormalNormalKnownVar(
                self.minimum_var,
                prior_mean=(new_mean),
                prior_var=self.minimum_var)

            print("Var main engine", self.oracle_main_engine.var)
            # kl_div = self.compute_kl_div(self.previous_oracle,
            #                              self.oracle_main_engine)

            self.kl_div_current = self.compute_kl_div(
                self.oracle_main_engine, self.shield_distribution_main_engine)

            self.update_shield_main_from_oracle()

            return 0
        # elif np.greater(self.kl_div_current, self.minimum_kl_div):

        #     new_mean = average(scores)

        #     self.oracle_main_engine = NormalNormalKnownVar(
        #         self.minimum_var,
        #         prior_mean=(new_mean),
        #         prior_var=self.minimum_var)

        #     print("kl_div is {} self.minim_kl_div is {}".format(
        #         self.kl_div_current, self.minimum_kl_div))
        #     self.kl_div_current = self.compute_kl_div(
        #         self.oracle_main_engine, self.shield_distribution_main_engine)
        #     self.update_shield_main_from_oracle()
        #     return 0
        else:
            # self.kl_div_current = self.minimum_kl_div

            return 1


if __name__ == '__main__':

    # np.random.seed(10)

    np.random.seed(100)

    from matplotlib import pyplot as plt
    ufs = UserFeedbackShield(number_of_oracles=10, n_coin_flippers=0)
    # ufs.demo_updates()
    ufs.demo_with_people()
    #TODO:KL
    plt.title("KL_div oracles:{} flippers:{}".format(ufs.number_of_oracles,
                                                     ufs.n_coin_flippers))
    print(ufs.KL_div[1:])
    plt.plot(ufs.KL_div[1:])
    plt.show()

    kl_div1 = {"kl": ufs.KL_div[1:]}
    means_vars_10_0 = {
        "means": ufs.means_of_shield_distrib,
        "vars": ufs.variences_of_shield_distrib
    }
    pkl.dump(means_vars_10_0, open("means_vars_10_0.pkl", "wb"))
    pkl.dump(kl_div1, open("kls1.pkl", "wb"))

    ufs = UserFeedbackShield(number_of_oracles=9, n_coin_flippers=1)
    # ufs.demo_updates()
    ufs.demo_with_people()
    #TODO:KL
    plt.title("KL_div oracles:{} flippers:{}".format(ufs.number_of_oracles,
                                                     ufs.n_coin_flippers))
    print(ufs.KL_div[1:])
    plt.plot(ufs.KL_div[1:])
    plt.show()

    kl_div2 = {"kl": ufs.KL_div[1:]}
    pkl.dump(kl_div2, open("kls2.pkl", "wb"))
    means_vars_9_1 = {
        "means": ufs.means_of_shield_distrib,
        "vars": ufs.variences_of_shield_distrib
    }
    pkl.dump(means_vars_9_1, open("means_vars_9_1.pkl", "wb"))

    ufs = UserFeedbackShield(number_of_oracles=7, n_coin_flippers=3)
    # ufs.demo_updates()
    ufs.demo_with_people()
    #TODO:KL
    plt.title("KL_div oracles:{} flippers:{}".format(ufs.number_of_oracles,
                                                     ufs.n_coin_flippers))
    print(ufs.KL_div[1:])
    plt.plot(ufs.KL_div[1:])
    plt.show()

    kl_div3 = {"kl": ufs.KL_div[1:]}
    pkl.dump(kl_div3, open("kls3.pkl", "wb"))

    means_vars_7_3 = {
        "means": ufs.means_of_shield_distrib,
        "vars": ufs.variences_of_shield_distrib
    }
    pkl.dump(means_vars_7_3, open("means_vars_7_3.pkl", "wb"))

    # dict_kl = {'kl': kl_div1, "10%": kl_div2, "30%": kl_div3}
    # pkl.dump(dict_kl, open("kls.pkl", "wb"))

    # ufs.demo()