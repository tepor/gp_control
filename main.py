import random
import operator
from functools import partial

import numpy as np

from deap import base, creator, gp, tools, algorithms
import gym


class GPLearner:

    def __init__(self, sim):
        self.sim = sim

        self.pop_count = 300
        self.generations = 10

        # Create the set of primitives that can become nodes in the tree
        self.pset = gp.PrimitiveSet("MAIN", 4)
        # Input variables from the gym
        self.pset.renameArguments(ARG0="CartPosition")
        self.pset.renameArguments(ARG1="CartVelocity")
        self.pset.renameArguments(ARG2="BarAngle")
        self.pset.renameArguments(ARG3="BarVelocity")
        # Functions to apply and their arity
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.neg, 1)
        # Ephemeral functions are called once at the creation of their node
        self.pset.addEphemeralConstant("rand101",
                                       lambda: random.uniform(-1, 1))

        # Encapsulating the creator in a class is a bit strange since it
        # operates in global scope
        creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
        creator.create("Individual",
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMax)

        # Lots of wrapping and partial application of functions to make
        # the lives of the learning algorithms a little easier
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr",
                              gp.genHalfAndHalf,
                              pset=self.pset,
                              min_=1,
                              max_=2)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.eval_one)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate",
                              gp.mutUniform,
                              expr=self.toolbox.expr_mut,
                              pset=self.pset)

        # Set some limits on the size of the trees (up to 17 is normal)
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

        # Some logging and statistics with the built in tools
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        self.hof = tools.HallOfFame(1)

        # Create the initial population and run the algorithm
        self.pop = self.toolbox.population(n=self.pop_count)
        self.pop, self.log = algorithms.eaSimple(self.pop,
                                                 self.toolbox,
                                                 0.5,
                                                 0.1,
                                                 self.generations,
                                                 stats=mstats,
                                                 halloffame=self.hof,
                                                 verbose=True)

    def eval_one(self, individual):
        # Compile the individual to callable function
        ind_func = self.toolbox.compile(expr=individual)

        # Ooh sneaky functional stuff
        # I was inspired by the partial application used in the DEAP docs
        def shim(ind_func, action_space, obs):
            prediction = ind_func(obs[0], obs[1], obs[2], obs[3])
            # Need to figure out how to interact with action spaces to do
            # this on the fly
            if prediction > 0.5:
                return 1
            else:
                return 0

        shimmed = partial(shim, ind_func)
        self.sim.set_planner(shimmed)

        reward = self.sim.run()

        return (reward, )


class Simulation:

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.episodes = 100
        self.max_timesteps = 300
        self.planner = None

    def set_planner(self, planner):
        self.planner = planner

    def run(self):
        if not self.planner:
            print("No planner supplied, using built in")
            self.planner = self.plan

        run_reward = 0

        for i_episode in range(self.episodes):
            obs = self.env.reset()
            episode_reward = 0

            # print(f"Initial state:\n"
            #       f"Cart Pos.: {obs[0]:.3f}, "
            #       f"Cart Vel.: {obs[1]:.3f}, "
            #       f"Pole Angle: {obs[2]:.3f}, "
            #       f"Pole Vel.: {obs[3]:.3f}")

            for step in range(self.max_timesteps):

                # action = self.env.action_space.sample() # Random
                action = self.planner(self.env.action_space, obs)

                obs, reward, done, info = self.env.step(action)

                # print(f"Action: {action}, "
                #       f"Cart Pos.: {obs[0]:.3f}, "
                #       f"Cart Vel.: {obs[1]:.3f}, "
                #       f"Pole Angle: {obs[2]:.3f}, "
                #       f"Pole Vel.: {obs[3]:.3f}")

                episode_reward += reward

                if done:
                    # print(f"Episode complete after {step + 1} timesteps. "
                    #       f"Reward: {episode_reward}")
                    break

            run_reward += episode_reward

        self.env.close()

        # print(
        #     f"Simlation complete. Average reward: {run_reward/self.episodes}")

        return run_reward / self.episodes

    def plan(self, action_space, obs):
        # Magic number determined by GP algorithm once
        magic = obs[1] + obs[2] + obs[3] + 0.6321199184471549

        if magic > 0.5:
            action = 1
        else:
            action = 0

        return action


if __name__ == "__main__":
    sim = Simulation()
    gptree = GPLearner(sim)

    nodes, edges, labels = gp.graph(gptree.hof[0])
    print(labels)
    print(edges)
