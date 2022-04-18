# David Hedger 2022

import random
import operator
import math
from functools import partial
import time

# Pathos MP has better support for pickling functions inside classes
# Saves refactoring the GPLearner to its own file and making everything global
from pathos import multiprocessing
import numpy as np
import pygraphviz as pgv

from deap import base, creator, gp, tools, algorithms
import gym


def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1


def mp(fun, obj):
    pool = multiprocessing.Pool()
    return pool.map(fun, obj)


class GPLearner:

    def __init__(self, sim):
        self.sim = sim

        self.pop_count = 300
        self.generations = 40

        self.choose_action_process()
        self.create_individual_definition()
        self.create_primitive_set()
        self.create_toolbox()
        self.create_logger()

    def run(self):
        # Create the initial population and run
        self.pop = self.toolbox.population(n=self.pop_count)
        self.pop, self.log = algorithms.eaSimple(self.pop,
                                                 self.toolbox,
                                                 0.5,
                                                 0.1,
                                                 self.generations,
                                                 stats=self.mstats,
                                                 halloffame=self.hof,
                                                 verbose=True)

    def choose_action_process(self):
        # Figure out how to interact with the action space
        space = self.sim.env.action_space
        if type(space) == gym.spaces.discrete.Discrete:
            actions = space.n
            # It's difficult to know what the discrete actions map to
            # If they're "move left, stop, move right" then we can cast all of
            # -inf to inf and assume the controller doesn't want to move if
            # it's near 0, but if it's in any other order then the controller
            # will have a hard time.
            # Beyond the case of two or three actions it should be hand-tuned
            if actions == 2:
                self.post_process = partial(np.digitize, bins=[0])
            elif actions == 3:
                self.post_process = partial(np.digitize, bins=[-0.5, 0.5])
            else:
                raise NotImplementedError(
                    "Too many discrete actions, please hand-tune")
        elif type(space) == gym.spaces.box.Box:
            actions = space.shape[0]
            if actions > 1:
                # How do we do multiple continuous outputs?
                # Can't do multiple roots in PriimitiveTree
                # Could evolve multiple trees at the same time
                # Cross that bridge when we find a suitable problem
                raise NotImplementedError("Too many continuous actions")
            else:
                # Just pass the continuous value out with limits
                self.post_process = np.clip(a_min=space.low, a_max=space.high)
        else:
            raise NotImplementedError("Unknown action type")

    def create_individual_definition(self):
        # Encapsulating the creator in a class is a bit strange since it
        # operates in global scope
        # Rip them from the global scope and use them locally
        creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
        self.FitnessMax = creator.FitnessMax
        creator.create("Individual",
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMax)
        self.Individual = creator.Individual
        # This is code duplication and will make multiple classes weird
        # I think it's worthwhile for later modularity though
        # A caller instatiating a gptree shouldn't need to know about the other
        # implementations, it should just be able to provide its own
        # weights, shape, etc. without worrying about the right name

    def create_primitive_set(self):
        # Determine how many variables we can use from the environment
        observations = self.sim.env.observation_space.shape[0]

        # Create the set of primitives that can become nodes in the tree
        self.pset = gp.PrimitiveSet("MAIN", observations)
        # Input variables from the gym
        self.pset.renameArguments(ARG0="CosT1")
        self.pset.renameArguments(ARG1="SinT1")
        self.pset.renameArguments(ARG2="CosT2")
        self.pset.renameArguments(ARG3="SinT2")
        self.pset.renameArguments(ARG4="VelT1")
        self.pset.renameArguments(ARG5="VelT2")
        # Functions to apply and their arity
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(operator.neg, 1)
        # self.pset.addPrimitive(math.sin, 1)
        # self.pset.addPrimitive(math.cos, 1)
        # Ephemeral functions are called once at the creation of their node
        self.pset.addEphemeralConstant("rand101",
                                       lambda: random.uniform(-1, 1))

    def create_toolbox(self):
        # Lots of wrapping and partial application of functions to make
        # the lives of the learning algorithms a little easier
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr",
                              gp.genHalfAndHalf,
                              pset=self.pset,
                              min_=1,
                              max_=2)
        self.toolbox.register("individual", tools.initIterate,
                              self.Individual, self.toolbox.expr)
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

        # Add multiprocessing for parallelism
        self.toolbox.register("map", mp)

        # Set some limits on the size of the trees (up to 17 is normal)
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=9))
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=9))

    def create_logger(self):
        # Some logging and statistics with the built in tools
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)
        self.hof = tools.HallOfFame(1)

    def eval_one(self, individual):
        # Compile the individual to callable function
        ind_func = self.toolbox.compile(expr=individual)

        # Ooh sneaky functional stuff
        # I was inspired by the partial application used in the DEAP docs
        def shim(ind_func, action_space, obs):
            prediction = ind_func(obs[0], obs[1], obs[2], obs[3], obs[4],
                                  obs[5])
            # Need to figure out how to interact with action spaces to do
            # this on the fly
            if prediction > 0.5:
                return 1
            if prediction < -0.5:
                return -1
            else:
                return 0

        shimmed = partial(shim, ind_func)
        self.sim.set_planner(shimmed)

        reward = self.sim.run()

        return (reward, )

    @staticmethod
    def visualise(individual):
        nodes, edges, labels = gp.graph(individual)
        print(labels)
        print(edges)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for node in nodes:
            n = g.get_node(node)
            n.attr["label"] = labels[node]

        g.draw("tree.pdf")


class Simulation:

    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        self.episodes = 100
        self.max_timesteps = 300
        self.planner = None
        self.wall_start = time.time()

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

        # wall_stop = time.time()
        # elapsed = wall_stop - self.wall_start
        # print(
        #     f"Simlation complete. Average reward: {run_reward/self.episodes}. Elapsed: {elapsed:.1f}s")

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
    gptree.run()
    gptree.visualise(gptree.hof[0])