"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import pickle

import neat
import visualize
from Box2D import b2World

import cart_pole
import cart_pendulum_demo


runs_per_net = 5
simulation_steps = 6000


class NeuralNetworkController(object):
    def __init__(self, net):
        self.net = net

    def get_force(self, system):
        inputs = system.scaled_state

        action = self.net.activate(inputs)
        return cart_pole.discrete_actuator_force(action)


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    controller = NeuralNetworkController(net)

    fitnesses = []

    eps = 0.00000001

    for runs in range(runs_per_net):
        system = cart_pendulum_demo.CartPendulumSystem(b2World(), controller)
        system.print_state = False

        steps = 0

        f2_fitnesses = []

        while steps < simulation_steps:
            system.step(True)

            if not system.in_legal_state():
                break

            f2_fitnesses.append(1.0/(abs(system.x) + abs(system.dx) + abs(system.theta) + abs(system.dtheta) + eps))
            steps += 1

        # two tier fitness scheme to make sure we at least get to having long-term balancers
        if steps >= simulation_steps:
            fitness = steps + sum(f2_fitnesses)
        else:
            fitness = steps

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    run_parallel = True

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    max_generations = 100

    if run_parallel:
        pe = neat.ParallelEvaluator(8, eval_genome)
        winner = pop.run(pe.evaluate, max_generations)
    else:
        winner = pop.run(eval_genomes, max_generations)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")

if __name__ == '__main__':
    run()
