
import gym

import logging
import os
import numpy as np
import neat
import pickle

episodes = 5

env_name = 'Pendulum-v0'
config_name = 'config-feedforward-pendulum'

# TODO factor out copy and pasted shit

def eval_genome(genome, config):
    # TODO can we actually get away with only one env? it seems like they'd all step on each other
    env = gym.make(env_name)
    scale = -env.observation_space.high
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(episodes):
        fitness = 0.0

        observation = env.reset()

        # run until gym says to stop
        while True:
            scaled_input = observation / scale

            output = net.activate(scaled_input)
            action = np.argmax(output)
            observation, reward, done, info = env.step(action)
            fitness += reward

            if done:
                break

        fitnesses.append(fitness)

    return sum(fitnesses) / len(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5))

    pe = neat.ParallelEvaluator(8, eval_genome)
    winner = pop.run(pe.evaluate, 200)

    # winner = pop.run(eval_genomes, 300)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    visualize_winner(net)


def visualize_winner(net):
    fitness = 0.0
    env = gym.make(env_name)
    scale = env.observation_space.high
    observation = env.reset()
    while True:
        scaled_input = observation / scale
        env.render()
        output = net.activate(scaled_input)
        action = np.argmax(output)

        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        fitness += reward

        if done:
            break
    print("Fitness", fitness)


def load_winner(config_name):
    with open('winner-feedforward', 'rb') as f:
        c = pickle.load(f)
    print('Loaded genome:')
    print(c)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    return neat.nn.FeedForwardNetwork.create(c, config)


if __name__ == '__main__':
    # quiet
    logger = logging.getLogger('gym.envs.registration')
    logger.setLevel(logging.ERROR)

    train = True

    if train:
        run()
    else:
        net = load_winner(config_name)

        visualize_winner(net)
