
import gym


import os
import numpy as np
import neat
import visualize

episodes = 5
steps = 5000

# env = gym.make('CartPole-v0')
env = gym.make('CartPole-v1')

DO_RENDER = False

POSITION_MAX = env.observation_space.high[0]
ROTATION_MAX = env.observation_space.high[2]
VELOCITY_MAX = 25  # arbitrary guess


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(episodes):
        fitness = 0.0

        observation = env.reset()
        for t in range(steps):
            scaled_input = [observation[0] / POSITION_MAX,
                            observation[1] / VELOCITY_MAX,
                            observation[2] / ROTATION_MAX,
                            observation[3] / VELOCITY_MAX]
            if DO_RENDER:
                env.render()
            output = net.activate(scaled_input)
            action = np.argmax(output)

            observation, reward, done, info = env.step(action)
            fitness += reward

            if done:
                break
        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
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

    # pe = neat.ParallelEvaluator(8, eval_genome)
    # winner = pop.run(pe.evaluate, 200)

    winner = pop.run(eval_genomes, 300)

    print(winner)

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    fitness = 0.0

    observation = env.reset()

    for t in range(steps):
        scaled_input = [observation[0] / POSITION_MAX,
                        observation[1] / VELOCITY_MAX,
                        observation[2] / ROTATION_MAX,
                        observation[3] / VELOCITY_MAX]
        env.render()
        output = net.activate(scaled_input)
        action = np.argmax(output)

        observation, reward, done, info = env.step(action)
        fitness += reward

        if done:
            break

    print("Fitness", fitness)


if __name__ == '__main__':
    run()