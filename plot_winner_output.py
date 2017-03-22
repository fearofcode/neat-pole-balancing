import os
import pickle
import math
from cart_pole import CartPole, discrete_actuator_force
import matplotlib.pyplot as plt

import neat


def plot_net_output(times, positions, rotations):
    plt.plot(times, positions, label="Position")
    plt.plot(times, rotations, label="Rotation")
    plt.title("Position and rotation over time")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # load the winner
    with open('winner-feedforward', 'rb') as f:
        c = pickle.load(f)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-cartpole')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)
    sim = CartPole(x=2.0, dx=0, dtheta=0, theta=-30.0*math.pi/180.0)

    print()
    print("Initial conditions:")
    print("        x = {0:.4f}".format(sim.x))
    print("    x_dot = {0:.4f}".format(sim.dx))
    print("    theta = {0:.4f}".format(sim.theta))
    print("theta_dot = {0:.4f}".format(sim.dtheta))
    print()

    # Run the given simulation for up to 120 seconds.
    balance_time = 0.0
    times = []
    positions = []
    rotations = []

    while sim.t < 30.0:
        inputs = sim.get_scaled_state()
        action = net.activate(inputs)

        # Apply action to the simulated cart-pole
        force = discrete_actuator_force(action)
        sim.step(force)

        # Stop if the network fails to keep the cart within the position or angle limits.
        # The per-run fitness is the number of time steps the network can balance the pole
        # without exceeding these limits.
        if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
            break

        balance_time = sim.t

        times.append(sim.t)
        positions.append(sim.x)
        rotations.append(sim.theta*180.0/math.pi)

    print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))

    print()
    print("Final conditions:")
    print("        x = {0:.4f}".format(sim.x))
    print("    x_dot = {0:.4f}".format(sim.dx))
    print("    theta = {0:.4f}".format(sim.theta))
    print("theta_dot = {0:.4f}".format(sim.dtheta))
    print()
    plot_net_output(times, positions, rotations)


