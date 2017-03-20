import math

from Box2D import b2World

from cart_pendulum_demo import load_winner_net_controller, CartPendulumSystem
from plot_winner_output import plot_net_output
from settings import fwSettings

if __name__ == "__main__":
    controller = load_winner_net_controller()
    initial_position = 2.3
    initial_rotation = 0.0
    sim = CartPendulumSystem(b2World(), controller, initial_position, initial_rotation)

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

    steps = 3000
    step = 0

    while step < steps:
        sim.step(True)

        times.append(float(step)/fwSettings.hz)
        positions.append(sim.x)
        rotations.append(sim.theta*180.0/math.pi)

        step += 1

    print()
    print("Final conditions:")
    print("        x = {0:.4f}".format(sim.x))
    print("    x_dot = {0:.4f}".format(sim.dx))
    print("    theta = {0:.4f}".format(sim.theta))
    print("theta_dot = {0:.4f}".format(sim.dtheta))
    print()
    plot_net_output(times, positions, rotations)


