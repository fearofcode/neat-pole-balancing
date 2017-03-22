#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version by Ken Lauer / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

from math import pi
import logging
import os
import pickle
import random

import neat
import visualize
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape)
from framework import (Framework, Keys, main)
import settings


def to_radians(degrees):
    return degrees*pi/180.0


def to_degrees(radians):
    return radians*180.0/pi


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class TrivialProportionalController(object):
    """A simple function that just applies a force proportional to the angle offset of the pole.

    It keeps the pole from falling over for the most part, but the cart will steadily drift off into space."""

    def get_force(self, system):
        return -system.pole.angle * 180 / pi * 5


class CartPendulumSystem(object):
    POSITION_LIMIT = 4  # larger than the benchmarks used in papers due to the dimensions of the objects
    ROTATION_LIMIT = pi / 5  # this is the same, though

    def __init__(self, world, controller, initial_position=None, initial_rotation=None):
        self.world = world
        self.controller = controller

        self.control_enabled = True
        self.print_state = False

        if initial_position is None:
            initial_position = random.uniform(-0.3 * self.POSITION_LIMIT, 0.3 * self.POSITION_LIMIT)
        else:
            initial_position = initial_position

        if initial_rotation is None:
            initial_rotation = random.uniform(0.5*-self.ROTATION_LIMIT, 0.5*self.ROTATION_LIMIT)

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        self.pole = self.world.CreateDynamicBody(
            position=(initial_position, 7),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.5, 2)), density=1.0),
        )

        fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(0.5, 0.5)),
            density=2,
            friction=0.0,
        )

        self.cart = self.world.CreateDynamicBody(
            position=(initial_position, 5),
            fixtures=fixture,
        )

        if initial_rotation < 0:
            motorSpeed = -1
        else:
            motorSpeed = 1

        self.pole_joint = self.world.CreateRevoluteJoint(
            bodyA=self.cart,
            bodyB=self.pole,
            localAnchorA=(0, 0),
            localAnchorB=(0, -2),
            enableMotor=True,
            maxMotorTorque=100,
            motorSpeed=motorSpeed
        )

        self.cart_joint = self.world.CreatePrismaticJoint(
            bodyA=ground,
            bodyB=self.cart,
            anchor=(0, 5),
            axis=(1, 0),
            maxMotorForce=1000,
            enableMotor=True,
        )

        self.first_time = True
        self.initial_rotation = initial_rotation

    @property
    def x(self):
        return self.cart.position.x

    @property
    def dx(self):
        return self.cart.linearVelocity.x

    @property
    def theta(self):
        return self.pole.angle

    @property
    def dtheta(self):
        return self.pole.angularVelocity

    @property
    def system_state(self):
        return [
            self.x,
            self.dx,
            to_degrees(self.theta),
            self.dtheta
        ]

    @property
    def scaled_state(self):
        """Get full state, scaled into (approximately) [0, 1]."""

        return [
            0.5 * (self.x + self.POSITION_LIMIT) / self.POSITION_LIMIT,
            (self.dx + 0.75) / 1.5,
            0.5 * (self.theta + self.ROTATION_LIMIT) / self.ROTATION_LIMIT,
            (self.dtheta + 1.0) / 2.0
        ]

    def in_legal_state(self):
        return -self.POSITION_LIMIT <= self.x <= self.POSITION_LIMIT and \
               -self.ROTATION_LIMIT <= self.theta <= self.ROTATION_LIMIT

    def step(self, step_world=False):
        step = 1.0 / settings.fwSettings.hz
        if self.first_time:
            self.first_time = False

            # shitty hack to use motor on revolute joint to actually rotate the pole
            max_steps = settings.fwSettings.hz
            steps = 0
            while abs(self.pole.angle - self.initial_rotation) > to_radians(0.5) and steps <= max_steps:
                steps += 1
                self.world.Step(step, settings.fwSettings.velocityIterations, settings.fwSettings.positionIterations)

            # now turn it off so gravity will take over

            self.pole_joint.enableMotor = False
            self.pole_joint.motorSpeed = 0
            self.pole_joint.maxMotorTorque = 0
            self.pole_joint.angularVelocity = 0

        if self.print_state:
            logging.info('{}'.format(self.system_state))

        if step_world:
            self.world.Step(step, settings.fwSettings.velocityIterations, settings.fwSettings.positionIterations)

        if self.control_enabled:
            force = self.controller.get_force(self)

            self.cart_joint.motorSpeed = force

    def discrete_loop(self):
        """Step the system for 60 seconds with control possibly applied."""

        time = 60.0
        step = 1.0 / settings.hz
        steps = int(time/step)

        assert steps == 6000

        for i in range(steps):
            self.world.Step(step, settings.velocityIterations, settings.positionIterations)
            self.step()


class CartPendulumDemo(Framework):
    name = "Cart/pendulum system"
    description = "Press 'c' to toggle control. Press 'p' to toggle printing system state to stdout"

    def __init__(self, controller, initial_position, initial_rotation):
        super(CartPendulumDemo, self).__init__()

        self.system = CartPendulumSystem(self.world, controller, initial_position, initial_rotation)
        self.setZoom(25.0)

    def get_starting_resolution(self):
        return 1280, 1024

    def Keyboard(self, key):
        if key == Keys.K_c:
            self.system.control_enabled = not self.system.control_enabled
        elif key == Keys.K_p:
            self.system.print_state = not self.system.print_state

    def Step(self, settings):
        super(CartPendulumDemo, self).Step(settings)

        self.system.step()

        if not self.system.in_legal_state():
            pass
            # settings.pause = True
            # self.Print('Simulation exceeded legal bounds, stopping')


def load_winner_net_controller(net_name='winner-feedforward.save'):
    global config, c
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-cartpole')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open(net_name, 'rb') as f:
        c = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(c, config)
    from evolve_feedforward_box2d import NeuralNetworkController
    return NeuralNetworkController(net)


if __name__ == "__main__":
    initial_position = 2.3
    initial_rotation = 0.0

    run_off_winner_neat = True

    if run_off_winner_neat:

        controller = load_winner_net_controller()

        # print(c)
        # node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
        #visualize.draw_net(config, c, view=True, node_names=node_names,
        #                   filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    else:
        controller = TrivialProportionalController()

    main(CartPendulumDemo, controller, initial_position, initial_rotation)
