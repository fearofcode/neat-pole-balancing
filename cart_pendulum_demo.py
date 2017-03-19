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

from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape)

from framework import (Framework, Keys, main)
import settings


class CartPendulumSystem(object):
    def __init__(self, world):
        self.world = world
        self.control_enabled = True

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        self.pole = self.world.CreateDynamicBody(
            position=(0, 7),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.5, 2)), density=1.0),
        )

        self.pole.angle = 10.0 * pi / 180.0

        fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(4, 0.5)),
            density=1,
            friction=0.0,
        )

        self.cart = self.world.CreateDynamicBody(
            position=(0, 5),
            fixtures=fixture,
        )

        self.world.CreateRevoluteJoint(
            bodyA=self.cart,
            bodyB=self.pole,
            localAnchorA=(0, 0),
            localAnchorB=(0, -2),
        )

        self.world.CreatePrismaticJoint(
            bodyA=ground,
            bodyB=self.cart,
            anchor=(0, 5),
            axis=(1, 0),
            maxMotorForce=1000,
            enableMotor=True,
            lowerTranslation=-20,
            upperTranslation=20,
            enableLimit=True
        )

    def step(self):
        print(self.pole.angle)
        if self.control_enabled:
            angle = self.pole.angle * 180 / pi * 15

            self.cart.ApplyLinearImpulse((-angle, 0), self.cart.worldCenter, True)

    def discrete_loop(self):
        timeStep = 1.0 / settings.hz

        for i in range(6000):
            self.world.Step(timeStep, settings.velocityIterations, settings.positionIterations)
            self.step()


class CartPendulumDemo(Framework):
    name = "Cart/pendulum system"
    description = "Press 'c' to toggle control"

    def __init__(self):
        super(CartPendulumDemo, self).__init__()

        self.system = CartPendulumSystem(self.world)
        self.setZoom(25.0)

    def get_starting_resolution(self):
        return 1280, 1024

    def Keyboard(self, key):
        if key == Keys.K_c:
            self.system.control_enabled = not self.system.control_enabled
    def Step(self, settings):
        super(CartPendulumDemo, self).Step(settings)

        self.system.step()

if __name__ == "__main__":
    main(CartPendulumDemo)