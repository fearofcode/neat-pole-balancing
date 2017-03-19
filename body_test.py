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

from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2World)

from framework import (Framework, main)


class BodySystem(object):
    def __init__(self, world):
        self.world = world

        # The ground
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 0), (20, 0)])
        )

        # The attachment
        self.attachment = self.world.CreateDynamicBody(
            position=(0, 7),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.5, 2)), density=1.0),
        )

        self.attachment.angle = 10.0*pi/180.0

        # The platform
        fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(4, 0.5)),
            density=1,
            friction=0.0,
        )

        self.platform = self.world.CreateDynamicBody(
            position=(0, 5),
            fixtures=fixture,
        )

        # The joints joining the attachment/platform and ground/platform
        self.world.CreateRevoluteJoint(
            bodyA=self.platform,
            bodyB=self.attachment,
            localAnchorA=(0, 0),
            localAnchorB=(0, -2),
        )

        self.world.CreatePrismaticJoint(
            bodyA=ground,
            bodyB=self.platform,
            anchor=(0, 5),
            axis=(1, 0),
            maxMotorForce=1000,
            enableMotor=True,
            lowerTranslation=-20,
            upperTranslation=20,
            enableLimit=True
        )

    def step(self):
        angle = self.attachment.angle*180/pi*15

        # print(self.attachment.angularVelocity)
        self.platform.ApplyLinearImpulse((-angle, 0), self.platform.worldCenter, True)

    def discrete_loop(self):
        timeStep = 1.0 / 60
        vel_iters, pos_iters = 6, 2

        # This is our little game loop.
        for i in range(6000):
            # Instruct the world to perform a single step of simulation. It is
            # generally best to keep the time step and iterations fixed.
            self.world.Step(timeStep, vel_iters, pos_iters)


class BodyTypes(Framework):
    name = "Body Types"
    description = "foo"
    speed = 3  # platform speed

    def __init__(self):
        super(BodyTypes, self).__init__()

        self.system = BodySystem(self.world)

    def Step(self, settings):
        super(BodyTypes, self).Step(settings)

        self.system.step()

if __name__ == "__main__":
    #main(BodyTypes)
    import datetime

    system = BodySystem(b2World())
    start = datetime.datetime.now()
    system.discrete_loop()
    elapsed = datetime.datetime.now() - start

    print(elapsed.total_seconds() * 1000)
