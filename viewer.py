from enum import StrEnum, auto

from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
from panda3d.core import NodePath
from panda3d.core import TransformState
from panda3d.core import BitMask32, Vec3


class Motions(StrEnum):

    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    TURN = auto()


class Viewer(NodePath):

    def __init__(self):
        super().__init__(BulletRigidBodyNode('aircraft'))
        self.set_collide_mask(BitMask32.bit(2))
        self.node().set_kinematic(True)
        self.node().set_ccd_motion_threshold(1e-7)
        self.node().set_ccd_swept_sphere_radius(0.5)

        self.test_shape = BulletSphereShape(0.5)
        self.angular_velocity = 100
        self.linear_velocity = 10

    def cast_cay(self, from_pos, to_pos):
        if (result := base.world.ray_test_closest(
                from_pos, to_pos, BitMask32.bit(1))).has_hit():
            return result.get_hit_pos()

    def detect_collosion(self, from_pos, to_pos):
        ts_from = TransformState.make_pos(from_pos)
        ts_to = TransformState.make_pos(to_pos)

        if (result := base.world.sweep_test_closest(
                self.test_shape, ts_from, ts_to, BitMask32.bit(1), 0.0)).has_hit():
            return result

    def control(self, direction, dt):
        current_pos = self.get_pos()
        distance = self.linear_velocity * dt
        forward_vec = self.get_quat(base.render).get_forward()

        next_pos = self.get_pos() + forward_vec * direction.y * distance
        next_pos.z += direction.z * distance

        if self.detect_collosion(current_pos, next_pos):
            return

        if below := self.cast_cay(next_pos, next_pos + Vec3(0, 0, -1.5)):
            if (next_z := below.z + 1.0) > next_pos.z:
                next_pos.z = next_z

        if top := self.cast_cay(next_pos, next_pos + Vec3(0, 0, 1.5)):
            if (next_z := top.z - 1.0) < next_pos.z:
                next_pos.z = next_z

        if direction.x:
            self.turn(dt, direction)

        self.set_pos(next_pos)

    def turn(self, dt, direction):
        angle = self.angular_velocity * dt * direction.x
        self.set_h(self.get_h() + angle)

    def get_relative_pos(self, pos):
        return self.get_relative_point(self, pos)