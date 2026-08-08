import sys
import types
from enum import Enum, auto

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.showbase.InputStateGlobal import inputState
from direct.interval.IntervalGlobal import Sequence, Func
from panda3d.bullet import BulletWorld, BulletDebugNode
from panda3d.core import Point3, Vec3, Vec2
from panda3d.core import NodePath
from panda3d.core import AntialiasAttrib, TransparencyAttrib
from panda3d.core import load_prc_file_data

from scene import Scene
from viewer import Viewer, Motions


load_prc_file_data("", """
    textures-power-2 none
    gl-coordinate-system default
    window-title Panda3D Voronoi City
    filled-wireframe-apply-shader true
    stm-max-views 8
    stm-max-chunk-count 2048
    framebuffer-multisample 1
    multisamples 2""")


class Status(Enum):

    SCREEN_CHANGE = auto()
    WAITING = auto()
    ACTIVE = auto()


class View(Enum):

    SKY = auto()
    GROUND = auto()


class VoronoiCity2(ShowBase):

    def __init__(self):
        super().__init__()
        self.disable_mouse()
        self.render.set_antialias(AntialiasAttrib.MAuto)

        self.world = BulletWorld()
        self.world.set_gravity(Vec3(0, 0, -9.81))
        self.debug = self.render.attach_new_node(BulletDebugNode('debug'))
        self.world.set_debug_node(self.debug.node())

        self.scene = Scene()
        self.camera_root = NodePath('camera_root')
        self.camera_root.reparent_to(self.render)

        self.sky_config = types.SimpleNamespace(
            parent=self.camera_root,
            pos=Point3(-300, -300, 150),
            look_at=Point3(100, 100, 0),
            fov=40,
            near=1,
            switch_func=self.view_from_sky
        )

        self.ground_config = types.SimpleNamespace(
            parent=self.render,
            pos=Point3(-140, -140, 2),
            look_at=Point3(0, 0, 0),
            fov=90,
            near=0.1,
            switch_func=self.view_while_moving
        )

        self.viewer = Viewer()
        self.viewer.reparent_to(self.render)
        self.world.attach(self.viewer.node())
        self.view_from_sky()

        self.clicked = False
        self.dragging = False
        self.before_mouse_pos = None
        self.screen_changed = False
        self.status = Status.ACTIVE
        self.view = View.SKY

        self.accept('escape', sys.exit)
        self.accept('mouse1', self.mouse_click)
        self.accept('mouse1-up', self.mouse_release)
        self.accept('t', self.toggle_debug)
        self.accept('v', self.toggle_view)
        self.accept('w', self.toggle_wireframe)

        # viewer control
        inputState.watch_with_modifiers(Motions.FORWARD, 'arrow_up')
        inputState.watch_with_modifiers(Motions.BACKWARD, 'arrow_down')
        inputState.watch_with_modifiers(Motions.LEFT, 'arrow_left')
        inputState.watch_with_modifiers(Motions.RIGHT, 'arrow_right')
        inputState.watch_with_modifiers(Motions.DOWN, 'd')
        inputState.watch_with_modifiers(Motions.UP, 'u')

        self.taskMgr.add(self.update, 'update')

    def toggle_view(self):
        self.status = Status.SCREEN_CHANGE

    def fade_camera(self, duration=2.0):
        config = self.ground_config if self.view == View.SKY else self.sky_config
        self.screen_changed = False

        props = self.win.get_properties()
        size = props.get_size()
        buffer = self.win.make_texture_buffer('tex_buffer', *size)
        buffer.set_clear_color_active(True)
        buffer.set_clear_color((0.5, 0.5, 0.5, 1.0))

        temp_cam = self.make_camera(buffer)
        temp_cam.node().get_lens().set_fov(config.fov)
        temp_cam.set_pos(config.pos)
        temp_cam.look_at(config.look_at)
        temp_cam.reparent_to(config.parent)

        card = buffer.get_texture_card()
        card.reparent_to(self.render2d)
        card.set_transparency(TransparencyAttrib.M_alpha)
        # card.set_transparency(TransparencyAttrib.M_multisample)

        Sequence(
            card.colorScaleInterval(duration, 1, 0, blendType='easeInOut'),
            Func(config.switch_func),
            Func(card.remove_node),
            Func(self.graphicsEngine.remove_window, buffer),
            Func(self.end_fade)
        ).start()

    def view_while_moving(self):
        """Use the keyboard to move around the city.
        """
        self.viewer.set_pos(self.ground_config.pos)
        self.viewer.look_at(self.ground_config.look_at)

        self.camera.set_pos_hpr(Point3(0, 0, 0), Vec3(0, 0, 0))
        self.camera.reparent_to(self.viewer)

        self.camLens.set_fov(self.ground_config.fov)
        self.camLens.set_near_far(self.ground_config.near, 100000)

    def view_from_sky(self):
        """View the city from above and rotate it by dragging the mouse.
        """
        self.camera.reparent_to(self.camera_root)
        self.camera.set_pos(self.sky_config.pos)
        self.camera.look_at(self.sky_config.look_at)

        self.camLens.set_fov(self.sky_config.fov)
        self.camLens.set_near_far(self.sky_config.near, 100000)

    def end_fade(self):
        self.screen_changed = True

    def toggle_debug(self):
        if self.debug.is_hidden():
            self.debug.show()
        else:
            self.debug.hide()

    def mouse_click(self):
        self.dragging = True
        self.dragging_start_time = globalClock.get_frame_time()

    def mouse_release(self):
        if globalClock.get_frame_time() - self.dragging_start_time < 0.2:
            self.clicked = True

        self.dragging = False
        self.before_mouse_pos = None

    def rotate_camera(self, mouse_pos, dt):
        if self.before_mouse_pos:
            angle = Vec3()

            if (delta := mouse_pos.x - self.before_mouse_pos.x) < 0:
                angle.x += 180
            elif delta > 0:
                angle.x -= 180

            if (delta := mouse_pos.y - self.before_mouse_pos.y) < 0:
                angle.z -= 180
            elif delta > 0:
                angle.z += 180

            angle *= dt
            self.camera_root.set_hpr(self.camera_root.get_hpr() + angle)

        self.before_mouse_pos = Vec2(mouse_pos.xy)

    def watch_keyboard(self):
        direction = Vec3()

        if inputState.is_set(Motions.FORWARD):
            direction.y += 1

        if inputState.is_set(Motions.BACKWARD):
            direction.y -= 1

        if inputState.is_set(Motions.LEFT):
            direction.x += 1

        if inputState.is_set(Motions.RIGHT):
            direction.x -= 1

        if inputState.is_set(Motions.UP):
            direction.z += 1

        if inputState.is_set(Motions.DOWN):
            direction.z -= 1

        return direction

    def update(self, task):
        dt = globalClock.get_dt()

        match self.status:

            case Status.SCREEN_CHANGE:
                self.fade_camera()
                self.status = Status.WAITING

            case Status.WAITING:
                if self.screen_changed:
                    self.view = View.GROUND if self.view == View.SKY else View.SKY
                    self.status = Status.ACTIVE

            case Status.ACTIVE:
                if self.view == View.GROUND:
                    direction = self.watch_keyboard()
                    self.viewer.control(direction, dt)

                if self.mouseWatcherNode.has_mouse():
                    mouse_pos = self.mouseWatcherNode.get_mouse()

                    if self.view == View.SKY and self.dragging:
                        if globalClock.get_frame_time() - self.dragging_start_time >= 0.2:
                            self.rotate_camera(mouse_pos, dt)

        self.world.do_physics(dt)
        return task.cont


if __name__ == '__main__':
    app = VoronoiCity2()
    app.run()