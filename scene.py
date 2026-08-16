import math
import random
from enum import Enum, auto

import numpy as np
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.bullet import BulletConvexHullShape, BulletCylinderShape, ZUp
from panda3d.core import NodePath, PandaNode
from panda3d.core import Point3, Vec3, BitMask32, LColor
from panda3d.core import TextureStage, TransformState, TexGenAttrib
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import OmniBoundingVolume, Quat
from panda3d.core import Shader, ShaderBuffer, GeomEnums
from panda3d.core import TransparencyAttrib

from noise import Fractal2D, Fractal3D, PerlinNoise
from shapes import RandomPolygonalPrism
from shapes import Plane, Sphere
from voronoi_generator.voronoi_2d import BoundedVoronoiGenerator, VoronoiSitesGenerator, RoundedVoronoiGenerator
from voronoi_generator.voronoi_2d import Polygon2DMixin
from voronoi_generator.polygon_mixin import PolygonMixin


class BuildingRoot(PolygonMixin, NodePath):

    def __init__(self, name):
        super().__init__(BulletRigidBodyNode(name))
        self.set_collide_mask(BitMask32.bit(1))
        self.node().set_mass(0)

    def assemble(self, model, pos, name, is_convex=True):
        if is_convex:
            shape = BulletConvexHullShape()
            shape.add_geom(model.node().get_geom(0))
        else:
            mesh = BulletTriangleMesh()
            mesh.add_geom(model.node().get_geom(0))
            shape = BulletTriangleMeshShape(mesh, dynamic=False)

        self.node().add_shape(shape, TransformState.make_pos(pos))
        model.set_pos(pos)
        model.set_name(name)
        model.reparent_to(self)

    def get_vdata_memview(self, name):
        """Retrieve a child node based on its name, convert the vertex data
            to a memoryview, and return it.
            Args:
                name (str): child name node
        """
        node_path = self.find(name)
        geom_node = node_path.node()
        geom = geom_node.modify_geom(0)
        vdata = geom.modify_vertex_data()
        vdata_arr = vdata.modify_array(0)
        vdata_mem = memoryview(vdata_arr).cast('B').cast('f')
        return vdata_mem


class Building(BuildingRoot):

    wall_heights = [10, 20, 30, 40, 50, 60, 70, 80]

    def __init__(self, serial, foundation_h, wall_h, roof_h):
        super().__init__(f'building_{serial}')
        self.foundation_h = foundation_h
        self.wall_h = wall_h
        self.roof_h = roof_h

    def create_foundation(self, model_creator, foundation_tex):
        """Create building foundation.
        """
        # create_model.
        model_creator.height = self.foundation_h
        model = model_creator.create()

        # set texture.
        su = self.round_off(model_creator.edge_length / 20)
        model.set_tex_scale(TextureStage.get_default(), (su, 0.5))
        model.set_texture(foundation_tex)

        # parent to self.
        self.assemble(model, Point3(0, 0, 0), name="foundation")

    def create_wall(self, model_creator, wall_tex):
        """Create building wall.
        """
        # create model.
        model_creator.height = self.wall_h
        model_creator.segs_a = int(self.wall_h / 2)
        model = model_creator.create()

        # set texture.
        u = model_creator.edge_length / 50
        su = np.ceil(u * 3) / 3
        v = self.wall_h / 50
        sv = np.ceil(v * 4) / 4
        model.set_tex_scale(TextureStage.get_default(), (su, sv))
        model.set_texture(wall_tex)

        # parent to self.
        pos = Point3(0, 0, self.foundation_h)
        self.assemble(model, pos, "wall")

    def create_roof(self, model_creator, roof_tex):
        """Create building foundation.
        """
        # create model.
        model_creator.height = self.roof_h
        model_creator.segs_a = 3
        model = model_creator.create()

        # set texture.
        su = self.round_off(model_creator.edge_length / 10)
        model.set_tex_scale(TextureStage.get_default(), (su, 0.2))
        model.set_texture(roof_tex)

        # parent to self.
        pos = Point3(0, 0, self.wall_h + self.foundation_h)
        self.assemble(model, pos, "roof")


class Land(BuildingRoot):

    def __init__(self, serial, land_h=4.0):
        super().__init__(f'land_{serial}')
        self.land_h = land_h

    def create_land(self, model_creator, tex):
        model_creator.height = self.land_h
        model = model_creator.create()
        model.set_texture(tex)

        self.assemble(model, Point3(0, 0, 0), "roof")


class Garden(BuildingRoot):

    def __init__(self, serial, height):
        super().__init__(f'garden_{serial}')
        self.height = height

    def create_flowerbed(self, model_creator, tex):
        """Create garden.
        """
        model_creator.height = self.height
        flowerbed = model_creator.create()
        flowerbed.set_texture(tex)
        self.assemble(flowerbed, Point3(0, 0, 0), 'flowerbed')

    def create_fence(self, model_creator, tex):
        """Create the edge of the garden.
        """
        model_creator.height = self.height + 0.5
        model = model_creator.create()

        # set texture.
        su = self.round_off(model_creator.edge_length / 20)
        model.set_tex_scale(TextureStage.get_default(), (su, 0.5))
        model.set_texture(tex)

        self.assemble(model, Point3(0, 0, 0), 'fence', is_convex=False)

    def plant_palmtree(self, pos):
        tree = base.loader.load_model('models/palmtree/tree3.bam')
        hpr = Vec3(random.uniform(0, 360), 0, 0)
        tree.set_pos_hpr_scale(pos, hpr, Vec3(5))

        # Unless I executed set_shader_off, I couldn't change the color scale.
        # tree.clear_material()
        tree.set_shader_off()
        tree.set_color_scale(LColor(1.3, 1.4, 1.0, 1.0), 1)

        # The 3D model of the palmtree is designed so that the center and the root section are
        # offset from each other. Place a CollisionShape at the coordinates of the root section.
        q = Quat()
        q.setFromAxisAngle(hpr.x, Vec3.up())
        shape_rel_pos = q.xform(Vec3(3, 0, 0))

        end, tip = tree.get_tight_bounds()
        size = tip - end
        height_offset = Vec3(0, 0, size.z / 2.0)

        shape = BulletCylinderShape(2.0, size.z, ZUp)
        self.node().add_shape(shape, TransformState.make_pos(shape_rel_pos + pos + height_offset))
        tree.reparent_to(self)


class VegetationMixin:

    def create_model(self, matrices, file_path, is_flower=True, color_scale=None):
        arr_matrices = np.array(matrices, dtype=np.float32)
        raw_buffer_data = arr_matrices.tobytes()
        model = base.loader.load_model(f'models/{file_path}')

        if color_scale is not None:
            # Set the color to white first, since the color of plant1 didn't change with just set_color_scale.
            model.set_color(1, 1, 1, 1)
            model.set_color_scale(*color_scale, 1)

        model.flatten_light()
        model.reparent_to(self)
        model.set_pos(0, 0, 0)
        model.set_hpr(0, 0, 0)

        # Prevent curling.
        for geom_np in model.find_all_matches("**/+GeomNode"):
            geom_nd = geom_np.node()
            geom_nd.set_bounds(OmniBoundingVolume())
            geom_nd.set_final(True)

        # model.node().set_bounds(OmniBoundingVolume())
        # model.node().set_final(True)

        # Set the number of instances.
        instance_cnt = len(matrices)
        model.set_instance_count(instance_cnt)
        # Set shader.
        v_shader = 'instancing_flower_v.glsl' if is_flower else 'instancing_v.glsl'
        model.set_shader(Shader.load(Shader.SL_GLSL, vertex=f'shaders/{v_shader}', fragment='shaders/instancing_f.glsl'))
        model.set_shader_input("instanced_object", ShaderBuffer('DataBuffer', raw_buffer_data, GeomEnums.UH_static))

    def transform_plant(self, matrices_list, dummy_np, pos, normal, scale):
        """Using a dummy NodePath, calculate the plant's translation, rotation,
           and scaling, and store the results in a temporary list.
        """
        dummy_np.set_pos(pos)
        dummy_np.set_scale(scale)

        if normal.length_squared() > 0.001:
            # If the plant is on a wall, rotate it so that it appears to be growing outward from the wall.
            dummy_np.look_at(pos + normal, Vec3(0, 0, 1))
            dummy_np.set_p(dummy_np, -90)
            dummy_np.set_h(dummy_np, random.uniform(0, 360))
        else:
            # If the plants are located on the rooftop, place it upright.
            dummy_np.set_h(random.uniform(0, 360))

        mat = dummy_np.get_mat()
        # Convert to a flat list.
        mat_data = [mat.get_cell(r, c) for r in range(4) for c in range(4)]
        matrices_list.append(mat_data)

    def determine_density(self, noise, lower_bound, upper_bound):
        if noise < lower_bound:
            return 0.0

        if noise > upper_bound:
            return 1.0

        # smoothstep
        t = (noise - lower_bound) / (upper_bound - lower_bound)
        density = t * t * (3.0 - 2.0 * t)
        return density


class WallGreening(VegetationMixin, NodePath):

    def __init__(self):
        super().__init__(PandaNode("vegetation"))
        self.create_noise()

        self.mat_plants1 = []
        self.mat_plants2 = []
        self.mat_shrubbery = []
        self.mat_fern = []

    def create_noise(self):
        perlin = PerlinNoise()

        # Noise for planting plants in narrow rows.
        self.noise_a = Fractal3D(
            perlin.pnoise3,
            gain=0.5,
            lacunarity=2.01,
            octaves=4,
            amplitude=1.0,
            frequency=1.2
        )
        # Noise for rough and fine
        self.noise_b = Fractal3D(
            perlin.pnoise3,
            gain=0.4,
            octaves=2,
            amplitude=1.0,
            frequency=0.09
        )
        # Noise that determines how far down from the roof to plant the plants.
        self.noise_c = Fractal3D(
            perlin.pnoise3,
            gain=0.3,
            octaves=1,
            amplitude=1.0,
            frequency=0.08
        )

    def get_dentisy(self, x, y, z, max_height=10.0, bias=0.0):
        """Calculating the density of plants covering a wall based on noise.
        """
        raw_a = self.noise_a.fractal(x, y, z)
        raw_b = self.noise_b.fractal(x, y, z)
        raw_c = self.noise_c.fractal(x, y, 0.0)

        # Just to be safe, clamp the range to 0.0–1.0
        noise_a = max(0.0, min(raw_a, 1.0))
        noise_b = max(0.0, min(raw_b, 1.0))
        noise_c = max(0.0, min(raw_c, 1.0))

        # Multiply the two noise patterns to create an area on the wall without plants.
        combined_noise = noise_a * noise_b
        combined_noise = math.sqrt(combined_noise)

        # Choose from four levels to determine how far down the wall from the roof the plants will cover.
        # The larger the value of hang_exponent, the larger the portion of the wall that is bare.
        if noise_c < 0.25:
            hang_exponent = max(0.1, 6.0 + bias)
        elif noise_c < 0.50:
            hang_exponent = max(0.1, 3.5 + bias)
        elif noise_c < 0.75:
            hang_exponent = max(0.1, 1.8 + bias)
        else:
            hang_exponent = max(0.1, 0.8 + bias)

        # Calculation of the Height Coefficient（0.0 〜 1.0）
        normalized_z = max(0.0, min(z / max_height, 1.0))
        height_factor = normalized_z ** hang_exponent

        # The smaller the height_factor, the higher the lower_bound becomes, and the fewer plants there are.
        lower_bound = 0.55 - (height_factor * 0.35)
        upper_bound = 0.75 - (height_factor * 0.25)

        density = self.determine_density(combined_noise, lower_bound, upper_bound)
        return density

    def distribute(self, building):
        """Using noise to plant vegetation on the walls and rooftops of buildings
        """
        vdata_mem = building.get_vdata_memview('wall')
        dummy = NodePath(PandaNode("dummy_transform"))
        wx, wy, wz = building.get_pos(base.render)
        bias = random.uniform(-1.2, 4.0)

        # Store the array data in a temporary list by plant type.
        for i in range(0, len(vdata_mem), 12):
            x, y, z = vdata_mem[i: i + 3]

            # Avoid planting plants on the base of the 3d model.
            if z <= 0:
                continue

            # Change the plants based on the noise level.
            pos = Point3(x + wx, y + wy, z + wz + building.foundation_h)
            normal = Vec3(*vdata_mem[i + 7: i + 10])

            if (dentisy := self.get_dentisy(x, y, z, max_height=building.wall_h, bias=bias)) >= 1.0:
                scale = Vec3(0.09)
                self.transform_plant(self.mat_plants2, dummy, pos, normal, scale)
            elif dentisy >= 0.8:
                scale = Vec3(0.09)
                self.transform_plant(self.mat_plants1, dummy, pos, normal, scale)
            elif dentisy >= 0.35:
                scale = Vec3(0.08)
                self.transform_plant(self.mat_shrubbery, dummy, pos, normal, scale)
            elif dentisy >= 0.25:
                scale = Vec3(0.01)
                self.transform_plant(self.mat_fern, dummy, pos, normal, scale)
            else:
                continue

    def planting(self):
        # plants1
        if len(self.mat_plants1) > 0:
            self.create_model(self.mat_plants1, 'plants1/plants1.egg', False)

        # plants1 which color_scale is changed
        if len(self.mat_plants2) > 0:
            color_scale = LColor(1.1, 1.4, 1.1, 1.0)
            self.create_model(self.mat_plants2, 'plants1/plants1.egg', False, color_scale)

        # shrubbery
        if len(self.mat_shrubbery) > 0:
            self.create_model(self.mat_shrubbery, 'shrubbery/shrubbery.egg', False)

        # fern
        if len(self.mat_fern) > 0:
            color_scale = LColor(0.2, 0.6, 0.2, 1.0)
            self.create_model(self.mat_fern, 'fern/Fern.egg', False, color_scale)


class Gardening(VegetationMixin, NodePath):

    def __init__(self):
        super().__init__(PandaNode("gardening"))
        self.create_noise()

        self.mat_plants1 = []
        self.mat_fern = []
        self.mat_tulip = []
        self.mat_sunflower = []
        self.mat_daisy = []

    def create_noise(self):
        perlin = PerlinNoise()

        self.noise_a = Fractal2D(
            perlin.pnoise2,
            gain=0.5,
            lacunarity=2.01,
            octaves=4,
            amplitude=1.0,
            frequency=1.8
        )
        self.noise_b = Fractal2D(
            perlin.pnoise2,
            gain=0.4,
            octaves=2,
            amplitude=1.0,
            frequency=0.25
        )

    def get_density(self, x, y):
        # Combining two types of noise.
        raw_a = self.noise_a.fractal(x, y)
        raw_b = self.noise_b.fractal(x, y)
        noise_a = max(0.0, min(raw_a, 1.0))
        noise_b = max(0.0, min(raw_b, 1.0))
        combined_noise = math.sqrt(noise_a * noise_b)

        # Thresholding and determining density
        lower_bound = 0.15
        upper_bound = 0.75
        density = self.determine_density(combined_noise, lower_bound, upper_bound)
        return density

    def distribute(self, garden, flower_type):
        # Retrieve the vertex coordinates of only the top face of a Voronoi cell's column.
        vdata_mem = garden.get_vdata_memview('flowerbed')
        arr = np.asarray(vdata_mem)
        verts = arr.reshape(-1, 12)[:, :3]
        verts = verts[verts[:, 2] >= garden.height]
        rounded_verts = np.round(verts, decimals=3)
        unique_verts = np.unique(rounded_verts, axis=0)

        # Ensure that the acquired vertex coordinates are scattered irregularly.
        jitter = np.random.uniform(-10, 10, size=unique_verts.shape)
        jitter[:, 2] = 0.0
        unique_verts += jitter

        # Just as plants are within a garden, move the vertex coordinates toward the inside of the garden.
        shrink_factor = np.array([0.75, 0.75, 1])
        unique_verts *= shrink_factor

        tree_z_offsets = np.linspace(0, garden.height, 2 + 1)
        wx, wy, _ = garden.get_pos()
        dummy = NodePath(PandaNode("dummy_transform"))

        for x, y, z in unique_verts:
            # Apply a CollisionShape only to the trunk of the palm tree.
            if (density := self.get_density(x, y)) >= 0.85:
                offset = random.choice(tree_z_offsets)
                pos = Point3(x, y, z - offset)
                garden.plant_palmtree(pos)
                continue

            pos = Point3(x + wx, y + wy, z)

            if density >= 0.7:
                self.transform_plant(self.mat_plants1, dummy, pos, Vec3.up(), Vec3(0.3))
            elif density >= 0.6:
                self.transform_plant(self.mat_fern, dummy, pos, Vec3.up(), Vec3(0.05))
            elif density >= 0.4:
                match flower_type:
                    case Flowers.SUNFLOWER:
                        self.transform_plant(self.mat_sunflower, dummy, pos, Vec3.up(), Vec3(6))
                    case Flowers.TULIP:
                        # self.transform_plant(self.mat_shrubbery2, dummy, pos, Vec3.up(), Vec3(0.02))
                        self.transform_plant(self.mat_tulip, dummy, pos, Vec3.up(), Vec3(15))
                    case Flowers.DAYSY:
                        self.transform_plant(self.mat_daisy, dummy, pos, Vec3.up(), Vec3(15))

    def planting(self):
        p_shader = 'instancing_v.glsl'
        f_shader = 'instancing_flower_v.glsl'

        # plants1
        if len(self.mat_plants1) > 0:
            color_scale = LColor(1.2, 1.8, 1.4, 1.0)
            self.create_model(self.mat_plants1, 'plants1/plants1.egg', False, color_scale)

        # fern
        if len(self.mat_fern) > 0:
            color_scale = LColor(0.45, 0.65, 0.15, 1.0)
            self.create_model(self.mat_fern, 'fern/Fern.egg', False, color_scale)

        # sunflower
        if len(self.mat_sunflower) > 0:
            color_scale = LColor(1.1, 0.85, 0.2, 1.0)
            self.create_model(self.mat_sunflower, 'Sunflower/Sunflower.egg', True, color_scale)

        # shrubbery
        if len(self.mat_tulip) > 0:
            color_scale = LColor(0.9, 0.22, 0.32, 1.0)
            self.create_model(self.mat_tulip, 'Tulip/Tulip.egg', True, color_scale)

        # daisy
        if len(self.mat_daisy) > 0:
            self.create_model(self.mat_daisy, 'daisy/daisy.egg', True)


class TownBuilder(Polygon2DMixin):

    def __init__(self, scale=256):
        self.scale = scale
        self.create_textures()

    def create_textures(self):
        self.foundation_tex = base.loader.load_texture('textures/foundation2.png')
        self.wall_tex = base.loader.load_texture('textures/gray_brick.png')
        self.roof_tex = base.loader.load_texture('textures/dark_gray_concrete.jpg')
        self.land_tex = base.loader.load_texture('textures/concrete_01.jpg')
        self.grass_tex = base.loader.load_texture('textures/grass_04.jpg')
        self.ground_tex = base.loader.load_texture('textures/board_01.jpg')
        self.tree_model = base.loader.load_model('models/plants3/plants3.egg')

    def build(self):
        segs = 0.0029

        # Generate the vertex coordinates of a Voronoi cell clipped to a 1x1 square.
        for i, region in enumerate(BoundedVoronoiGenerator(cnt_points=6, buffer_size_erosion=-0.06)):
            # The ground on which buildings and parks are built.
            rounded_poly = self.round_polygon_corners(
                region, buffer_size_dilation=0.05, segment_length=0.01)
            land_pts = np.insert(rounded_poly, rounded_poly.shape[1], 0, axis=1)
            land = self.create_land(land_pts, i)
            yield land

            if i % 2 == 0:
                garden = self.create_garden(land_pts, i)
                yield garden
                continue

            # Generate voronoi sites to further subdivide the Voronoi cell.
            sites = np.array([pt for pt in VoronoiSitesGenerator(region)])

            # Generate vertex coordinates of a rounded-corner Voronoi cell.
            for j, pts in enumerate(RoundedVoronoiGenerator(pts=sites, bnd=region, segment_length=segs)):
                if len(pts) == 0:
                    continue

                polygon = np.insert(pts, pts.shape[1], 0, axis=1)
                sorted_pts = self.sort_counter_clockwise(polygon)
                yield self.create_building(sorted_pts, f'{i}_{j}')

    def get_max_distance_from_center(self, verts):
        """Calculate the center point from the vertex coordinates that form a convex polygon,
           and retrieve the coordinates of the vertex farthest from the center point
            Args:
                verts (Numpy.ndarray): vertex coordinates that form a convex polygon
        """
        center = np.mean(verts, axis=0)
        distances = np.sum((verts - center) ** 2, axis=1)
        max_distance = np.max(distances) ** 0.5
        return center, max_distance

    def create_garden(self, sorted_pts, serial):
        garden = Garden(serial, height=2.0)
        scaled_pts = sorted_pts * self.scale

        # Determine the value of segs_top_cap based on the vertex farthest from the center.
        center, max_distance = self.get_max_distance_from_center(scaled_pts)
        segs_top_cap = 3 if max_distance <= 2 else int(max_distance / 2)

        # Make the area where plants will be planted slightly smaller than the base.
        model_creator = RandomPolygonalPrism(
            list(scaled_pts * 0.8), segs_bottom_cap=segs_top_cap)
        garden.create_flowerbed(model_creator, self.ground_tex)

        # Create garden fence
        model_creator = RandomPolygonalPrism(
            list(scaled_pts * 0.805), segs_top_cap=1, segs_bottom_cap=1, thickness=0.5)
        garden.create_fence(model_creator, self.foundation_tex)

        pos = Point3(*center) - Vec3(self.scale / 2, self.scale / 2, 0)
        garden.set_pos(pos)
        return garden

    def create_land(self, sorted_pts, serial):
        land = Land(serial, land_h=6)
        scaled_pts = sorted_pts * self.scale
        model_creator = RandomPolygonalPrism(list(scaled_pts))
        land.create_land(model_creator, self.land_tex)

        pos = Point3(*model_creator.center) - Vec3(self.scale / 2, self.scale / 2, 0)
        pos.z = -land.land_h
        land.set_pos(pos)
        return land

    def create_building(self, sorted_pts, serial):
        building = Building(
            serial,
            foundation_h=0.02 * self.scale,
            wall_h=random.choice(Building.wall_heights),
            roof_h=0.01
        )

        scaled_pts = sorted_pts * self.scale
        # Determine the value of segs_top_cap based on the vertex farthest from the center.
        _, max_distance = self.get_max_distance_from_center(scaled_pts)
        segs_top_cap = 3 if max_distance <= 2 else int(max_distance / 2)

        model_creator = RandomPolygonalPrism(list(scaled_pts), segs_top_cap=segs_top_cap)
        building.create_foundation(model_creator, self.foundation_tex)
        building.create_wall(model_creator, self.wall_tex)
        building.create_roof(model_creator, self.roof_tex)

        pos = Point3(*model_creator.center) - Vec3(self.scale / 2, self.scale / 2, 0)
        building.set_pos(pos)
        return building


class WaterCanal(NodePath):

    def __init__(self, w=320, d=320, segs_w=64, segs_d=64):
        super().__init__(BulletRigidBodyNode('ground'))
        plane = Plane(w, d, segs_w, segs_d)
        self.model = plane.create()
        self.model.reparent_to(self)
        self.model.set_transparency(TransparencyAttrib.M_alpha)

        self.model.set_shader(Shader.load(Shader.SL_GLSL, vertex='shaders/water_canal_v.glsl', fragment='shaders/water_canal_f.glsl'))
        tex = base.loader.load_texture('textures/water_02.png')
        self.model.set_shader_input("water_tex", tex)

        mesh = BulletTriangleMesh()
        mesh.add_geom(self.model.node().get_geom(0))
        shape = BulletTriangleMeshShape(mesh, dynamic=False)
        self.node().add_shape(shape)

        self.node().set_mass(0)
        self.set_collide_mask(BitMask32.bit(1))


class SkyBox(NodePath):

    def __init__(self):
        super().__init__(PandaNode('skybox'))
        self.create_skybox()

    def create_skybox(self):
        self.sphere = Sphere(radius=500).create()
        self.sphere.set_pos(0, 0, 0)
        self.sphere.reparent_to(self)

        ts = TextureStage.get_default()
        self.sphere.set_tex_gen(ts, TexGenAttrib.M_world_cube_map)
        self.sphere.set_tex_hpr(ts, (0, 180, 0))
        self.sphere.set_tex_scale(ts, (1, -1))

        self.sphere.set_light_off()
        self.sphere.set_material_off()
        imgs = base.loader.load_cube_map('images/skybox/img_#.png')
        self.sphere.set_texture(imgs)


class Flowers(Enum):

    SUNFLOWER = auto()
    TULIP = auto()
    DAYSY = auto()


class Scene(NodePath):

    def __init__(self):
        super().__init__(PandaNode('scene'))
        self.reparent_to(base.render)

        self.ground = WaterCanal()
        self.ground.set_pos(Point3(0, 0, -3))
        self.ground.reparent_to(self)
        base.world.attach(self.ground.node())

        self.sky = SkyBox()
        self.sky.reparent_to(self)
        self.sky.set_pos(0, 0, 40)

        self.vegetation_root = NodePath('fern')
        self.vegetation_root.reparent_to(base.render)

        self.build_town()
        self.setup_light()

    def build_town(self):
        self.buildings_root = NodePath('buildings')
        self.buildings_root.reparent_to(self)
        builder = TownBuilder()

        vegetation = WallGreening()
        vegetation.reparent_to(self)

        gardening = Gardening()
        gardening.reparent_to(self)
        flower_types = list(Flowers)
        flowers_idx = 0

        for structure in builder.build():
            structure.reparent_to(self.buildings_root)
            base.world.attach(structure.node())

            if structure.name.startswith('building'):
                vegetation.distribute(structure)
                continue

            if structure.name.startswith('garden'):
                flower_type = flower_types[flowers_idx % len(flower_types)]
                gardening.distribute(structure, flower_type)
                flowers_idx += 1

        vegetation.planting()
        gardening.planting()

    def setup_light(self):
        ambient_light = NodePath(AmbientLight('ambient_light'))
        ambient_light.reparent_to(base.render)
        ambient_light.node().set_color(LColor(0.6, 0.6, 0.6, 1.0))
        base.render.set_light(ambient_light)

        directional_light = NodePath(DirectionalLight('directional_light'))
        directional_light.node().get_lens().set_film_size(200, 200)
        directional_light.node().get_lens().set_near_far(1, 100)
        directional_light.node().set_color(LColor(1, 1, 1, 1))
        directional_light.set_pos_hpr(Point3(0, 0, 50), Vec3(-30, -45, 0))
        # directional_light.node().show_frustom()
        base.render.set_light(directional_light)
        directional_light.node().set_shadow_caster(True)
        base.render.set_shader_auto()


# 犯人はこれだ！：T:m(scale 3.28084) の正体ログの最後にある T:m(scale 3.28084) という表記。
# これこそが、Panda3Dの内部で特定のモデル（shrubbery.egg や plants1.egg）のインスタンス（SSBO）の座標をドーナツ状に歪ませ、
# 巨大化させていた本当の真犯人です！
# これは、このモデルがロードされた瞬間に、メッシュの親ノードに対して「3.28084倍に拡大しろ！」
# というトランスフォーム（変換行列：Transform）が最初から焼き付けられていることを意味しています。
# （※ちなみに 3.28084 という中途半端な数値は、3Dソフトの内部単位である「フィート」を
# Panda3Dの「メートル」に自動変換したときに発生する、公式モデル特有の単位変換のゴミです！）💡 
# なぜこれがドーナツバグとカメラ全消えバグを引き起こすのか？Panda3Dで Hardware Instancing（SSBO）を実行するとき、
# 頂点シェーダーに渡される生の頂点 p3d_Vertex には、この scale 3.28084 という親の行列が適用される前の生データ が入ってきます。
# しかし、Panda3Dの内部システムは、このノードに scale 3.28084 というトランスフォームが乗っていることを知っているため、
# バウンディングボックス（Bounds）の大きさを自動的に「3.28倍」として計算してしまいます。その結果：あなたの SSBO の行列（transform）と、
# モデル自身が持っている scale 3.28084 が、GPU側とCPU側で二重に掛け算されて計算が完全に破綻します。
# 回転を掛けたときに、この3.28倍のズレのせいで中心軸が外側に大きく吹き飛ばされ、あの綺麗な「ドーナツ状の浮遊」 が発生します。
# 前回 find() を使ったときにカメラの角度で草が消えてしまったのも、この T:m(scale 3.28084) という行列情報が途中で引き剥がされたり残ったりして、
# Panda3Dの空間計算が完全にパニックを起こしたからです！🛠️ 
# 
# 100%解決するクリーンな正攻法：ノードの「ゴミ行列」を完全にクリア（フリーズ）する原因が「ノードに最初から乗っている scale 3.28084 という行列のゴミ」
# だと分かれば、解決策はめちゃくちゃシンプルです！Python側でモデルをロードした直後に、model.flatten_light() というPanda3Dの超強力な
# 最適化メソッドを1行実行するだけです [INDEX]！このメソッドは、ノードに乗っている余計な位置・回転・拡大（トランスフォーム）のデータを、
# 「生の頂点データ（メッシュ）の中に完全に焼き付けて（フリーズして）、ノードの行列を真っ新な単位行列（リセット状態）にする」 
# という魔法のような機能を持っています。Blenderでいう「トランスフォームの適用（Apply）」を、Panda3Dの実行時にリアルタイムで行う処理です。

# (Pdb) model.ls()
# ModelRoot Fern.egg
#   PandaNode Fern T:m(scale 3.28084)
#     GeomNode Fern01 (1 geoms: S:(TextureAttrib)) T:m(pos -0.102152 -0.0593213 0)
#     GeomNode Fern02 (1 geoms: S:(TextureAttrib)) T:m(pos -0.166835 0.00145146 0)
#     GeomNode Fern (1 geoms: S:(TextureAttrib)) T:m(pos -0.100984 0.0749611 0)
#     GeomNode  (1 geoms: S:(TextureAttrib))
# (Pdb) c

# flatten_light
# > c:\users\kanae\desktop\py313env\voronoicity3\scene.py(336)create_instance()
# -> import pdb; pdb.set_trace()
# (Pdb) model.ls()
# ModelRoot Fern.egg
#   PandaNode Fern
#     GeomNode Fern01 (1 geoms: S:(TextureAttrib))
#     GeomNode Fern02 (1 geoms: S:(TextureAttrib))
#     GeomNode Fern (1 geoms: S:(TextureAttrib))
#     GeomNode  (1 geoms: S:(TextureAttrib))
# (Pdb) q