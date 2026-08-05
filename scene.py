import math
import random

import numpy as np
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
from panda3d.bullet import BulletConvexHullShape, BulletCylinderShape, ZUp
from panda3d.core import NodePath, PandaNode
from panda3d.core import Point3, Vec3, BitMask32, LColor
from panda3d.core import TextureStage, TransformState, TexGenAttrib
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import OmniBoundingVolume
from panda3d.core import Shader, ShaderBuffer, GeomEnums

from noise import Fractal3D, PerlinNoise
from shapes import RandomPolygonalPrism
from shapes import Plane, Cylinder, Sphere
from voronoi_generator.voronoi_2d import BoundedVoronoiGenerator, ConvexPolygonGenerator
from voronoi_generator.voronoi_2d import Polygon2DMixin
from voronoi_generator.polygon_mixin import PolygonMixin
from voronoi_generator.voronoi_2d.rounded_voronoi import RoundedVoronoiGenerator


class Building(PolygonMixin, NodePath):

    wall_heights = [10, 20, 30, 40, 50, 60, 70, 80]

    def __init__(self, serial, foundation_h, wall_h, roof_h):
        super().__init__(BulletRigidBodyNode(f'building_{serial}'))
        self.set_collide_mask(BitMask32.bit(1))
        self.node().set_mass(0)

        self.foundation_h = foundation_h
        self.wall_h = wall_h
        self.roof_h = roof_h

    def assemble(self, model, pos, name):
        shape = BulletConvexHullShape()
        shape.add_geom(model.node().get_geom(0))
        self.node().add_shape(shape, TransformState.make_pos(pos))
        model.set_pos(pos)
        model.set_name(name)
        model.reparent_to(self)

    def create_foundation(self, model_creator, foundation_tex):
        """Create building foundation.
        """
        # create_model.
        model_creator.height = self.foundation_h
        foundation = model_creator.create()

        # set texture.
        su = self.round_off(model_creator.edge_length / 20)
        foundation.set_tex_scale(TextureStage.get_default(), (su, 0.5))
        foundation.set_texture(foundation_tex)

        # parent to self.
        self.assemble(foundation, Point3(0, 0, 0), name="foundation")

    def create_wall(self, model_creator, wall_tex):
        """Create building wall.
        """
        # create model.
        model_creator.height = self.wall_h
        model_creator.segs_a = int(self.wall_h / 2)
        wall = model_creator.create()

        # set texture.
        u = model_creator.edge_length / 50
        su = np.ceil(u * 3) / 3
        v = self.wall_h / 50
        sv = np.ceil(v * 4) / 4
        wall.set_tex_scale(TextureStage.get_default(), (su, sv))
        wall.set_texture(wall_tex)

        # parent to self.
        pos = Point3(0, 0, self.foundation_h)
        self.assemble(wall, pos, "wall")

    def create_roof(self, model_creator, roof_tex):
        """Create building foundation.
        """
        # create model.
        model_creator.height = self.roof_h
        model_creator.segs_a = 3
        roof = model_creator.create()

        # set texture.
        su = self.round_off(model_creator.edge_length / 10)
        roof.set_tex_scale(TextureStage.get_default(), (su, 0.2))
        roof.set_texture(roof_tex)

        # parent to self.
        pos = Point3(0, 0, self.wall_h + self.foundation_h)
        self.assemble(roof, pos, "roof")

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


class Vegetation:

    def __init__(self):
        self.create_noise()

    def create_noise(self):
        perlin = PerlinNoise()

        self.noise_a = Fractal3D(
            perlin.pnoise3,
            gain=0.5,
            lacunarity=2.01,
            octaves=4,
            amplitude=1.0,
            frequency=1.2
        )
        self.noise_b = Fractal3D(
            perlin.pnoise3,
            gain=0.4,
            octaves=2,
            amplitude=1.0,
            frequency=0.09
        )
        self.noise_c = Fractal3D(
            perlin.pnoise3,
            gain=0.3,
            octaves=1,
            amplitude=1.0,
            frequency=0.08
        )

    def get_dentisy(self, x, y, z, max_height=10.0):
        """Calculating the density of plants covering a wall based on noise.
        """

        # Noise for planting plants in narrow rows.
        raw_a = self.noise_a.fractal(x, y, z)
        # Rough and Fine
        raw_b = self.noise_b.fractal(x, y, z)
        # Noise that determines how far down from the roof to plant the plants
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
            hang_exponent = 6.0
        elif noise_c < 0.50:
            hang_exponent = 3.5
        elif noise_c < 0.75:
            hang_exponent = 1.8
        else:
            hang_exponent = 0.8

        # Calculation of the Height Coefficient（0.0 〜 1.0）
        normalized_z = max(0.0, min(z / max_height, 1.0))
        height_factor = normalized_z ** hang_exponent

        # The smaller the height_factor, the higher the lower_bound becomes, and the fewer plants there are.
        lower_bound = 0.55 - (height_factor * 0.35)
        upper_bound = 0.75 - (height_factor * 0.25)

        if combined_noise < lower_bound:
            density = 0.0
        elif combined_noise > upper_bound:
            density = 1.0
        else:
            # smoothstep
            t = (combined_noise - lower_bound) / (upper_bound - lower_bound)
            density = t * t * (3.0 - 2.0 * t)

        return density

    def transform_plant(self, matrices_list, dummy_np, pos, normal, scale):
        """Using a dummy NodePath, calculate the plant's translation, rotation, and scaling,
           and store the results in a temporary list.
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

    def plant_vegetation(self, building):
        """Using noise to plant vegetation on the walls and rooftops of buildings
        """
        matrices_plants1 = []
        matrices_plants2 = []
        matrices_shrubbery = []
        matrices_fern = []

        vdata_mem = building.get_vdata_memview('wall')
        dummy = NodePath(PandaNode("dummy_transform"))

        # Store the array data in a temporary list by plant type.
        for i in range(0, len(vdata_mem), 12):
            x, y, z = vdata_mem[i: i + 3]

            # Avoid planting plants on the base of the 3d model.
            if z <= 0:
                continue

            # Change the plants based on the noise level.
            pos = Point3(x, y, z + building.foundation_h)
            normal = Vec3(*vdata_mem[i + 7: i + 10])

            if (val := self.get_dentisy(x, y, z, max_height=building.wall_h)) >= 1.0:
                scale = Vec3(0.1)
                self.transform_plant(matrices_plants2, dummy, pos, normal, scale)
            elif val >= 0.8:
                scale = Vec3(0.1)
                self.transform_plant(matrices_plants1, dummy, pos, normal, scale)
            elif val >= 0.35:
                scale = Vec3(0.08)
                self.transform_plant(matrices_shrubbery, dummy, pos, normal, scale)
            elif val >= 0.25:
                scale = Vec3(0.01)
                self.transform_plant(matrices_fern, dummy, pos, normal, scale)
            else:
                continue

        # plants1
        if len(matrices_plants1) > 0:
            self.create_model(building, matrices_plants1, 'plants1/plants1.egg')

        # plants1 which color_scale is changed
        if len(matrices_plants2) > 0:
            color_scale = LColor(1.1, 1.4, 1.1, 1.0)
            self.create_model(building, matrices_plants2, 'plants1/plants1.egg', color_scale=color_scale)

        # shrubbery
        if len(matrices_shrubbery) > 0:
            self.create_model(building, matrices_shrubbery, 'shrubbery/shrubbery.egg')

        # fern
        if len(matrices_fern) > 0:
            color_scale = LColor(0.2, 0.6, 0.2, 1.0)
            self.create_model(building, matrices_fern, 'fern/Fern.egg', color_scale=color_scale)

    def create_model(self, building, matrices, file_path, color_scale=None):
        arr_matrices = np.array(matrices, dtype=np.float32)
        raw_buffer_data = arr_matrices.tobytes()
        model = base.loader.load_model(f'models/{file_path}')

        if color_scale is not None:
            # Set the color to white first, since the color of plant1 didn't change with just set_color_scale.
            model.set_color(1, 1, 1, 1)
            model.set_color_scale(*color_scale, 1)

        model.flatten_light()
        model.reparent_to(building)
        model.set_pos(0, 0, 0)
        model.set_hpr(0, 0, 0)

        # Prevent curling.
        model.node().set_bounds(building.node().get_bounds())
        model.node().set_final(True)

        # Set the number of instances.
        instance_cnt = len(matrices)
        model.set_instance_count(instance_cnt)
        model.set_shader(Shader.load(Shader.SL_GLSL, vertex='shaders/instancing_v.glsl', fragment='shaders/instancing_f.glsl'))
        model.set_shader_input("instanced_object", ShaderBuffer('DataBuffer', raw_buffer_data, GeomEnums.UH_static))


class TownBuilder(Polygon2DMixin):

    def __init__(self, scale=256):
        self.scale = scale
        self.create_textures()

    def create_textures(self):
        self.foundation_tex = base.loader.load_texture('textures/foundation2.png')
        # self.foundation_tex = base.loader.load_texture('textures/dark_gray_concrete.jpg')
        self.wall_tex = base.loader.load_texture('textures/gray_brick.png')
        self.roof_tex = base.loader.load_texture('textures/dark_gray_concrete.jpg')
        self.spot_tex = base.loader.load_texture('textures/concrete_01.jpg')
        self.grass_tex = base.loader.load_texture('textures/grass_04.jpg')
        self.tree_model = base.loader.load_model('models/pinetree/tree2.bam')

    def build(self):
        for i, region in enumerate(BoundedVoronoiGenerator(cnt_points=5, shrink=0.03)):
            poly_pts = np.array([pt for pt in ConvexPolygonGenerator(region)])

            if i == 1:
                return

            # for j, pts in enumerate(BoundedVoronoiGenerator(pts=poly_pts, bnd=region, shrink=0.003)):
            for j, pts in enumerate(RoundedVoronoiGenerator(pts=poly_pts, bnd=region)):
                # import pdb; pdb.set_trace()

                if len(pts) == 0:
                    continue

                polygon = np.insert(pts, pts.shape[1], 0, axis=1)

                serial = f'{i}_{j}'

                # if j == 0:
                #     if nd := self.create_green(polygon, serial):
                #         yield nd
                #         continue

                sorted_pts = self.sort_counter_clockwise(polygon)
                yield self.create_building(sorted_pts, serial)

    def create_green(self, sorted_pts, serial):
        center, radius = self.get_max_inscribed_circle(sorted_pts)
        spot_rad = radius * self.scale
        inner_radius = spot_rad - 0.5
        height = 0.001 * self.scale

        # If the radius of a circular garden is too small, do not create the garden.
        if (n := int(inner_radius) - 2) <= 0:
            return None

        garden_np = Garden(serial)
        # Create the edge of the circular garden.
        edge = Cylinder(spot_rad, inner_radius=inner_radius, height=height).create()
        edge.set_texture(self.spot_tex)
        garden_np.assemble(edge, Point3(0, 0, 0), is_convex=False)

        # Create the lawn area of the circular garden
        green = Cylinder(inner_radius, height=height - 0.1).create()
        green.set_texture(self.grass_tex)
        garden_np.assemble(green, Point3(0, 0, 0))

        # Plant trees.
        pos_candidates = random.sample(range(-n, n), 2 * n - 2)

        for i in range(0, len(pos_candidates) - 1, 2):
            x, y = pos_candidates[i: i + 2]
            dist = (x ** 2 + y ** 2) ** 0.5
            if dist < inner_radius:
                garden_np.plant_tree(self.tree_model, Point3(x, y, 0))

        pos = Point3(*center, 0) * self.scale - Vec3(self.scale / 2, self.scale / 2, 0)
        garden_np.set_pos(pos)
        return garden_np

    def create_building(self, sorted_pts, serial):
        scaled_pts = sorted_pts * self.scale
        # Determine the value of segs_top_cap based on the vertex farthest from the center.
        center = np.mean(scaled_pts, axis=0)
        distances = np.sum((scaled_pts - center) ** 2, axis=1)
        max_distance = np.max(distances) ** 0.5
        segs_top_cap = 3 if max_distance <= 2 else int(max_distance / 2)
        model_creator = RandomPolygonalPrism(list(scaled_pts), segs_top_cap=segs_top_cap)

        building = Building(
            serial,
            foundation_h=0.02 * self.scale,
            wall_h=random.choice(Building.wall_heights),
            roof_h=0.01
        )

        building.create_foundation(model_creator, self.foundation_tex)
        building.create_wall(model_creator, self.wall_tex)
        building.create_roof(model_creator, self.roof_tex)

        pos = Point3(*model_creator.center) - Vec3(self.scale / 2, self.scale / 2, 0)
        building.set_pos(pos)
        return building


class Ground(NodePath):

    def __init__(self, w=280, d=280, segs_w=16, segs_d=16):
        super().__init__(BulletRigidBodyNode('ground'))
        plane = Plane(w, d, segs_w, segs_d)
        self.model = plane.create()
        self.model.set_texture(base.loader.load_texture('textures/concrete_01.jpg'))
        self.model.reparent_to(self)

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


class Scene(NodePath):

    def __init__(self):
        super().__init__(PandaNode('scene'))
        self.reparent_to(base.render)

        self.ground = Ground()
        self.ground.set_pos(Point3(0, 0, 0))
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
        builder = TownBuilder()
        vegetation = Vegetation()
        # vegetation.reparent_to(base.render)

        for building in builder.build():
            building.reparent_to(self.buildings_root)
            # import pdb; pdb.set_trace()
            base.world.attach(building.node())
            vegetation.plant_vegetation(building)
            # self.create_instance(building)
            # break
            # vegetation.plant()
        self.buildings_root.reparent_to(self)

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



class Garden(NodePath):

    def __init__(self, serial):
        super().__init__(BulletRigidBodyNode(f'garden_{serial}'))
        self.set_collide_mask(BitMask32.bit(1))
        self.node().set_mass(0)

    def assemble(self, model, pos, is_convex=True):
        if is_convex:
            shape = BulletConvexHullShape()
            shape.add_geom(model.node().get_geom(0))
        else:
            mesh = BulletTriangleMesh()
            mesh.add_geom(model.node().get_geom(0))
            shape = BulletTriangleMeshShape(mesh, dynamic=False)

        self.node().add_shape(shape, TransformState.make_pos(pos))
        model.set_pos(pos)
        model.reparent_to(self)

    def plant_tree(self, model, pos):
        tree = model.copy_to(self)
        tree.set_transform(TransformState.make_pos(Vec3(0, 0, -4)))

        end, tip = tree.get_tight_bounds()
        height = (tip - end).z
        shape = BulletCylinderShape(0.5, height, ZUp)
        self.node().add_shape(shape, TransformState.make_pos(pos))
        tree.set_pos_hpr_scale(pos, Vec3(), 1.6)
        tree.reparent_to(self)


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