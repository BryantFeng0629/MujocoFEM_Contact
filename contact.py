#此程序用到的相关工具库和包：
#Python version==3.9.1;物理引擎仿真 Mujoco3.2.7;有限元仿真工具库(在Windows上兼容性较好) Sfepy version==2024.4;网格生成工具库 tetgen version==0.6.4
import mujoco
import numpy as np
import mujoco_viewer
from sfepy.discrete.fem import FEDomain, Field
from sfepy.discrete import Integral, Material, Problem, Equations, FieldVariable, Equation
from sfepy.terms import Term
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.base.base import IndexedStruct
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
import meshio
from stl.mesh import Mesh as ms
import tetgen
from sfepy.discrete.evaluate import Evaluator
from sfepy.discrete.fem import Mesh

#1.定义mujoco所使用的XML文件，这里描述的是一个任意形状的物体自由落体在另外一个固定的任意形状物体上（通过外部的.stl文件导入，这里使用conaffinity="0"关掉接触
free_body_MJCF = """
<mujoco>
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <asset>
    <mesh file="./MujocoModels/boxm.STL"/>
    <mesh file="./MujocoModels/pan.STL"/>
  </asset>
  
  <worldbody>
    <light name="light" pos="0 0 1" diffuse="1.0 1.0 1.0" specular="1.5 1.5 1.5" ambient="0.5 0.5 0.5"/>
    <!-- 地面 -->
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.6 0.4 1"/>

    <!-- 第一个球体（静止） -->
    <body name="geom1" pos="0 0 0" >
       <geom name="geom1_geom" type="mesh" mesh="boxm" density="1000" rgba="1 0 0 1" conaffinity="0"/>
    </body>

    <!-- 第二个球体（自由落体） -->
    <body name="geom2" pos="0.7 0.2 2">
      <freejoint/>
      <geom name="geom2_geom" type="mesh" mesh="pan" density="1000" rgba="0 0 1 1" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# 2.定义stl文件生成函数，此函数用来将mujoco每一次迭代过程中的几何体输出为对应stl
def export_stl(geom_id, step, model, data, inputfile):
    # 获取几何体类型和mesh信息
    # 获取关联的物体ID和mesh数据
    body_id = model.geom_bodyid[geom_id]
    mesh_id = model.geom_dataid[geom_id]
    vertadr = model.mesh_vertadr[mesh_id]
    vertnum = model.mesh_vertnum[mesh_id]
    faceadr = model.mesh_faceadr[mesh_id]
    facenum = model.mesh_facenum[mesh_id]
    # 提取顶点和面
    vertices = model.mesh_vert[vertadr:vertadr + vertnum].copy()
    faces = model.mesh_face[faceadr:faceadr + facenum]
    # 获取物体的全局位姿
    body_quat = data.xquat[body_id].copy().reshape(4, 1)
    # 获取几何体相对于物体的局部位姿
    geom_quat = model.geom_quat[geom_id].copy().reshape(4, 1)
    # 计算全局位姿：物体位姿 * 几何体局部位姿
    rot_body = np.zeros(9, dtype=np.float64).reshape(9, 1)  # 形状 (9,1)
    rot_geom = np.zeros(9, dtype=np.float64).reshape(9, 1)
    rot_body.flags.writeable = True
    rot_geom.flags.writeable = True
    # 物体旋转矩阵
    mujoco.mju_quat2Mat(rot_body, body_quat)
    # 几何体局部旋转矩阵
    mujoco.mju_quat2Mat(rot_geom, geom_quat)
    # 合并旋转：R_body * R_geom
    rot_body_mat = rot_body.reshape(3, 3, order='F')
    rot_geom_mat = rot_geom.reshape(3, 3, order='F')
    combined_rot = np.dot(rot_body_mat, rot_geom_mat)
    # 全局位置：body_pos + rot_body * geom_pos
    global_pos = data.xpos[body_id] + np.dot(rot_body_mat, model.geom_pos[geom_id])
    # 变换顶点到世界坐标系
    world_vertices = np.dot(vertices, combined_rot.T) + global_pos
    # 生成STL文件,这里使用stl工具库的Mesh功能（注意这里生成的是stl的表面mesh而不是有限元中的体mesh),ms为Mesh的简化写法
    mesh_data = np.zeros(faces.shape[0], dtype=ms.dtype)
    stl_mesh = ms(mesh_data, remove_empty_areas=False)
    #将顶点信息录入stl_mesh
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = world_vertices[face[j]]
    filename = f"geom_{geom_id}_step_{step}.stl"
    stl_mesh.save(inputfile)

# 3. stl到medit .mesh文件的转换函数。是将上述函数生成的stl文件转化为.mesh格式的文件以供FEM使用
def convert_stl_to_medit_mesh(stl_path, output_mesh_path):
    # 首先定义用于生成网格的单元类型映射表
    medit_cell_type_map = {
        2: "line",
        3: "triangle",
        4: "quad",
        9: "prism",
        10: "tetra",
    }
    # 读取STL表面网格
    surface_mesh = meshio.read(stl_path)
    points = surface_mesh.points.astype(np.float64)
    vertices = surface_mesh.points.reshape(-1, 3)
    # 计算STL几何体的顶点和低点坐标
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    # 表面为三角形网格
    triangles = surface_mesh.cells_dict["triangle"]
    # 根据读取的表面节点和三角形等坐标和矩阵信息，生成四面体网格
    tet = tetgen.TetGen(
        points.astype(np.float64),  # 确保数据类型正确 points为float64而triangles为int32
        triangles.astype(np.int32)
    )
    tet.tetrahedralize("pq2.0a0.001Y") # 参数说明： p执行 Delaunay 四面体化，q控制四面体形状质量（最小面角约束），a限制单个四面体的最大体积（细化密度），Y保持输入表面网格的原始边界（防止修改）
    volume_mesh = tet.grid
    # 这里需要根据函数开始的时候自定义的单元类型映射表去定义单元类型，这里选择10: "tetra"，也就是四面体单元
    cell_code = list(volume_mesh.cells_dict.keys())[0]
    cell_type = medit_cell_type_map.get(cell_code, "unknown")
    print("TetGen生成的单元类型代码:", cell_code)  # 例如输出: 'tetra'
    if cell_type == "unknown":
        raise ValueError(f"不支持的单元类型编码: {cell_code}")
    print(cell_type)
    cells = [meshio.CellBlock(
        cell_type,
        volume_mesh.cells_dict[cell_code]
    )]
    # 使用meshio输出medit .mesh格式的网格文件，适用于绝大多数FEM需要的格式
    mesh = meshio.Mesh(volume_mesh.points, cells)
    meshio.write(output_mesh_path, mesh, file_format="medit")
    return min_coords,max_coords

# 4. 定义合并多个网格到一个成体网格的函数。在将多个stl文件转换为多个.mesh文件后，需要将多个mesh文件（这里是两个）合并为一整个mesh文件以供FEM使用
def mergemesh(outputfile1,outputfile2):
    # 导入生成的mesh网格文件
    m0 = Mesh.from_file(outputfile1)
    m1 = Mesh.from_file(outputfile2)
    # 合并网格
    coors = np.vstack([m0.coors, m1.coors + [0.0, 0.0, 0.0]])  # 这里实际上可以自由定义初始的间距，不过这里导入的mesh文件已经包含mujoco导出stl时的空间坐标关系，所以这里设置为0
    # 单元连接关系
    conn = np.vstack([m0.get_conn('3_4'), m1.get_conn('3_4') + m0.n_nod])
    # 材料分组 (0: 底部集合体, 1: 顶部集合体)
    mat_id = np.zeros(conn.shape[0], dtype=np.int32)
    mat_id[m0.n_el:] = 1
    # 这里直接使用FEM工具sfepy中的Mesh类（注意和stl工具库的Mesh类区别开）生成合并的网格数据，返回这个网格数据
    return Mesh.from_data('two_blocks', coors, None, [conn], [mat_id], ['3_4'])

# 5. 使用有限元工具库sfepy定义有限元接触问题，并进行位移和接触力的计算
def define_contact_problem(outputfile1,outputfile2,vtkfile,min_coords_1,min_coords_2,max_coords_1,max_coords_2):
    # 生成网格
    mesh = mergemesh(outputfile1,outputfile2)
    # 定义计算域
    domain = FEDomain('domain', mesh)
    # 定义三个计算域，'Omega'为全局，'Omega0'为mujoco中的几何体1，'Omega1'为mujoco中的几何体2
    domain.create_region('Omega', 'all')
    domain.create_region('Omega0', 'cells of group 0')
    domain.create_region('Omega1', 'cells of group 1')
    # 定义边界和接触面，这里的边界条件仅适用于自上而下的自由落体，如果有更多维度和方向的接触，需要在Mujoco/FEM里增加一个能够识别接触点具体坐标和位置的算法，然后根据具体接触的顶点来定义接触面。
    eps = 0.1
    bottom = domain.create_region('Bottom', 'vertices in z < %.10f' % (min_coords_1[2]), 'facet')
    Contact0 = domain.create_region('Contact0', f'(vertices in (z > %f) *v vertices in (z<%f) *v r.Omega0)' % (min_coords_2[2]-eps,max_coords_1[2]+eps), 'facet')
    Contact1 = domain.create_region('Contact1', f'(vertices in (z > %f) *v vertices in (z<%f) *v r.Omega1)' % (min_coords_2[2]-eps,max_coords_1[2]+eps), 'facet')
    Contact = domain.create_region('Contact', 'r.Contact0 +s r.Contact1', 'facet')
    # 定义场（此处为经典的位移场）
    field = Field.from_args('displacement', np.float64, 'vector', domain.regions['Omega'], approx_order=1)
    # 定义变量
    u = FieldVariable('u', 'unknown', field,0)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    # 定义材料，接触函数所需的参数（接触惩罚刚度.epss），以及外部作用力
    D = stiffness_from_youngpoisson(3, 2e10, 0.3)
    solid = Material('solid', D=D)
    contact_param = {'.epss': 2e7}
    contact_params = {**contact_param}
    contact = Material('contact', **contact_params)
    f = Material('f', val=[[0], [0],[-2352]])
    # 定义体积积分和表面积分
    integral_volume = Integral('i', order=3)
    integral_surface = Integral('i', order=2)
    # 定义接触函数
    t1 = Term.new('dw_lin_elastic(solid.D, v, u)',
                  integral_volume, domain.regions['Omega'], solid=solid, v=v, u=u)  # 弹性项
    t2 = Term.new('dw_contact(contact.epss, v, u)',
                  integral_surface, Contact, contact=contact, v=v,u=u)  # 接触项
    t3 = Term.new('dw_volume_lvf(f.val, v)',
                  integral_volume,  domain.regions['Omega'], f=f, v=v) # 外力项 （e.g. 重力）
    eq = Equation('balance', t1+t2-t3 )
    eqs = Equations([eq])
    # 边界条件
    fix_u = EssentialBC('fix_u', bottom, {'u.all': 0.0})
    # 定义求解器，通过ScipyDirect和Newton类，实现了一个可定制化的牛顿法非线性求解器
    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton({}, lin_solver=ls, status=nls_status)
    # 定义有限元接触问题
    pb = Problem('contact', equations=eqs)
    pb.set_bcs(ebcs=Conditions([fix_u]))
    pb.set_solver(nls)
    # 求解
    state = pb.solve()
    # 提取接触力残差（包含在t2公式中）
    ev = pb.evaluate
    evaluator = Evaluator(pb)
    residual_full = evaluator.eval_residual(
        state(),  # 当前解向量
        select_term=lambda term: term.name == 'dw_contact' ,
        is_full=True # 关键：强制返回全自由度残差
    )
    # 提取接触顶点的信息（因几何体0固定，接触力施加在几何体1上）
    # 获取几何体1的接触区域对象（预定义的Contact1区域）
    contact_region = pb.domain.regions['Contact1']
    # 获取位移场对象（用于后续坐标提取）
    field = pb.fields['displacement']
    # 提取接触区域的顶点索引（n_nodes表示接触节点总数）
    contact_nodes = contact_region.vertices  # 形状 (n_nodes,)
    print("contact_nodes",contact_nodes)
    # 根据顶点索引获取实际坐标（mesh.coors存储全网格坐标）
    contact_nodes_coordinates = mesh.coors[contact_nodes]
    # 从全局残差向量中提取接触力分量
    contact_force_vector = residual_full
    # 将一维残差向量重塑为二维力矩阵（每行对应一个节点的xyz分量）
    contact_force_matrix = contact_force_vector.reshape((-1, 3))  # (n_nodes, 3)
    # 调整节点索引（这里因全局编号偏移需要自行校准，后续也可以改进算法，在Mujoco中判断好接触点然后返回到这个函数里进行自动计算）
    contact_nodes_shifted = contact_nodes-4
    # 创建存储接触力信息的字典结构
    contact_info_dict = {}
    for node, coord in zip(contact_nodes_shifted, contact_nodes_coordinates):
        # 提取力
        force = contact_force_matrix[node,:3]
        # 以节点号为键，存储坐标和力向量
        contact_info_dict[node] = {
            'coordinate': coord.tolist(),
            'force': force.tolist()
        }
    # 输出结果
    pb.save_state(vtkfile, state)
    return pb, contact_info_dict

# 6. 主仿真循环
# ==== MuJoCo仿真初始化 ====
model = mujoco.MjModel.from_xml_string(free_body_MJCF)
data = mujoco.MjData(model)
n_steps = 10000
viewer = mujoco_viewer.MujocoViewer(model, data)
# ==== 几何体参数获取 ====
# 通过名称获取几何ID
geom1_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom1_geom")
geom2_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom2_geom")
# 获取几何体尺寸数据
geom1_radius = model.geom_size[geom1_geom_id][0]  # 第一个球体的半径
geom2_radius = model.geom_size[geom2_geom_id][0]  # 第二个球体的半径

# ==== 主仿真循环 ====
for i in range(n_steps):
    mujoco.mj_step(model, data)
    viewer.render()
    # ==== 实时数据采集 ====
    # 获取几何体中心坐标（geom_xpos存储所有几何体的全局坐标）
    geom1_pos = data.geom_xpos[geom1_geom_id]
    geom2_pos = data.geom_xpos[geom2_geom_id]
    # 生成当前步的STL和mesh文件路径（用于后续离线分析）
    inputfile1 = "./model/stl/geom_1_step_" + str(i) + ".stl"
    outputfile1 = "./model/mesh/geom_1_step_" + str(i) + ".mesh"
    inputfile2 = "./model/stl/geom_2_step_" + str(i) + ".stl"
    outputfile2 = "./model/mesh/geom_2_step_" + str(i) + ".mesh"
    vtkfile = "./model/vtk/geom_step_" + str(i) + ".vtk"
    # 导出STL模型
    export_stl(geom1_geom_id, i, model, data, inputfile1)
    export_stl(geom2_geom_id, i, model, data, inputfile2)
    # 将导出的STL模型转化为medit mesh网格文件,并得到两个几何体的最高点和最低点坐标信息，以便在FEM中设置边界条件使用
    min_coords_1,max_coords_1=convert_stl_to_medit_mesh(inputfile1, outputfile1)
    min_coords_2,max_coords_2=convert_stl_to_medit_mesh(inputfile2, outputfile2)
    # ==== 碰撞检测逻辑 ====
    # 计算两球中心距离（目前只采用最简单的逻辑）
    distance = np.linalg.norm(geom1_pos - geom2_pos)
    body1_id = 1
    body2_id = 2
    # 碰撞条件：实际距离 ≤ 中心和+0.5（0.5为安全阈值，防止穿透），注意这里的0.5是粗鲁的估算距离，因为现在还没有考虑能精准判断是否接触的算法。如果需要精准判断接触与否，需要更高级的算法，一种可能的操作是使用bounding box来判断
    if distance <= (geom1_radius+geom2_radius+0.45):
        print(f"碰撞发生！距离：{distance:.4f}，阈值：{geom1_radius + geom2_radius:.4f}")
        # 改变几何体颜色（RGBA格式，[红,绿,蓝,透明度]），这里若碰撞发生，则将几何体在mujoco中的颜色改变来进行区分
        model.geom_rgba[geom1_geom_id] = [0, 1, 0, 1]  # 红色球变绿色
        model.geom_rgba[geom2_geom_id] = [0, 1, 0, 1]  # 蓝色球变绿色

        # ==== 接触力计算 ====
        # 获取几何体质心坐标
        com_geom1 = data.xipos[body1_id]
        com_geom2 = data.xipos[body2_id]
        # 调用FEM接触力计算模块
        pb, contact_force_dict = define_contact_problem(outputfile1, outputfile2, vtkfile,min_coords_1,min_coords_2,max_coords_1,max_coords_2)
        print("Contact_force",contact_force_dict)

        # ==== 合力/扭矩计算 ====
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for node, info in contact_force_dict.items():
            force = np.array(info['force'])  # 力向量 [Fx, Fy, Fz]
            coord = np.array(info['coordinate'])  # 接触点坐标 [x, y, z]
            r_geom2 = coord - com_geom2
            # 计算扭矩：r × F（相对质心的位置矢量）
            total_torque = np.cross(r_geom2, force)
            total_force  = force
            # ==== 力施加机制 ====
            # 清空之前施加的力（xfrc_applied存储所有body的外力/力矩）
            data.xfrc_applied[:, :] = 0
            # 向刚体2施加合力和扭矩（前3元素为力，后3为扭矩）
            data.xfrc_applied[body2_id, :3] = total_force
            data.xfrc_applied[body2_id, 3:] = total_torque
    # 无碰撞时重置颜色和外力
    else:
        model.geom_rgba[geom1_geom_id] = [1, 0, 0, 1]  # 恢复红色
        model.geom_rgba[geom2_geom_id] = [0, 0, 1, 1]  # 恢复蓝色
        data.xfrc_applied[:, :] = 0