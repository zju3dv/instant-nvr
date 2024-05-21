import os
import json
import os.path as osp
import numpy as np
import cv2
# import mesh_to_sdf
from psbody.mesh import Mesh
import argparse
import pickle
import trimesh
from tqdm import tqdm
from termcolor import colored

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def load_obj(path):
    model = {}
    pts = []
    tex = []
    faces = []

    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                pts.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                tex.append((float(strs[1]), float(strs[2])))

    uv = np.zeros([len(pts), 2], dtype=np.float32)
    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "f":
                face = (int(strs[1].split("/")[0]) - 1,
                        int(strs[2].split("/")[0]) - 1,
                        int(strs[4].split("/")[0]) - 1)
                texcoord = (int(strs[1].split("/")[1]) - 1,
                            int(strs[2].split("/")[1]) - 1,
                            int(strs[4].split("/")[1]) - 1)
                faces.append(face)
                for i in range(3):
                    uv[face[i]] = tex[texcoord[i]]

        model['pts'] = np.array(pts)
        model['faces'] = np.array(faces)
        model['uv'] = uv

    return model

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_smpl_faces(pkl_path):
    smpl = read_pickle(pkl_path)
    faces = smpl['f']
    return faces

def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat

def get_rigid_transformation(rot_mats, joints, parents):
    """
    rot_mats: 24 x 3 x 3
    joints: 24 x 3
    parents: 24
    """
    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints

    return transforms



def get_transform_params(smpl, params):
    """ obtain the transformation parameters for linear blend skinning
    """
    v_template = np.array(smpl['v_template'])

    # add shape blend shapes
    shapedirs = np.array(smpl['shapedirs'])
    betas = params['shapes']
    n = betas.shape[-1]
    m = shapedirs.shape[-1]
    n = min(m, n)
    v_shaped = v_template + np.sum(shapedirs[..., :n] * betas[None][..., :n], axis=2)

    # add pose blend shapes
    poses = params['poses'].reshape(-1, 3)
    # 24 x 3 x 3
    rot_mats = batch_rodrigues(poses)

    # obtain the joints
    joints = smpl['J_regressor'].dot(v_shaped)

    # obtain the rigid transformation
    parents = smpl['kintree_table'][0]
    A = get_rigid_transformation(rot_mats, joints, parents)

    # apply global transformation
    R = cv2.Rodrigues(params['Rh'][0])[0]
    Th = params['Th']

    return A, R, Th, joints


def get_grid_points(xyz):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    vsize = 0.025
    voxel_size = [vsize, vsize, vsize]
    # TODO: 这里有一个风险是, 可能对应的格点和原来的bound不是完全一致的, 在grid sample的时候可能会出现问题. 后面有时间解决一下.
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return pts

def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret

def get_bigpose_uv(frame, params_dir, vertices_dir, output_root):
    i = frame
    param_path = os.path.join(params_dir, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_model_path)
    # mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(smpl_model_path)
    # obtain the transformation parameters for linear blend skinning
    a, r, th, joints = get_transform_params(smpl, params)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - th, r)
    smpl_mesh = Mesh(pxyz, faces)

    bweights = smpl['weights']
    a = np.dot(bweights, a.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pxyz - a[:, :3, 3]
    r_inv = np.linalg.inv(a[:, :3, :3])
    pxyz = np.sum(r_inv * can_pts[:, None], axis=2)

    # calculate big pose
    poses = params['poses'].reshape(-1, 3)
    big_poses = np.zeros_like(poses).ravel()
    angle = 30
    big_poses[5] = np.deg2rad(angle)
    big_poses[8] = np.deg2rad(-angle)
    # big_poses = big_poses.reshape(-1, 3)
    # big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
    # big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
    # big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
    # big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])

    big_poses = big_poses.reshape(-1, 3)
    rot_mats = batch_rodrigues(big_poses)
    parents = smpl['kintree_table'][0]
    big_a = get_rigid_transformation(rot_mats, joints, parents)
    big_a = np.dot(bweights, big_a.reshape(24, -1)).reshape(-1, 4, 4)

    bigpose_vertices = np.sum(big_a[:, :3, :3] * pxyz[:, None], axis=2)
    bigpose_vertices = bigpose_vertices + big_a[:, :3, 3]

    smpl_mesh = Mesh(bigpose_vertices, faces)

    # create grid points in the pose space
    pts = get_grid_points(bigpose_vertices)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))

    uvs = barycentric_interpolation(uv_model['uv'][vert_ids], bary_coords)

    uvs = uvs.reshape(*sh[:3], 2).astype(np.float32)
    uv_path = os.path.join(output_root, "bigpose_uv.npy")
    np.save(uv_path, uvs)

    return uvs

def get_bigpose_blend_weights(frame, params_dir, vertices_dir, lbs_root):
    i = frame
    param_path = os.path.join(params_dir, '{}.npy'.format(i))
    vertices_path = os.path.join(vertices_dir, '{}.npy'.format(i))

    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_model_path)
    # mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(smpl_model_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    parent_path = os.path.join(lbs_root, 'parents.npy')
    np.save(parent_path, smpl['kintree_table'][0])
    joint_path = os.path.join(lbs_root, 'joints.npy')
    np.save(joint_path, joints)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    bweights = smpl['weights']
    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pxyz - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    pxyz = np.sum(R_inv * can_pts[:, None], axis=2)

    # calculate big pose
    poses = params['poses'].reshape(-1, 3)
    big_poses = np.zeros_like(poses).ravel()
    angle = 30
    big_poses[5] = np.deg2rad(angle)
    big_poses[8] = np.deg2rad(-angle)
    # big_poses = big_poses.reshape(-1, 3)
    # big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
    # big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
    # big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
    # big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])

    big_poses = big_poses.reshape(-1, 3)
    rot_mats = batch_rodrigues(big_poses)
    parents = smpl['kintree_table'][0]
    big_A = get_rigid_transformation(rot_mats, joints, parents)
    big_A = np.dot(bweights, big_A.reshape(24, -1)).reshape(-1, 4, 4)

    bigpose_vertices = np.sum(big_A[:, :3, :3] * pxyz[:, None], axis=2)
    bigpose_vertices = bigpose_vertices + big_A[:, :3, 3]

    bigpose_vertices_path = os.path.join(lbs_root, 'bigpose_vertices.npy')
    np.save(bigpose_vertices_path, bigpose_vertices)

    faces_path = os.path.join(lbs_root, 'faces.npy')
    np.save(faces_path, faces)

    smpl_mesh = Mesh(bigpose_vertices, faces)

    # create grid points in the pose space
    pts = get_grid_points(bigpose_vertices)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))
    bweights = barycentric_interpolation(smpl['weights'][vert_ids],
                                         bary_coords)

    # calculate the distance to the smpl surface
    norm = np.linalg.norm(pts - closest_points, axis=1)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)
    bweight_path = os.path.join(lbs_root, 'bigpose_bw.npy')
    np.save(bweight_path, bweights)

    # calculate sdf
    mesh = trimesh.Trimesh(bigpose_vertices, faces)
    # points, sdf = mesh_to_sdf.sample_sdf_near_surface(mesh,
    #                                                   number_of_points=250000)
    # translation, scale = compute_unit_sphere_transform(mesh)
    # points = (points / scale) - translation
    # sdf /= scale
    # sdf_path = os.path.join(lbs_root, 'bigpose_sdf.npy')
    # np.save(sdf_path, {'points': points, 'sdf': sdf})

    return bweights

def get_bweights(param_path, vertices_path, smpl_path):
    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)
    faces = get_smpl_faces(smpl_path)
    # mesh = get_o3d_mesh(vertices, faces)

    smpl = read_pickle(smpl_path)
    # obtain the transformation parameters for linear blend skinning
    A, R, Th, joints = get_transform_params(smpl, params)

    # transform points from the world space to the pose space
    pxyz = np.dot(vertices - Th, R)
    smpl_mesh = Mesh(pxyz, faces)

    # create grid points in the pose space
    pts = get_grid_points(pxyz)
    sh = pts.shape
    pts = pts.reshape(-1, 3)

    # obtain the blending weights for grid points
    vert_ids, norm = smpl_mesh.closest_vertices(pts, use_cgal=True)
    bweights = smpl['weights'][vert_ids]

    # closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    # vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
    #     closest_points, closest_face.astype('int32'))
    # bweights = barycentric_interpolation(smpl['weights'][vert_ids],
    #                                      bary_coords)

    # calculate the distance to the smpl surface
    # norm = np.linalg.norm(pts - closest_points, axis=1)

    A = np.dot(bweights, A.reshape(24, -1)).reshape(-1, 4, 4)
    can_pts = pts - A[:, :3, 3]
    R_inv = np.linalg.inv(A[:, :3, :3])
    can_pts = np.sum(R_inv * can_pts[:, None], axis=2)

    bweights = np.concatenate((bweights, norm[:, None]), axis=1)
    bweights = bweights.reshape(*sh[:3], 25).astype(np.float32)

    return bweights

def prepare_blend_weights(params_dir, vertices_dir, lbs_root):
    bweight_dir = os.path.join(lbs_root, 'bweights')
    os.makedirs(bweight_dir, exist_ok=True)

    for i in tqdm(range(begin_frame, end_frame, frame_interval)):
        param_path = os.path.join(params_dir, f'{i}.npy')
        vertices_path = os.path.join(vertices_dir, f'{i}.npy')
        bweights = get_bweights(param_path, vertices_path, smpl_model_path)
        bweight_path = os.path.join(bweight_dir, f'{i}.npy')
        np.save(bweight_path, bweights)


def prepare_dataset(data_root, output_root):
    params_dir=osp.join(data_root, 'smpl_params')
    vertices_dir=osp.join(data_root, 'smpl_vertices')
    lbs_root=osp.join(output_root, 'smpl_lbs')
    os.makedirs(lbs_root, exist_ok=True)

    log('getting bigpose uv ...')
    get_bigpose_uv(begin_frame, params_dir, vertices_dir, output_root)
    log('getting bigpose blend weights ...')
    get_bigpose_blend_weights(begin_frame, params_dir, vertices_dir,lbs_root)

    log('getting blend weights ...')
    prepare_blend_weights(params_dir, vertices_dir, lbs_root)



if __name__=='__main__':
    """
    ZJUMOCAP's pre-processing

    Inputs
    ----------
    for every human in zju-mocap : 
        params i: from params/{i}.npy
        vertices i: from vertices/{i}.npy
    SMPL model : SMPL_NEUTRAL.pkl
    SMPL uv : smpl_uv.obj

    Outputs
    -------
    smpl joints : lbs/joints.npy
    smpl parents : lbs/parents.npy
    bigpose vertices : lbs/bigpose_vertices.npy
    bigpose blend weight : lbs/bigpose_bw.npy
    bigpose uv : bigpose_uv.npy
    blend weight for frame i : bweights/{i}.npy
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--output_root", default="data", type=str)
    parser.add_argument("--smpl_model_path", default="SMPL_NEUTRAL.pkl", type=str)
    parser.add_argument("--smpl_uv_path", default="smpl_uv.obj", type=str)
    parser.add_argument('--ranges', type=int, default=None, nargs=3)
    parser.add_argument("--humans", default=None, type=str, nargs='+')
    args = parser.parse_args()

    data_root = args.data_root
    output_root = args.output_root
    smpl_model_path = args.smpl_model_path
    smpl_uv_path = args.smpl_uv_path
    begin_frame, end_frame, frame_interval = args.ranges
    humans = args.humans
    
    uv_model=load_obj(smpl_uv_path)
    if humans is None:
        log(f'Processing {data_root.split(os.sep)[-1]} ...')
        prepare_dataset(data_root, output_root)
    else:
        for human in humans:
            log(f'Processing {human} ...')
            sub_data_root = osp.join(data_root, human)
            sub_output_root = osp.join(output_root, human)

            prepare_dataset(sub_data_root, sub_output_root)


