import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
import numpy as np

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def load_custom_cameras_and_images(cameras_path, images_path):
    cameras = {}
    images = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            cameras[camera_id] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]]
            }
    with open(images_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith('#') or not line:
                idx += 1
                continue
            parts = line.split()
            if len(parts) == 10:  # First line of the block (IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME)
                try:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    image_name = parts[9]
                except Exception as e:
                    raise ValueError(f"Error parsing image metadata on line {idx}: {line}\n{e}")
                points_2d = []
                images[image_id] = {
                    'quaternion': [qw, qx, qy, qz],
                    'translation': [tx, ty, tz],
                    'camera_id': camera_id,
                    'name': image_name,
                    'points2d': points_2d
                }
                idx += 1  # Move to the next line
                continue
            points_line = line.split()
            if len(points_line) % 3 != 0:  # Each entry should have (X, Y, POINT3D_ID)
                raise ValueError(f"Malformed POINTS2D[] line: {line}")
            for i in range(0, len(points_line), 3):
                try:
                    x, y, point3d_id = float(points_line[i]), float(points_line[i + 1]), int(points_line[i + 2])
                    points_2d.append((x, y, point3d_id))
                except Exception as e:
                    raise ValueError(f"Error parsing POINTS2D[] on line {idx}: {line}\n{e}")
            idx += 1
    return cameras, images

def create_camera_views_from_custom_data(cameras, images):
    camera_views = []
    for image_id, image_info in images.items():
        camera_id = image_info['camera_id']
        camera_params = cameras[camera_id]
        qw, qx, qy, qz = image_info['quaternion']
        tx, ty, tz = image_info['translation']
        R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        T_cw = np.array([tx, ty, tz])
        R_wc = np.linalg.inv(R_cw)
        camera_center = -R_wc @ T_cw
        print(f"[DEBUG] Camera {image_id}: Quaternion {qw, qx, qy, qz}")
        print(f"[DEBUG] Camera {image_id}: Translation {tx, ty, tz}")
        print(f"[DEBUG] Camera {image_id}: Camera center (world coords) {camera_center}")
        fx = camera_params['params'][0]
        fy = camera_params['params'][1]
        width = camera_params['width']
        height = camera_params['height']
        FoVx = 2 * np.arctan(width / (2 * fx))
        FoVy = 2 * np.arctan(height / (2 * fy))
        view = Camera(
            resolution=(width, height),
            colmap_id=camera_id,
            R=R_wc,
            T=camera_center,
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name=image_info['name'],
            uid=image_id
        )
        camera_views.append(view)
    return camera_views

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        try:
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
            if rendering is None or rendering.sum() == 0:
                print(f"Warning: Rendering result is empty for view {view.image_name}")
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name))
        except Exception as e:
            print(f"Error rendering view {view.image_name}: {e}")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                custom_cameras_path=None, custom_images_path=None,
                skip_train: bool = False, skip_test: bool = False,
                separate_sh: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1]  
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if custom_cameras_path and custom_images_path:
            cameras, images = load_custom_cameras_and_images(custom_cameras_path, custom_images_path)
            custom_views = create_camera_views_from_custom_data(cameras, images)
            render_set(
                dataset.model_path,
                "novel_view",
                scene.loaded_iter,
                custom_views,
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                separate_sh
            )
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_cameras", type=str, help="Path to custom cameras.txt")
    parser.add_argument("--custom_images", type=str, help="Path to custom images.txt")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        custom_cameras_path=args.custom_cameras,
        custom_images_path=args.custom_images,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        separate_sh=SPARSE_ADAM_AVAILABLE
    )