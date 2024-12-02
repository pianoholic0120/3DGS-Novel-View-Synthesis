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
import shutil

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def move_and_rename_point_cloud(src_path, dest_path):
    """Move and rename point cloud file automatically."""
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")
    makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(src_path, dest_path)
    print(f"Copied and renamed point cloud from {src_path} to {dest_path}")

def read_image_names_in_order(images_path):
    image_names = []
    with open(images_path, 'r') as f:
        while True:
            header_line = f.readline().strip()
            if not header_line or header_line.startswith('#'):
                if not header_line:  
                    break
                continue
            points_line = f.readline().strip()
            if not points_line:
                break
            header_parts = header_line.split()
            if len(header_parts) < 9:
                print(f"Skipping invalid image header: {header_line}")
                continue
            image_name = header_parts[9] if len(header_parts) > 9 else ''
            points2d = []
            point_parts = points_line.split()
            for i in range(0, len(point_parts), 3):
                if i + 2 < len(point_parts):
                    x = float(point_parts[i])
                    y = float(point_parts[i+1])
                    point3d_id = int(point_parts[i+2])
                    points2d.append((x, y, point3d_id))
            if not image_name.endswith('.png'):
                image_name += '.png'
            image_names.append(image_name)
    print(f"Total images found: {len(image_names)}")
    return image_names

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, image_names, output_folder):
    makedirs(output_folder, exist_ok=True)
    if len(views) != len(image_names):
        print("Warning: Number of views and image names do not match!")
        for i, view in enumerate(views):
            print(f"View {i}: {view}")
        for view in views:
            print(f"View image name: {view.image_name}")
    name_to_view = {view.image_name: view for view in views}
    for image_name in tqdm(image_names, total=len(image_names), desc="Rendering progress"):
        if image_name in name_to_view:
            view = name_to_view[image_name]
            rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
            gt = view.original_image[0:3, :, :]
            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]
            torchvision.utils.save_image(rendering, os.path.join(output_folder, image_name))
        else:
            print(f"Warning: No matching view found for image {image_name}")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, separate_sh: bool, images_txt_path: str, output_folder: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        image_names = read_image_names_in_order(images_txt_path)
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, image_names, output_folder)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, image_names, output_folder)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=100000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_folder", required=True)
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Move and rename point cloud file
    point_cloud_src = os.path.join(args.model_path, f"point_cloud/iteration_{args.iteration}", "point_cloud.ply")
    point_cloud_dest = os.path.join(args.source_path, f"sparse/0/points3D.ply") 
    move_and_rename_point_cloud(point_cloud_src, point_cloud_dest)
    images_txt_path = os.path.join(args.source_path, f"sparse/0/images.txt")
    # Render sets
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, images_txt_path, args.output_folder)
    shutil.move(point_cloud_dest, point_cloud_src)