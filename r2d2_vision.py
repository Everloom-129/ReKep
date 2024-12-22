"""
Rekep Vision Module for Real-world Manipulation
- By Tony Wang, University of Pennsylvania

This module provides vision capabilities for the Rekep robotic manipulation system.

1. SAM2 for object segmentation
2. Grounding DINO for object detection and recognition 
3. RealSense depth camera for RGB-D perception
4. Keypoint proposal generation for manipulation planning
5. Constraint generation for motion planning

The vision system processes RGB-D images to:
- Detect and segment objects in the scene
- Generate keypoints&metadata for VLM / low level control 
- Create spatial constraints for motion planning
- Provide visual feedback for manipulation tasks

Note: This module run once in each task of R2D2 experiment.
Reset after success or failure.
"""

import os
import torch
import numpy as np
import argparse
import pyrealsense2 as rs
import supervision as sv
import cv2
import json
from rekep.keypoint_proposal import KeypointProposer
from rekep.constraint_generation import ConstraintGenerator
from rekep.utils import (
    bcolors,
    get_config,
    timer_decorator,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

SAM2_CKPT =  '/home/franka/ACReKep/model_ckpt/sam2_hiera_small.pt'
SMA2_CFIG = "./configs/sam2/sam2_hiera_s.yaml"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from rekep.perception.realsense import initialize_realsense
from rekep.perception.gdino import GroundingDINO

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper



@timer_decorator
class R2D2Vision:
    @timer_decorator
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.visualize = visualize
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])

    @timer_decorator
    def load_camera_intrinsics(self, file_path=None):
        # D435i default 
        class RS_Intrinsics:
            def __init__(self):
                self.fx = 386.738  # focal length x
                self.fy = 386.738  # focal length y
                self.ppx = 319.5   # principal point x
                self.ppy = 239.5   # principal point y
        class ZED_Intrinsics:
            def __init__(self):
                self.fx = 527.53936768  # focal length x
                self.fy = 527.53936768  # focal length y
                self.ppx = 646.46374512  # principal point x
                self.ppy = 353.03808594  # principal point y
                
        # Use fixed camera intrinsics
        if file_path is not None:
            with open(file_path, 'r') as file:
                data = json.load(file)
            intrinsics = data['camera']['fixed']['intrinsics']
            depth_scale = data['camera']['fixed']['depth_scale']
        else:
            intrinsics = ZED_Intrinsics()
            depth_scale = 0.001  # Default depth scale (1mm)

        return intrinsics, depth_scale

    @timer_decorator
    def depth_to_pointcloud(self, depth):
        intrinsics, depth_scale = self.load_camera_intrinsics()

        height, width = depth.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        
        # 创建一个与原图像大小相同的空点云
        points = np.zeros((height * width, 3))
        
        # 只处理非零点
        valid_mask = depth > 0
        
        # 计算有效点的3D坐标
        x = (u[valid_mask].flatten() - intrinsics.ppx) / intrinsics.fx
        y = (v[valid_mask].flatten() - intrinsics.ppy) / intrinsics.fy
        z = depth[valid_mask].flatten() * depth_scale
        
        x = np.multiply(x, z)
        y = np.multiply(y, z)

        # 将有效点放回对应位置
        valid_indices = np.where(valid_mask.flatten())[0]
        points[valid_indices] = np.stack((x, y, z), axis=-1)

        return points  # shape: (height * width, 3)
    
    @timer_decorator
    def perform_task(self, instruction,obj_list, data_path):
        if 1:
            color_path = os.path.join(data_path, 'varied_camera_raw.png')
            depth_path = os.path.join(data_path, 'varied_camera_depth.npy')
        else:
            color_path = os.path.join(data_path, 'fixed_camera_raw.png')
            depth_path = os.path.join(data_path, 'fixed_camera_depth.npy')


        print(f"\033[92mDebug: Looking for files at:\033[0m")
        print(f"\033[92mDebug: Color path: {color_path}\033[0m")
        print(f"\033[92mDebug: Depth path: {depth_path}\033[0m")
        
        bgr = cv2.imread(color_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        print(f"\033[92mDebug: Input image shape: {rgb.shape}\033[0m") # (480, 640, 3)
        print(f"\033[92mDebug: Input depth shape: {depth.shape}\033[0m") # (480, 640)  

        # copy the rgb and depth file to ./data/r2d2_vision/zed
        os.makedirs('./data/r2d2_vision/zed', exist_ok=True)
        cv2.imwrite(f'./data/r2d2_vision/zed/color_{instruction}_{time.strftime("%Y%m%d_%H%M%S")}.png', bgr)
        np.save(f'./data/r2d2_vision/zed/depth_{instruction}_{time.strftime("%Y%m%d_%H%M%S")}.npy', depth)
        
        # TODO: add DINOX mode
        if obj_list:
            print(f"\033[92mDebug: Manual mask mode\033[0m")
            sam_model = build_sam2(SMA2_CFIG, SAM2_CKPT).to(device)
            self.mask_generator = SAM2ImagePredictor(sam_model)
            self.mask_generator.set_image(rgb)

            gdino = GroundingDINO()
            if isinstance(obj_list, str):
                obj_list = obj_list.split(',')  # 如果输入是逗号分隔的字符串
            results = gdino.detect_objects(color_path, obj_list)
            # self._show_objects(rgb, results.objects)
            boxes = []
            for obj in results.objects:
                print(f"class: {obj.category}, conf: {obj.score:.2f}, bbox: {obj.bbox}")
                boxes.append(obj.bbox)
            print(f"\033[92mDebug: obj_list: {obj_list}\033[0m")
            print(f"\033[92mDebug: Boxes: {boxes}\033[0m")
            # SAM
            with torch.no_grad():
                masks, scores, logits = self.mask_generator.predict(box=boxes, multimask_output=False)
                masks = np.stack(masks, axis=0)  # Convert list to 3D array
        else:
            print(f"\033[92mDebug: Auto mask mode\033[0m")
            sam_model = build_sam2(SMA2_CFIG, SAM2_CKPT).to(device)
            self.mask_generator = SAM2AutomaticMaskGenerator(sam_model)
            masks_dict = self.mask_generator.generate(rgb)
            masks = [m['segmentation'] for m in masks_dict]
            masks = np.stack(masks, axis=0)  # Convert list to 3D array

        print(f"\033[92mDebug: Generated {len(masks)} masks\033[0m")
        print(f"\033[92mDebug: masks shape: {masks[0].shape}\033[0m")
        print(f"\033[92mDebug: Type of masks: {type(masks)}\033[0m")

        # Point cloud
        points = self.depth_to_pointcloud(depth)
        # points = depth
        print(f"\033[92mDebug: Generated point cloud with shape: {points.shape}\033[0m")
        # import pdb; pdb.set_trace()
        # ====================================
        # = Keypoint Proposal and Constraint Generation
        # ====================================
        keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, masks)
        print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
        if self.visualize:
            self._show_image(projected_img,rgb)
        metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
        rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
        print(f'{bcolors.HEADER}Constraints generated and saved in {rekep_program_dir}{bcolors.ENDC}')
        return rekep_program_dir
        
    def _show_objects(self, rgb, results):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.axis('on')
        plt.title('Detected Objects')
        for obj in results:
            plt.text(obj.bbox[0], obj.bbox[1], obj.category, color='red', fontsize=12)
            plt.box(obj.bbox)
        plt.savefig('data/gdino_objects.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def _show_image(self, idx_img, rgb, masks = None,bboxes = None):
        # Save the annotated image with keypoints
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(idx_img)
        plt.axis('on')
        plt.title('Annotated Image with Keypoints')
        plt.savefig('data/rekep_with_keypoints.png', bbox_inches='tight', dpi=300)
        plt.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=False, help='Instruction for the task')
    parser.add_argument('--obj_list', type=str, required=False, help='String List of objects to detect')
    parser. add_argument('--data_path', type=str, required=False, help='Path to the directory containing color and depth frames')
    parser.add_argument('--visualize', action='store_true', help='Visualize the keypoints on the image')
    args = parser.parse_args()
    
    if args.instruction is None:
        # args.instruction = "Put down the green package into drawer."
        # args.instruction = "Pour the object in the bowl into the pot."
        args.instruction = "Place the pasta bag into the drawer, the end-effector is already at the drawer's keypoint, the drawer is already aligned with the pasta bag and at the proper height."
        # args.instruction = "Pour the object in the bowl into the pot, the end-effector is already at the bowl's keypoint, the bowl is already aligned with the pot and at the proper height."
    if args.data_path is None:
        args.data_path = "/home/franka/R2D2_3dhat/images/current_images"
    # if args.obj_list is None:
        # args.obj_list = "bowl, pan, robot end_effector"
    main = R2D2Vision(visualize=args.visualize)
    rekep_program_dir = main.perform_task(instruction=args.instruction, obj_list=args.obj_list, data_path=args.data_path)
    print(f"\033[92mDebug: rekep_program_dir: {rekep_program_dir}\033[0m")