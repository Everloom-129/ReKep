import torch
import numpy as np
import argparse
import cv2
import os
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from utils import (
    bcolors,
    get_config,
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class MainVision:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.visualize = visualize
        # Set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # Initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        self.mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large")
    
    def perform_task(self, instruction, image_path):
        # Load image
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        print(f"Debug: Input image shape: {rgb.shape}")

        # Generate masks
        masks_dict = self.mask_generator.generate(rgb)
        masks = [m['segmentation'] for m in masks_dict]
        print(f"Debug: Generated {len(masks)} masks")
       
        # Since we don't have point cloud data, we can simulate or omit it
        # For now, we'll set points to None
        points = None
        # TODO: Add point cloud data from Dense estimation model 
        # replace env.get_cam_obs() with Mujoco camera observation


        # ====================================
        # = Keypoint Proposal and Constraint Generation
        # ====================================
        keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, masks)
        if self.visualize:
            self._show_image(projected_img)
        metadata = {'init_keypoint_positions': None, 'num_keypoints': 5}
        rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
        print(f'{bcolors.HEADER}Constraints generated and saved in {rekep_program_dir}{bcolors.ENDC}')

    def _show_image(self, image):
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=True, help='Instruction for the task')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input RGB image')
    parser.add_argument('--visualize', action='store_true', help='Visualize the keypoints on the image')
    args = parser.parse_args()

    main = MainVision(visualize=args.visualize)
    main.perform_task(instruction=args.instruction, image_path=args.image_path)
