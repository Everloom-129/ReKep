import torch
import numpy as np
import json
import os
import sys
import pdb 

import argparse
from rekep.environment import R2D2Env
from rekep.ik_solver import FrankaIKSolver
from rekep.subgoal_solver import SubgoalSolver
from rekep.path_solver import PathSolver
import rekep.transform_utils as T
from rekep.visualizer import Visualizer

from rekep.utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
    timer_decorator,
)

'''
metadata.json
{
    "init_keypoint_positions": [
        [-0.1457058783982955, -0.47766187961876, 0.98],
        [-0.0144477656708159, 0.012521396914707113, 0.745],
        [0.14099338570298237, 0.5722672713826932, 1.283],
        [0.2693722882157947, -0.3018593983523729, 1.047],
        [0.43524427390119413, -0.04595746991503292, 0.6970000000000001]
    ],
    "num_keypoints": 5,
    "num_stages": 4,
    "grasp_keypoints": [1, -1, 2, -1],
    "release_keypoints": [-1, 1, -1, 2]
}
'''
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="xFormers is not available")


@timer_decorator
class MainR2D2:
    def __init__(self, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

        # self.vision = R2D2Vision(visualize=self.visualize)

        self.env = R2D2Env(global_config['env'])
        
        ik_solver = FrankaIKSolver(
            reset_joint_pos= self.env.reset_joint_pos,
            world2robot_homo= self.env.world2robot_homo,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        self.visualizer = Visualizer(global_config['visualizer'])

        if visualize:
            self.visualizer = Visualizer(global_config['visualizer'])
            self.data_path = "/home/franka/R2D2_3dhat/images/current_images"

    @timer_decorator
    def perform_task(self, instruction, obj_list=None, rekep_program_dir=None):
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        # obj_list = ['scissors']
        
        data_path = "/home/franka/R2D2_3dhat/images/current_images"
        
        if rekep_program_dir is None:
            pass
            # realworld_rekep_program_dir = self.vision.perform_task(instruction, obj_list, data_path, 3)
        else:
            realworld_rekep_program_dir = rekep_program_dir
        # ====================================
        self._execute(realworld_rekep_program_dir)

    @timer_decorator
    def _execute(self, rekep_program_dir):
        # Load program info and constraints
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        
        # Register initial keypoints
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        
        # Load all stage constraints
        self.constraint_fns = self._load_constraints(rekep_program_dir)
        
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # Generate action sequences for all stages
        self.all_actions = []
        # pdb.set_trace()
        # Process each stage sequentially
        if 1:
            # Read stage from robot state file
            with open('./robot_state.json', 'r') as f:
                robot_state = json.load(f)
                stage = robot_state.get('rekep_stage', 1)  # !!! @Tianyou Default to stage 1 if not found
            # store robot state in rekep_program_dir
            with open(os.path.join(rekep_program_dir, f'robot_state_{stage}.json'), 'w') as f:
                json.dump(robot_state, f, indent=4)
                 # Get current state
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)
            self.curr_ee_pose = self.env.get_ee_pose()  # TODO check, may be constant? 
            self.curr_joint_pos = self.env.get_arm_joint_positions() 
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size']) # TODO ???
            self.collision_points = self.env.get_collision_points()
            
            # stage = input(f"Enter stage number (1-{self.program_info['num_stages']}): ")
            stage = int(stage)
            if stage > self.program_info['num_stages']:
                print(f"{bcolors.FAIL}Stage {stage} is out of bounds, skipping\n{bcolors.ENDC}")
                return
            self._update_stage(stage)
            # if self.stage > 1:
            #     path_constraints = self.constraint_fns[self.stage]['path']
            #     for constraints in path_constraints:
            #         violation = constraints(self.keypoints[0], self.keypoints[1:])
            #         if violation > self.config['constraint_tolerance']:
            #             # backtrack = True
            #             print(f"\n\n\nConstraint violation: {violation}\n\n\n")
            #             break
       
            # Generate actions for this stage
            next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
            next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
            self.first_iter = False


            # pdb.set_trace()
            # Add gripper actions based on stage type
            # True or False from metadata.json
            # if self.is_grasp_stage: 
            #     next_path[-1, 7] = self.env.get_gripper_close_action() # Todo aliagn shape?
                
            # elif self.is_release_stage:
            #     next_path[-1, 7] = self.env.get_gripper_open_action() 
                
            self.all_actions.append(next_path)

        # Combine all action sequences
        combined_actions = np.concatenate(self.all_actions, axis=0)

        if self.stage <= self.program_info['num_stages']: 
            # self.env.sleep(2.0)
            save_path = os.path.join('./outputs', 'action.json') # TODO: save by stage?
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({"ee_action_seq": combined_actions.tolist(), "stage": stage}, f, indent=4)

            with open(os.path.join(rekep_program_dir, f'stage{stage}_actions.json'), 'w') as f:
                json.dump({"ee_action_seq": combined_actions.tolist(), "stage": stage}, f, indent=4)
            print(f"{bcolors.OKGREEN}Actions saved to {save_path}\n and added to {rekep_program_dir}\n{bcolors.ENDC}")
            return
        else:
            print(f"{bcolors.OKGREEN}All stages completed\n\n{bcolors.ENDC}")
            # TODO: return to reset pose?
            # self.env.reset()
            return  
            
            return
    def _load_constraints(self, rekep_program_dir):
        """Helper to load all stage constraints"""
        constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            constraint_fns[stage] = stage_dict
        return constraint_fns

    @timer_decorator
    def _get_next_subgoal(self, from_scratch):
        # pdb.set_trace()
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        # if self.is_grasp_stage:
        #     subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose, self.data_path)
        return subgoal_pose

    @timer_decorator
    def _get_next_path(self, next_subgoal, from_scratch):
        # pdb.set_trace()
        print(f"Start solving path from {self.curr_ee_pose} to {next_subgoal}")
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        
        if self.visualize:
            self.visualizer.visualize_path(processed_path, self.data_path)
            
        return processed_path
    
    # TODO: check action sequence
    @timer_decorator
    def _process_path(self, path):
        # pdb.set_trace()
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        # pdb.set_trace()
        print("Grasp action")
    
    def _execute_release_action(self):
        print("Release action")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=False, help='Instruction for the task')
    parser.add_argument('--rekep_program_dir', type=str, required=False, help='keypoint constrain proposed folder')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    # args.instruction = "Put the green package in the drawer, the robot is already grasping the package and the package is already aligned with the drawer opening."
    # args.obj_list = ['cloth']

    vlm_query_dir = "/home/franka/R2D2_3dhat/ReKep/vlm_query/"

    vlm_dirs = [os.path.join(vlm_query_dir, d) for d in os.listdir(vlm_query_dir) 
                if os.path.isdir(os.path.join(vlm_query_dir, d))]
    
    if vlm_dirs:
        newest_rekep_dir = max(vlm_dirs, key=os.path.getmtime)
        print(f"\033[92mUsing most recent directory: {newest_rekep_dir}\033[0m")
    else:
        print("No directories found under vlm_query")
        sys.exit(1)

    main = MainR2D2(visualize=args.visualize)
    main.perform_task(instruction=args.instruction, rekep_program_dir=newest_rekep_dir)
