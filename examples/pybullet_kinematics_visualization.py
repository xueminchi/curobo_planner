#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Example visualizing robot kinematics using PyBullet"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_path, get_assets_path, join_path, load_yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using numpy for joint angles generation.")


class PyBulletKinematicsVisualizer:
    """PyBullet visualizer for robot kinematics"""
    
    def __init__(self, robot_config_name="franka.yml", gui=True):
        """Initialize the PyBullet visualizer
        
        Args:
            robot_config_name: Name of the robot configuration file
            gui: Whether to use GUI mode
        """
        self.gui = gui
        self.robot_config_name = robot_config_name
        self.robot_id = None
        self.joint_indices = []
        self.joint_names = []
        
        # Initialize PyBullet
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Load robot configuration
        self._load_robot_config()
        
        # Setup PyBullet environment
        self._setup_environment()
        
        # Load robot model
        self._load_robot_model()
    
    def _load_robot_config(self):
        """Load robot configuration from YAML file"""
        config_file = load_yaml(join_path(get_robot_path(), self.robot_config_name))
        self.robot_cfg_dict = config_file["robot_cfg"]
        
        # Get URDF path
        urdf_path = self.robot_cfg_dict["kinematics"]["urdf_path"]
        asset_root = self.robot_cfg_dict["kinematics"]["asset_root_path"]
        
        # Construct full path - URDF and assets are in the assets directory
        self.urdf_file = join_path(get_assets_path(), urdf_path)
        self.asset_root_path = join_path(get_assets_path(), asset_root)
        
        # Get joint names and limits
        self.joint_names = self.robot_cfg_dict["kinematics"]["cspace"]["joint_names"]
        self.retract_config = self.robot_cfg_dict["kinematics"]["cspace"]["retract_config"]
        
        print(f"Loading robot from: {self.urdf_file}")
        print(f"Asset root path: {self.asset_root_path}")
        print(f"Joint names: {self.joint_names}")
    
    def _setup_environment(self):
        """Setup PyBullet environment"""
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        # Add data path for PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
    
    def _load_robot_model(self):
        """Load robot model from URDF"""
        # Change to asset root directory for relative mesh paths
        original_dir = os.getcwd()
        try:
            os.chdir(self.asset_root_path)
            
            # Load robot URDF
            start_pos = [0, 0, 0]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            self.robot_id = p.loadURDF(
                self.urdf_file,
                start_pos,
                start_orientation,
                useFixedBase=True
            )
            
        finally:
            os.chdir(original_dir)
        
        if self.robot_id is None:
            raise RuntimeError("Failed to load robot URDF")
        
        # Get joint information
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot loaded with {num_joints} joints")
        
        # Map joint names to indices
        self.joint_indices = []
        for joint_name in self.joint_names:
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[1].decode('utf-8') == joint_name:
                    self.joint_indices.append(i)
                    break
        
        print(f"Mapped {len(self.joint_indices)} joints: {self.joint_indices}")
        
        # Print joint information
        print("\nJoint Information:")
        for i, idx in enumerate(self.joint_indices):
            joint_info = p.getJointInfo(self.robot_id, idx)
            print(f"  {i}: {joint_info[1].decode('utf-8')} (index {idx})")
    
    def set_joint_angles(self, joint_angles):
        """Set robot joint angles
        
        Args:
            joint_angles: List or array of joint angles (should match number of controlled joints)
        """
        if len(joint_angles) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} joint angles, got {len(joint_angles)}")
        
        for i, (joint_idx, angle) in enumerate(zip(self.joint_indices, joint_angles)):
            p.resetJointState(self.robot_id, joint_idx, angle)
    
    def get_joint_angles(self):
        """Get current joint angles"""
        joint_angles = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_angles.append(joint_state[0])
        return joint_angles
    
    def get_end_effector_pose(self):
        """Get end effector pose"""
        # Get link state for end effector
        ee_link_name = self.robot_cfg_dict["kinematics"]["ee_link"]
        
        # Find end effector link index
        num_joints = p.getNumJoints(self.robot_id)
        ee_link_idx = -1
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == ee_link_name:
                ee_link_idx = i
                break
        
        if ee_link_idx == -1:
            # If not found as child link, try as parent
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[1].decode('utf-8') == ee_link_name:
                    ee_link_idx = i
                    break
        
        if ee_link_idx != -1:
            link_state = p.getLinkState(self.robot_id, ee_link_idx)
            return link_state[0], link_state[1]  # position, orientation
        else:
            print(f"Warning: End effector link '{ee_link_name}' not found")
            return None, None
    
    def visualize_random_configurations(self, num_configs=10, delay=1.0):
        """Visualize random robot configurations"""
        print(f"\nVisualizing {num_configs} random configurations...")
        
        for i in range(num_configs):
            # Generate random joint angles
            joint_angles = []
            for joint_idx in self.joint_indices:
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                
                # Use joint limits if available, otherwise use reasonable defaults
                if lower_limit < upper_limit:
                    angle = np.random.uniform(lower_limit, upper_limit)
                else:
                    angle = np.random.uniform(-np.pi, np.pi)
                
                joint_angles.append(angle)
            
            # Set joint angles
            self.set_joint_angles(joint_angles)
            
            # Get end effector pose
            ee_pos, ee_orn = self.get_end_effector_pose()
            if ee_pos is not None:
                print(f"Config {i+1}: EE Position = {ee_pos[:3]}")
            
            # Step simulation for visualization
            p.stepSimulation()
            
            if self.gui:
                time.sleep(delay)
    
    def visualize_trajectory(self, joint_trajectories, delay=0.1):
        """Visualize a trajectory of joint configurations
        
        Args:
            joint_trajectories: List of joint angle configurations
            delay: Delay between configurations in seconds
        """
        print(f"\nVisualizing trajectory with {len(joint_trajectories)} waypoints...")
        
        for i, joint_angles in enumerate(joint_trajectories):
            self.set_joint_angles(joint_angles)
            
            # Get end effector pose
            ee_pos, ee_orn = self.get_end_effector_pose()
            if ee_pos is not None:
                print(f"Waypoint {i+1}: EE Position = {ee_pos[:3]}")
            
            # Step simulation
            p.stepSimulation()
            
            if self.gui:
                time.sleep(delay)
    
    def reset_to_retract_config(self):
        """Reset robot to retract configuration"""
        print(f"\nResetting to retract configuration: {self.retract_config}")
        self.set_joint_angles(self.retract_config)
        p.stepSimulation()
    
    def _extend_joint_configuration(self, curobo_joints, default_gripper_values=[0.04, 0.04]):
        """Extend CuRobo joint configuration to match PyBullet expectations
        
        Args:
            curobo_joints: Joint angles from CuRobo (usually 7 for arm)
            default_gripper_values: Default values for gripper joints
            
        Returns:
            Extended joint configuration matching PyBullet joint count
        """
        extended_config = list(curobo_joints)
        
        # Add gripper joint values if needed
        if len(extended_config) < len(self.joint_names):
            # Add default gripper values
            extended_config.extend(default_gripper_values)
        
        return extended_config

    def disconnect(self):
        """Disconnect from PyBullet"""
        p.disconnect()


def demo_pybullet_visualization():
    """Demonstrate PyBullet visualization"""
    print("=== PyBullet Kinematics Visualization Demo ===")
    
    # Create visualizer
    visualizer = PyBulletKinematicsVisualizer(gui=True)
    
    try:
        # Reset to retract configuration
        visualizer.reset_to_retract_config()
        time.sleep(2)
        
        # Visualize random configurations
        visualizer.visualize_random_configurations(num_configs=5, delay=2.0)
        
        # Create a simple trajectory (sine wave motion for first joint)
        trajectory = []
        for t in np.linspace(0, 2*np.pi, 50):
            config = visualizer.retract_config.copy()
            config[0] = np.sin(t) * 0.5  # Vary first joint
            config[1] = np.sin(t * 0.5) * 0.3  # Vary second joint slower
            trajectory.append(config)
        
        # Visualize trajectory
        visualizer.visualize_trajectory(trajectory, delay=0.1)
        
        # Reset to retract configuration
        visualizer.reset_to_retract_config()
        
        print("\nDemo completed. Press any key to exit...")
        input()
        
    finally:
        visualizer.disconnect()


def demo_compare_with_curobo():
    """Compare PyBullet visualization with CuRobo computation"""
    print("=== Comparing PyBullet with CuRobo ===")
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot run CuRobo comparison.")
        return
    
    # Setup CuRobo
    study_flag = False

    tensor_args = TensorDeviceType()
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))
    robot_cfg = RobotConfig.from_dict(config_file["robot_cfg"], tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    if study_flag:
        _state = kin_model.get_state(torch.rand((1, kin_model.get_dof()),**(tensor_args.as_torch_dict())))
        # extract info from _state
        print("ee_position",_state.ee_position)
        print("ee_quaternion",_state.ee_quaternion)
        print("links_position",_state.links_position)
        print("links_quaternion",_state.links_quaternion)
        print("link_spheres_tensor",_state.link_spheres_tensor)

 
    # Create PyBullet visualizer
    visualizer = PyBulletKinematicsVisualizer(gui=True)
    
    try:
        # Generate random joint configurations
        q = torch.rand((5, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
        
        print(f"Generated {q.shape[0]} random configurations with {q.shape[1]} joints")
        print(f"PyBullet expects {len(visualizer.joint_names)} joints")
        
        for i in range(q.shape[0]):
            curobo_joint_angles = q[i].cpu().numpy()
            
            # Extend configuration to match PyBullet joint count
            extended_joint_angles = visualizer._extend_joint_configuration(curobo_joint_angles)
            
            # Set in PyBullet
            visualizer.set_joint_angles(extended_joint_angles)
            
            # Get CuRobo forward kinematics
            curobo_state = kin_model.get_state(q[i:i+1])
            curobo_ee_pos = curobo_state.ee_position[0].cpu().numpy()
            curobo_ee_quat = curobo_state.ee_quaternion[0].cpu().numpy()
            
            # Get PyBullet end effector pose
            pb_ee_pos, pb_ee_quat = visualizer.get_end_effector_pose()
            
            print(f"\nConfiguration {i+1}:")
            print(f"  CuRobo joints ({len(curobo_joint_angles)}): {curobo_joint_angles}")
            print(f"  Extended joints ({len(extended_joint_angles)}): {extended_joint_angles}")
            print(f"  CuRobo EE pos: {curobo_ee_pos}")
            print(f"  PyBullet EE pos: {pb_ee_pos}")
            
            if pb_ee_pos is not None:
                pos_diff = np.linalg.norm(np.array(pb_ee_pos) - curobo_ee_pos)
                print(f"  Position difference: {pos_diff:.4f}")
            
            p.stepSimulation()
            time.sleep(2)
        
        print("\nComparison completed. Press any key to exit...")
        input()
        
    finally:
        visualizer.disconnect()


if __name__ == "__main__":
    # Run basic visualization demo
    # demo_pybullet_visualization()
    
    # Uncomment to run comparison with CuRobo
    demo_compare_with_curobo() 