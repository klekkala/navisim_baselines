"""
Navigation Environment Configuration for NaviSim

This configuration defines the scene, robot, and environment parameters.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class NavisimNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for the Navisim navigation environment."""

    # Basic environment settings
    episode_length_s = 20.0  # 20 seconds per episode
    decimation = 2  # Control frequency = 60 Hz / 2 = 30 Hz
    action_scale = 1.0
    
    # Spaces (required by DirectRLEnvCfg)
    action_space = 2  # Differential drive: [left_wheel, right_wheel]
    observation_space = 10  # Placeholder - adjust based on your sensors
    state_space = 0  # No separate critic state
    
    # Legacy fields (for compatibility)
    num_actions = 2  # Differential drive: [left_wheel, right_wheel]
    num_observations = 10  # Placeholder - adjust based on your sensors
    num_states = 0

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,  # 60 Hz physics
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Robot configuration (Jetbot)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/kkandorange/data/isaac-lab/assets/Robots/Jetbot/jetbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),  # Slightly above ground
            joint_pos={
                "left_wheel_joint": 0.0,
                "right_wheel_joint": 0.0,
            },
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_joint"],
                velocity_limit=100.0,
                effort_limit=10.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )

    # Camera sensor (Jetbot's first-person view)
    jetbot_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/chassis/rgb_camera/jetbot_camera",
        spawn=None,  # Reference existing camera in USD
        update_period=0.1,  # 10 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )

    # Contact sensor (for collision detection)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # Reward scales (tune these for your task)
    reward_scales = {
        "forward": 1.0,          # Reward for moving forward
        "angular": -0.05,        # Penalty for turning
        "collision": -10.0,      # Penalty for collision
        "time_penalty": -0.01,   # Small time penalty
    }

    # Navigation parameters
    goal_distance_threshold = 0.5  # meters
    max_linear_velocity = 0.5      # m/s
    max_angular_velocity = 1.0     # rad/s

    def __post_init__(self):
        """Post-initialization processing."""
        # Set viewer camera
        self.viewer = ViewerCfg(
            eye=(5.0, 5.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
        )
        
        # Decimation
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation