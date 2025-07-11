from isaaclab.utils import configclass

from unitree import UNITREE_GO2_CFG
from .rough_env_cfg import Go2RoughEnvCfg
from .climb_terrain import CLIMB_TERRAINS_CFG

import go2_lab.tasks.locomotion.velocity.mdp as mdp
import math


from isaaclab.managers import SceneEntityCfg


@configclass
class Go2ClimbEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        # 부모 클래스 초기화
        super().__post_init__()
        self.scene.terrain.terrain_generator = CLIMB_TERRAINS_CFG  # necessary config for climb task

        self.rewards.track_lin_vel_xy_exp.weight = 2.0  # 직진 명령 추종을 조금 강하게 주어 (넘어지는것or멈추는것) < (명령추종) 으로 만들어야한다.
        self.rewards.track_ang_vel_z_exp.weight = 1.0  # 회전 명령 추종을 강하게 주어 (멈추는 것) < (명령추종) 으로 만들어야한다.

        self.rewards.lin_vel_z_l2 = None  # z축에 대한 패널티를 없애 점프 가능하도록
        self.rewards.flat_orientation_l2 = None

        self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = None

        self.rewards.trot_symmetry = None
        self.rewards.roll_penalty.weight = -0.5  # 몸체 롤링에 대한 패널티

        self.rewards.base_height.params["target_height"] = 0.5

        self.rewards.foot_clearance_dir_penalty = None
        # self.rewards.foot_clearance_dir_penalty.weight = -0.5
        # self.rewards.foot_clearance_dir_penalty.params["raycast_cfg"] = SceneEntityCfg("height_scanner")
        # self.rewards.foot_clearance_dir_penalty.params["contact_cfg"] = SceneEntityCfg("contact_forces", body_names=".*foot")

        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2), lin_vel_y=(-1.2, 1.2), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        )
        # self.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}

class Go2ClimbEnvCfg_PLAY(Go2ClimbEnvCfg):
    def __post_init__(self) -> None:
        # 부모 클래스 초기화
        super().__post_init__()

        # 플레이 모드를 위한 작은 환경 설정
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 로봇을 지형 레벨 대신 그리드에 랜덤하게 배치
        self.scene.terrain.max_init_terrain_level = None
        # 메모리 절약을 위해 지형 수 감소
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # 플레이 모드에서는 랜덤화 비활성화
        self.observations.policy.enable_corruption = False
        # 높이 스캐너 디버그 시각화 활성화
        self.scene.height_scanner.debug_vis = True
        # 랜덤한 외부 힘 제거
        self.events.base_external_force_torque = None
        self.events.push_robot = None


        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(-math.pi, math.pi)
        )
        self.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}