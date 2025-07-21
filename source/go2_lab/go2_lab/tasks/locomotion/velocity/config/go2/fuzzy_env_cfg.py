from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from .rough_env_cfg import Go2RoughEnvCfg

import go2_lab.tasks.locomotion.velocity.mdp as mdp

@configclass
class Go2FuzzyEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        # 부모 클래스 초기화
        super().__post_init__()

        # 평평한 지형으로 변경
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # 높이 스캐너 비활성화 (평평한 지형에서는 불필요)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # 지형 난이도 커리큘럼 비활성화
        self.curriculum.terrain_levels = None

        # 보상 가중치 조정
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        # 패널티
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.dof_torques_l2.weight = -2e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.base_height.weight = -5.0
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_pos_limits.weight = -10.0
        self.rewards.stand_still.weight = -0.5
        self.rewards.trot_symmetry.weight = 0.5
        """위에 보상/패널티의 가중치는 flat 환경에서 잘 되었던 것"""
        

        # 수정하고 싶은 보상항
        self.rewards.base_height.params["target_height"] = 0.35
        # self.rewards.trot_symmetry.params["weight"] = 0.0
        # pouncing
        # self.rewards.trot_symmetry.params["joint_pairs"] = [
        #     ("FR_hip_joint", "FL_hip_joint"),
        #     ("RL_hip_joint", "RR_hip_joint"),
        #     ("FR_thigh_joint", "FL_thigh_joint"),
        #     ("RL_thigh_joint", "RR_thigh_joint"),
        #     ("FR_calf_joint", "FL_calf_joint"),
        #     ("RL_calf_joint", "RR_calf_joint"),
        # ]

        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 0), lin_vel_y=(0, 0), ang_vel_z=(0, 0), heading=(0, 0)
        )


class Go2FuzzyEnvCfg_PLAY(Go2FuzzyEnvCfg):
    def __post_init__(self) -> None:
        # 부모 클래스 초기화
        super().__post_init__()

        # 플레이 모드를 위한 작은 환경 설정
        self.scene.num_envs = 50  # 환경 수를 50개로 제한
        self.scene.env_spacing = 2.5  # 환경 간 간격 설정

        # 플레이 모드에서는 랜덤화 비활성화
        self.observations.policy.enable_corruption = False

        # 플레이 모드에서는 랜덤한 외부 힘 제거
        self.events.base_external_force_torque = None
        self.events.push_robot = None
