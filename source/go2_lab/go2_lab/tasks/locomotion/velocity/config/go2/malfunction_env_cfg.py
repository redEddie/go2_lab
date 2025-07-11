from isaaclab.utils import configclass

from go2_lab.tasks.locomotion.velocity.go2_velocity_env_cfg import (
    Go2LocomotionVelocityRoughEnvCfg,
    Go2LocomotionVelocityMalfunctionEnvCfg,
)

from unitree import UNITREE_GO2_CFG


@configclass
class Go2MalfunctionEnvCfg(Go2LocomotionVelocityMalfunctionEnvCfg):
    def __post_init__(self):
        # 부모 클래스 초기화
        super().__post_init__()
        # 로봇을 Go2로 변경
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.rewards.trot_symmetry = None


@configclass
class Go2MalfunctionEnvCfg_PLAY(Go2MalfunctionEnvCfg):
    def __post_init__(self):
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
        # self.scene.height_scanner.debug_vis = True
        # 랜덤한 외부 힘 제거
        self.events.base_external_force_torque = None
        self.events.push_robot = None
