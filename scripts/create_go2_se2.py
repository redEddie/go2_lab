# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
이 스크립트는 Go2 로봇을 위한 평지 환경을 시연합니다.

이 예제에서는 로봇을 제어하기 위해 보행 정책을 사용합니다.
로봇은 일정한 속도로 앞으로 이동하도록 명령받습니다.

.. code-block:: bash

    # 스크립트 실행
    ./isaaclab.sh -p scripts/tutorials/03_envs/create_go2_flat_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="평지 환경에서 Go2 로봇을 위한 튜토리얼")
parser.add_argument("--num_envs", type=int, default=64, help="생성할 환경의 수")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.devices import Se2Keyboard  # 키보드 디바이스 추가

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


##
# Custom observation terms
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


def keyboard_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """키보드로부터 명령을 받아옵니다."""
    if not hasattr(env, "keyboard"):
        env.keyboard = Se2Keyboard(
            v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=3.0
        )
        env.keyboard.reset()
    
    command = env.keyboard.advance()
    return torch.tensor(command, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    height_scanner = None

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """MDP를 위한 액션 명세"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.5, 
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """MDP를 위한 관측 명세"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=constant_commands) # keyboard_commands
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """이벤트 설정"""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class Go2FlatEnvCfg(ManagerBasedEnvCfg):
    """평지 환경에서의 속도 추적 환경 설정"""

    # 장면 설정
    scene: MySceneCfg = MySceneCfg(
        num_envs=args_cli.num_envs, 
        env_spacing=2.5
    )
    # 기본 설정
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """초기화 후 처리"""
        # 일반 설정
        self.decimation = 1
        # 시뮬레이션 설정
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


def main():
    """메인 함수"""
    # 기본 환경 설정
    env_cfg = Go2FlatEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # 레벨 정책 로드
    policy_path = "pretrained/policy_15k.pt"
    # 정책 파일 존재 확인
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"정책 파일 '{policy_path}'이(가) 존재하지 않습니다.")
    file_bytes = read_file(policy_path)
    # 정책 jit 로드
    policy = torch.jit.load(file_bytes).to(env.device).eval()

    # 물리 시뮬레이션
    count = 0
    obs, _ = env.reset()
    
    # 시간 추적을 위한 변수 추가
    sim_dt = env.sim.get_physics_dt()  # 시뮬레이션 시간 간격 가져오기
    
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % (10/sim_dt) == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: 환경 리셋 중...")
            
            # 2초마다 command 출력
            if count % (5/sim_dt) == 0:
                print("키 바인딩:")
                print("====================== ========================= ========================")
                print("명령                    키 (+축)                  키 (-축)")  
                print("====================== ========================= ========================")
                print("x축 이동               숫자패드 8 / 위쪽 화살표    숫자패드 2 / 아래쪽 화살표")
                print("y축 이동               숫자패드 4 / 오른쪽 화살표  숫자패드 6 / 왼쪽 화살표") 
                print("z축 회전               숫자패드 7 / Z             숫자패드 9 / X")
                print("====================== ========================= ========================")
            elif count % (2/sim_dt) == 0:
                print(f"Time: {count*sim_dt:.2f}s")
            
            # 액션 추론
            action = policy(obs["policy"])
            # 환경 스텝
            obs, _ = env.step(action)
            # 카운터 업데이트
            count += 1

    # 환경 종료
    env.close()


if __name__ == "__main__":
    # 메인 함수 실행
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close() 