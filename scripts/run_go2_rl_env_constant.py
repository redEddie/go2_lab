# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""이 스크립트는 Go2 로봇을 상수 속도로 제어합니다.

.. code-block:: bash

    # 사용법
    ./isaaclab.sh -p scripts/tutorials/03_envs/run_go2_rl_env_constant.py --checkpoint h1_policy.pt

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import glob

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run Go2 RL environment with constant velocity.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--checkpoint",
    type=str,
    help="Path to model checkpoint exported as jit. If not provided, will use the latest policy.pt from logs.",
    required=False,
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import io
import torch
import omni
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import ObservationTermCfg as ObsTerm

# 현재 workspace의 설정을 사용
from go2_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg import (
    Go2FlatEnvCfg_PLAY,
)


def constant_commands(env):
    """상수 속도 명령을 반환합니다."""
    return torch.tensor([[0.0, 0.0, 0.0]], device=env.device)


@configclass
class Go2ConstantEnvCfg_PLAY(Go2FlatEnvCfg_PLAY):
    """상수 속도 명령을 포함하는 환경 설정"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # velocity_commands observation을 상수 속도로 변경
        self.observations.policy.velocity_commands = ObsTerm(func=constant_commands)


def main():
    # 체크포인트 경로 설정
    policy_path = "pretrained/policy_15k.pt"
    if not os.path.exists(policy_path):
        print(f"[ERROR] Policy file not found at {policy_path}")
        return
    print(f"[INFO] Using policy from: {policy_path}")

    # load the trained jit policy
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file)

    device = torch.device("cuda")
    policy = policy.to(device)
    args_cli.device = str(device)

    # configure environment
    env_cfg = Go2ConstantEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False
    
    # 시뮬레이션 성능 최적화 설정
    env_cfg.sim.dt = 0.02  # 50Hz 렌더링 (1/50 = 0.02)
    env_cfg.sim.substeps = 10  # (50 * 10 = 500Hz)
    
    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # reset environment
    obs, _ = env.reset()

    # 시간 추적을 위한 변수 추가
    sim_time = 0.0
    sim_dt = env.sim.get_physics_dt()

    count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # if count % 300 == 0:
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")

            # 1초마다 상태 출력
            if sim_time % 1.0 < sim_dt:
                print(f"Time: {sim_time:.2f}s")

            # infer action
            policy_input = obs["policy"].to(torch.float32)
            action = policy(policy_input)
            action = action.to(env.device)

            # step environment
            obs, rew, terminated, truncated, info = env.step(action)

            # print some observation info
            # 0.5초마다 출력    
            if sim_time % 0.5 < sim_dt:
                base_vel = obs['policy'][0][:3].cpu().numpy()
                print(f"[Env 0] base_lin_vel: [{base_vel[0]:.2f}, {base_vel[1]:.2f}, {base_vel[2]:.2f}], reward: {rew[0].item():.2f}")

            count += 1
            sim_time += sim_dt

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close() 