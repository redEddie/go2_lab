"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""
Launch Isaac Sim Simulator first.


Example usage:

.. code-block:: bash

    # play arguments
    --model <model_path> --num_envs 100

    # play(flat)
    python scripts/rsl_rl/play.py --task=Template-Isaac-Velocity-Flat-Go2-Play-v0
    
    # play(rough)
    python scripts/rsl_rl/play.py --task=Template-Isaac-Velocity-Rough-Go2-Play-v0

    # play(climb)
    python scripts/rsl_rl/play.py --task=Template-Isaac-Velocity-Climb-Go2-Play-v0

    # play(malfunction)
    python scripts/rsl_rl/play.py --task=Template-Isaac-Velocity-Malfunction-Go2-Play-v0

    # play(fuzzy)
    python scripts/rsl_rl/play.py --task=Template-Isaac-Velocity-Fuzzy-Go2-Play-v0
    
    # tensorboard
    python -m tensorboard.main --logdir logs/rsl_rl

"""

import argparse
import numpy as np
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.envs.mdp.observations import SceneEntityCfg

# Import extensions to set up environment tasks
import go2_lab.tasks  # noqa: F401


def print_observation(obs):
    """Print observation in a readable format."""
    # 첫 번째 환경(0번째 env)의 관측값만 선택
    obs = obs[0] if obs.ndim > 1 else obs
    
    # 관절 이름 정의
    joint_names = [
        "FL_hip", "FR_hip", "RL_hip", "RR_hip",
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
        "FL_calf", "FR_calf", "RL_calf", "RR_calf"
    ]
    
    print("\n=== Observation (Env 0) ===")
    print(f"base_lin_vel: {[f'{x:.2f}' for x in obs[0:3]]} - 로봇의 선형 속도(vx, vy, vz)")
    print(f"base_ang_vel: {[f'{x:.2f}' for x in obs[3:6]]} - 로봇의 각속도(wx, wy, wz)")
    print(f"projected_gravity: {[f'{x:.2f}' for x in obs[6:9]]} - 중력 벡터(gx, gy, gz)")
    print(f"velocity_commands: {[f'{x:.2f}' for x in obs[9:12]]} - 속도 명령 (vx, vy, wz)")
    print(f"joint_pos: {[f'{joint_names[i]}: {x:>5.2f}' for i, x in enumerate(obs[12:24])]}")
    print(f"joint_vel: {[f'{joint_names[i]}: {x:>5.2f}' for i, x in enumerate(obs[24:36])]}")
    print(f"commands : {[f'{joint_names[i]}: {x:>5.2f}' for i, x in enumerate(obs[36:48])]}")


def get_robot_state(env):
    """로봇의 추가 상태 정보를 얻어서 출력합니다."""
    try:
        # 로봇 설정 가져오기
        robot_cfg = SceneEntityCfg("robot")
        robot = env.unwrapped.scene[robot_cfg.name]
        
        # 10개 환경의 로봇 높이를 한 줄에 출력
        heights = [f"{robot.data.root_pos_w[i, 2]:.3f}" for i in range(10)]
        print(f"로봇 높이 (env 0-9): {', '.join(heights)}")
            
    except Exception as e:
        print(f"[ERROR] 로봇 상태 정보 가져오기 실패: {e}")


def convert_gym_to_isaac_actions(actions):
    """
    Gym 환경에서 학습된 actions를 IsaacLab 환경에 맞게 순서를 변환합니다.
    
    입력: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    출력: [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11]
    
    추가적으로 0번과 1번 인덱스의 부호를 반전시킵니다.
    
    Parameters:
        actions (torch.Tensor or numpy.ndarray): 원본 gym 액션 배열
        
    Returns:
        변환된 IsaacLab 액션 배열 (입력과 동일한 타입)
    """
    import torch
    
    # 입력 타입 확인 및 보존
    is_torch = isinstance(actions, torch.Tensor)
    original_shape = actions.shape
    
    # numpy로 변환하여 작업
    if is_torch:
        actions_np = actions.clone().cpu().numpy()
    else:
        actions_np = actions.copy()
    
    # 1차원으로 변환
    if len(original_shape) > 1:
        actions_np = actions_np.reshape(-1, 12)
    
    # 인덱스 교환 (1↔2, 5↔6, 9↔10)
    for i in range(actions_np.shape[0] if len(original_shape) > 1 else 1):
        idx = i if len(original_shape) > 1 else slice(None)
        # 1번과 2번 교환
        actions_np[idx][1], actions_np[idx][2] = actions_np[idx][2], actions_np[idx][1]
        # 5번과 6번 교환
        actions_np[idx][5], actions_np[idx][6] = actions_np[idx][6], actions_np[idx][5]
        # 9번과 10번 교환
        actions_np[idx][9], actions_np[idx][10] = actions_np[idx][10], actions_np[idx][9]
    
    # 원래 형태로 복원
    if len(original_shape) > 1:
        actions_np = actions_np.reshape(original_shape)
    
    # 원래 타입으로 반환
    if is_torch:
        return torch.from_numpy(actions_np).to(actions.device)
    else:
        return actions_np


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # resume_path = os.path.join("navaneet", "model_final.pt")
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    all_actions = []
    data_dir = os.path.join(os.path.dirname(resume_path), "play_data")
    os.makedirs(data_dir, exist_ok=True)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            # print_observation(obs)
            # get_robot_state(env)
            all_actions.append(actions[0].cpu().numpy())
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    np.savez(
        os.path.join(data_dir, "actions.npz"),
        actions=np.array(all_actions),
    )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
