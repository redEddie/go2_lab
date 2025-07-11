from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def stand_still(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward standing still.

    This function rewards the agent for standing still when the command is zero.
    It penalizes joint position deviations from the default position only when the command is below a threshold.

    Args:
        env: The environment instance.
        command_name: The name of the command to check.
        threshold: The threshold below which the command is considered zero.
        asset_cfg: The configuration for the asset.

    Returns:
        A tensor containing the reward for standing still.
    """
    # Get the command
    command = env.command_manager.get_command(command_name)
    # Get the joint positions
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    # Get the default joint positions
    default_joint_pos = env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids]
    # Compute the deviation from default position
    deviation = torch.abs(joint_pos - default_joint_pos)
    # Sum the deviations
    total_deviation = torch.sum(deviation, dim=1)
    # Check if command is below threshold
    is_zero_command = torch.norm(command[:, :2], dim=1) < threshold
    # Apply penalty only when command is zero
    reward = total_deviation * is_zero_command.float()
    # 반환값 검증 및 경고
    if torch.isnan(reward).any():
        print("[WARN] stand_still: NaN detected!", reward)
    if torch.isinf(reward).any():
        print("[WARN] stand_still: Inf detected!", reward)
    if (reward < 0).any():
        print("[WARN] stand_still: Negative value detected!", reward)
    if reward.abs().max() > 1e3:
        print("[WARN] stand_still: Large value detected!", reward.abs().max())
    return reward


def trot_symmetry_exp(
    env: ManagerBasedRLEnv, joint_pairs: list[tuple[str, str]], threshold: float = 0.1, sigma: float = 0.5
) -> torch.Tensor:
    """Exponential reward for symmetry between left and right leg joints.

    Args:
        env: The environment instance.
        joint_pairs: List of (left_joint, right_joint) names.
        threshold: Ignore differences below this value.
        sigma: Spread of the exponential curve.

    Returns:
        Reward tensor (higher is better).
    """
    robot = env.scene["robot"]
    dof_names = robot.data.joint_names
    penalties = []
    for left_joint, right_joint in joint_pairs:
        left_idx = dof_names.index(left_joint)
        right_idx = dof_names.index(right_joint)
        left_pos = robot.data.joint_pos[:, left_idx]
        right_pos = robot.data.joint_pos[:, right_idx]
        diff = torch.abs(left_pos - right_pos)
        excess = torch.clamp(diff - threshold, min=0.0)
        penalties.append(excess)
    mean_excess = torch.sum(torch.stack(penalties), dim=0) / len(joint_pairs)
    reward = torch.exp(- (mean_excess ** 2) / (2 * sigma ** 2))
    # 반환값 검증 및 경고
    if torch.isnan(reward).any():
        print("[WARN] trot_symmetry_exp: NaN detected!", reward)
    if torch.isinf(reward).any():
        print("[WARN] trot_symmetry_exp: Inf detected!", reward)
    if (reward < 0).any():
        print("[WARN] trot_symmetry_exp: Negative value detected!", reward)
    if reward.abs().max() > 1e3:
        print("[WARN] trot_symmetry_exp: Large value detected!", reward.abs().max())
    return reward


def roll_penalty_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only roll (side tilt) using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b: (batch, 3) → [:, :2] 중 [:,1]은 roll 성분
    roll_comp = asset.data.projected_gravity_b[:, 1]
    reward = torch.square(roll_comp)
    # 반환값 검증 및 경고
    if torch.isnan(reward).any():
        print("[WARN] roll_penalty_l2: NaN detected!", reward)
    if torch.isinf(reward).any():
        print("[WARN] roll_penalty_l2: Inf detected!", reward)
    if (reward < 0).any():
        print("[WARN] roll_penalty_l2: Negative value detected!", reward)
    if reward.abs().max() > 1e3:
        print("[WARN] roll_penalty_l2: Large value detected!", reward.abs().max())
    return reward


def foot_clearance_dir_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    desired_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    raycast_cfg: SceneEntityCfg | None = None,
    contact_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    Directional foot clearance penalty. 0 is better.
    “앞으로 달리라는 속도 명령이 있을 때, 로봇 발 바로 아래가 아니라 명령 방향(=앞) 지형의 높이를 기준으로 클리어런스를 평가”해서 리워드에 반영

    1) 명령 방향 계산
    2) 하나의 RayCaster로부터 ray_hits_w, pos_w를 가져와
       명령 방향에 가까운 지형 높이를 가중평균
    3) desired_clearance 이하 거리² 페널티 반환
    """
    if raycast_cfg is None or contact_cfg is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 1) 스윙 레그 판별 (contact forces)
    contact_sensor: ContactSensor = env.scene[contact_cfg.name]
    # (B, n_contacts, 3) -> 배치별 총 접촉력 크기
    foot_indices = [4, 8, 14, 18]
    foot_forces = contact_sensor.data.net_forces_w[:, foot_indices, :]  # (B, 4, 3)
    foot_force_magnitudes = torch.norm(foot_forces, dim=-1)  # (B, 4)
    is_stance = foot_force_magnitudes > 1e-2

    # 2) 명령 방향 계산
    cmd = env.command_manager.get_command(command_name)[:, :2]  # [B,2]
    speed = torch.norm(cmd, dim=1, keepdim=True)               # [B,1]
    cmd_dir = cmd / (speed + 1e-6)                             # [B,2]

    # 3) RayCaster -> 가중평균 지형 높이
    raycast = env.scene[raycast_cfg.name].data
    # raycast.pos_w: (N_sensors, 3), raycast.ray_hits_w: (N_sensors, N_rays, 3)
    origin = raycast.pos_w[0]                     # (3,)
    hits   = raycast.ray_hits_w[0]                # (N_rays, 3)
    raw_dirs = hits - origin.unsqueeze(0)      # (N_rays, 3)
    norms    = torch.norm(raw_dirs, dim=-1, keepdim=True)  # (N_rays,1)
    dirs_xy  = raw_dirs[:, :2] / (norms + 1e-6)            # (N_rays,2)
    hits_z   = hits[:, 2]                                 # (N_rays,)

    # 4) 명령 방향과 레이 방향 유사도로 가중평균 지형 높이
    #    cmd_dir: [B,2], dirs_xy: [R,2] → cos_sim: [B,R]
    cos_sim = torch.clamp(torch.einsum('bi,ri->br', cmd_dir, dirs_xy), min=0.0)
    weighted_z = (cos_sim * hits_z.unsqueeze(0)).sum(dim=1) \
               / (cos_sim.sum(dim=1) + 1e-6)  # [B]

    # 5) 발 높이 가져오기
    robot_asset = env.scene[asset_cfg.name]
    foot_z = robot_asset.data.body_pos_w[:, foot_indices, 2]  # (B, 4)

    # 6) 실제 클리어런스 계산 및 페널티
    clearance = foot_z - weighted_z.unsqueeze(-1)         # [B,4]
    deficit = torch.relu(desired_clearance - clearance)   # [B,4]
    penalty = deficit.square()
    penalty = penalty * (~is_stance).float()              # [B,4]
    penalty = penalty.sum(dim=1)                          # [B,]
    # 7) 반환값 검증 및 경고
    if torch.isnan(penalty).any():
        print("[WARN] foot_clearance_dir_penalty: NaN detected!", penalty)
    if torch.isinf(penalty).any():
        print("[WARN] foot_clearance_dir_penalty: Inf detected!", penalty)
    if (penalty < 0).any():
        print("[WARN] foot_clearance_dir_penalty: Negative value detected!", penalty)
    if penalty.abs().max() > 1e3:
        print("[WARN] foot_clearance_dir_penalty: Large value detected!", penalty.abs().max())

    return penalty