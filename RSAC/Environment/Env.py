import math
import random
from collections import deque

import airsim
import numpy as np
import torch
import matplotlib.pyplot as plt
from airsim import Vector3r


class Config:
    depth_image_height = 42
    depth_image_width = 42
    max_lidar_distance = 20
    max_drone_speed = 5
    goal_threshold = 2
    max_episode_steps = 1000

# 라이다 데이터를 깊이 이미지로 변환
def lidar_to_depth_image(lidar_points, height=Config.depth_image_height,
                         width=Config.depth_image_width, max_distance=Config.max_lidar_distance):
    """
    라이다 포인트 클라우드를 구형 좌표계 기반 depth 이미지로 변환
    로컬 좌표계 기준 (드론 중심)
    """
    # 빈 depth 이미지 초기화 (최대 거리로)
    depth_image = np.ones((height, width)) * max_distance

    if not lidar_points:
        return depth_image / max_distance  # 정규화된 이미지 반환

    for point in lidar_points:
        # 각 점은 (x, y, z) 튜플입니다
        x, y, z = point

        # 구면 좌표로 변환 (r, azimuth, elevation)
        r = math.sqrt(x * x + y * y + z * z)

        # 범위 내의 포인트만 처리
        if r <= max_distance:
            # 방위각: xy 평면에서의 각도 (-π ~ π)
            azimuth = math.atan2(y, x)

            # 고도각: z와 xy 평면 사이의 각도 (-π/2 ~ π/2)
            elevation = math.asin(z / max(r, 1e-5))

            # 구면 좌표를 이미지 픽셀로 변환
            col = int((azimuth + math.pi) / (2 * math.pi) * width)
            row = int((elevation + math.pi / 2) / math.pi * height)

            # 픽셀 범위 확인
            if 0 <= row < height and 0 <= col < width:
                # 해당 픽셀 위치에 가장 가까운 거리 저장
                depth_image[row, col] = min(depth_image[row, col], r)

    # 깊이 이미지 정규화 (0~1 범위로)
    normalized_depth = depth_image / max_distance

    return normalized_depth

class DroneEnv:
    """
    AirSim의 환경을 시뮬레이션하는 간단한 환경 래퍼.
    실제 구현에서는 AirSim API를 사용하여 대체해야 함.
    """

    def __init__(self):
        # AirSim 클라이언트 초기화
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)


        # 목표 위치 설정 (임의로 설정, 실제 환경에 맞게 수정 필요)
        self.drone_trajectory = deque(maxlen=100)
        self.goal_position = np.array([50.0, 0.0, -10.0])

        self.goal_distance = 0
        self.prev_goal_distance = 0

        # 에피소드 스텝 카운터
        self.steps = 0

    def reset(self):
        # 드론 초기화
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        # 초기 위치로 이동 (홈 위치, 약간의 랜덤성 추가)
        initial_x, initial_y, initial_z = [0, 0, 0]

        self.client.moveToPositionAsync(initial_x, initial_y, initial_z, 5).join()

        # 새로운 목표 위치 (랜덤)
        while True:
            self.goal_position = np.array([
                random.uniform(-18.0, 18.0),
                random.uniform(-18.0, 18.0),
                random.uniform(-18.0, 18.0)
            ])

            self.goal_distance = np.linalg.norm(self.goal_position)
            self.prev_goal_distance = self.goal_distance
            if self.goal_distance >= 15:
                print(f"New goal position: {self.goal_position}")
                break

        # 스텝 카운터 초기화
        self.steps = 0

        # 초기 상태 가져오기
        depth_image, drone_state, position = self._get_state()

        self.drone_trajectory = [position]

        return depth_image, drone_state

    def step(self, action):
        # 상태 및 라이다 데이터 가져오기
        depth_image, state, position = self._get_state()

        dir = state[3:] / np.linalg.norm(state[3:])

        blend_action = dir * Config.max_drone_speed

        action = 0.2 * blend_action + action * 0.8


        # 액션 적용 (vx, vy, vz 속도)
        vx, vy, vz = action

        # AirSim에서는 NED 좌표계 사용
        self.client.moveByVelocityAsync(
            float(vx),
            float(vy),
            float(vz),  # AirSim에서 z는 아래 방향이 양수
            1.0  # 1초 동안 속도 유지
        )



        self.drone_trajectory.append(position)

        # 보상과 종료 여부 계산
        reward, done, info = self._compute_reward(depth_image, state, position)

        state[3:] /= np.linalg.norm(state[3:])

        # 스텝 카운터 증가
        self.steps += 1

        # self.visualize_3d_lidar(depth_image, show=True)

        # 최대 스텝 수 초과 시 종료
        if self.steps >= Config.max_episode_steps:
            print(f"timeout: {position}")
            done = True
            info['timeout'] = True

        return depth_image, state, reward, done, info

    def _compute_reward(self, depth_image, drone_state, position):
        # 현재 위치와 목표 위치 거리
        current_goal_distance = np.linalg.norm(drone_state[3:6])

        # 충돌 정보
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided

        # 최소 깊이 계산 (가장 가까운 장애물 거리)
        min_depth = np.min(depth_image) * Config.max_lidar_distance

        # 보상 초기화
        reward = 0.0
        done = False

        # 터미널 상태 처리
        if current_goal_distance < Config.goal_threshold:
            reward = 50.0  # 목표 도달 보너스 (더 큰 값)
            self.goal_position = np.array([
                random.uniform(-18.0, 18.0),
                random.uniform(-18.0, 18.0),
                random.uniform(-18.0, 18.0)
            ])
            done = True
            info = {"status": "goal_reached", "distance_to_goal": current_goal_distance}
            print(f"Reached: {position}")
        elif has_collided:
            reward = -50.0  # 충돌 페널티 (더 큰 부정값)
            done = True
            info = {"status": "collision", "distance_to_goal": current_goal_distance}
            print(f"Collided: {position}")
        else:
            # 지속적인 보상 계산 (비터미널 상태)

            # 1. 목표 접근 보상: 이전 거리와 현재 거리의 차이 (진전 보상)
            progress_reward = (self.prev_goal_distance - current_goal_distance) * 10.0

            # 3. 장애물 회피 보상: 안전 거리에 따른 보상/페널티
            danger_zone = 2.0
            caution_zone = 4.0

            obstacle_reward = 0

            if min_depth < danger_zone:
                # 매우 위험한 상황 (강한 페널티)
                obstacle_reward = -2.0 * ((danger_zone - min_depth) / danger_zone)
            elif min_depth < caution_zone:
                # 주의 구역 (약한 페널티)
                obstacle_reward = -0.5 * ((caution_zone - min_depth) / (caution_zone - danger_zone))

            # 5. 시간 패널티: 시간이 지날수록 작은 패널티
            time_penalty = -0.001 * self.steps

            # 모든 보상 요소 결합
            reward = progress_reward + 5 * obstacle_reward + time_penalty

            info = {
                "status": "moving",
                "distance_to_goal": current_goal_distance,
                "obstacle_distance": min_depth,
                "progress_reward": progress_reward,
                "obstacle_reward": obstacle_reward
            }

        # 이전 거리 업데이트
        self.prev_goal_distance = current_goal_distance

        return reward, done, info

    def _get_lidar_data(self):
        lidar_data = self.client.getLidarData("LidarSensor1", "HelloDrone")
        lidar_points = lidar_data.point_cloud
        pose = lidar_data.pose

        position = pose.position
        position = np.array([
            position.x_val,
            position.y_val,
            position.z_val
        ], dtype=np.float32)

        if len(lidar_points) >= 3:

            # 모든 포인트를 numpy 배열로 변환 (벡터화)
            points_count = len(lidar_points) // 3
            points = np.array(lidar_points).reshape(points_count, 3)

            qw, qx, qy, qz = pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val

            # ③ 쿼터니언 → 회전 행렬
            R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)

            # 역행렬(전치행렬)을 사용하여 body에서 NED로 변환
            pts_ned = points @ R.T

            # 변환된 포인트를 튜플 리스트로 변환
            points_list = [tuple(point) for point in pts_ned]

            return lidar_to_depth_image(points_list), position

        return lidar_to_depth_image([]), position

    def _get_state(self):
        # 드론 상태와 방향 가져오기
        kinematics = self.client.simGetGroundTruthKinematics()

        # 라이다 데이터에서 깊이 이미지 가져오기
        depth_image, position = self._get_lidar_data()

        # 속도 가져오기 (NED 좌표계)
        velocity_ned = np.array([
            kinematics.linear_velocity.x_val,
            kinematics.linear_velocity.y_val,
            kinematics.linear_velocity.z_val
        ])

        # NED 좌표계에서 목표까지의 상대 벡터
        relative_goal_ned = self.goal_position - position

        # 바디 프레임 기준 상태 벡터: 속도(3), 목표 상대 위치(3), 드론 pose(3)
        drone_state = np.concatenate([velocity_ned, relative_goal_ned])

        return depth_image, drone_state, position

    def _quaternion_to_rotation_matrix(self, w, x, y, z):
        # 쿼터니언을 회전 행렬로 변환
        rotation_matrix = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
        return rotation_matrix

    def visualize_3d_lidar(self, depth_image, frame_count=0, show=True, save_path=None):
        """
        드론 내비게이션 시각화 - 고정된 환경 내에서 드론의 이동과 라이다 데이터 표시
        """
        self.trajectory = self.drone_trajectory

        # 현재 드론 위치 (경로의 마지막 지점)
        current_pos = self.drone_trajectory[-1]

        # 그림 초기화
        if not hasattr(self, 'vis_fig') or self.vis_fig is None:
            plt.ion()  # 대화형 모드 활성화
            self.vis_fig = plt.figure(figsize=(12, 10))
            self.vis_ax = self.vis_fig.add_subplot(111, projection='3d')
        else:
            # 이전 그림 요소 제거
            self.vis_ax.clear()

        # 각도 설정
        x_size, y_size = Config.depth_image_width, Config.depth_image_height
        az = np.linspace(-np.pi, np.pi, x_size)
        el = np.linspace(np.pi / 2, -np.pi / 2, y_size)
        AZ, EL = np.meshgrid(az, el)

        # 깊이 값 변환 (최대 거리 10m로 제한)  # 최대 10m 제한
        R = depth_image * Config.max_lidar_distance

        # 표시할 데이터 수 줄이기
        skip = 4

        # 구면 좌표를 3D 직교 좌표로 변환
        X = R[::skip, ::skip] * np.cos(EL[::skip, ::skip]) * np.cos(AZ[::skip, ::skip])
        Y = R[::skip, ::skip] * np.cos(EL[::skip, ::skip]) * np.sin(AZ[::skip, ::skip])
        Z = R[::skip, ::skip] * np.sin(EL[::skip, ::skip])

        # 데이터 평탄화
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        depth_flat = depth_image[::skip, ::skip].flatten()

        # 유효한 깊이 값만 표시
        mask = depth_flat < 5.0
        X_plot = X_flat[mask]
        Y_plot = Y_flat[mask]
        Z_plot = Z_flat[mask]

        # 중요: 라이다 포인트를 드론의 현재 위치 기준으로 변환
        X_global = X_plot + current_pos[0]
        Y_global = Y_plot + current_pos[1]
        Z_global = Z_plot - current_pos[2]

        # 라이다 데이터 시각화
        self.vis_ax.scatter(
            X_global, Y_global, Z_global,
            c='royalblue',
            s=8,
            alpha=0.9,
            marker='.',
            label='Detected Points'
        )

        # 드론 경로 시각화
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            # Z 좌표 반전
            traj_plot = traj_array.copy()
            traj_plot[:, 2] = -traj_plot[:, 2]  # Z축 반전

            self.vis_ax.plot(
                traj_plot[:, 0], traj_plot[:, 1], traj_plot[:, 2],
                'r-', linewidth=2.5, label='Drone Path'
            )

        # 현재 드론 위치 표시
        self.vis_ax.scatter(
            [current_pos[0]], [current_pos[1]], [-current_pos[2]],
            color='red', s=100, marker='o', label='Drone'
        )

        # 목표점 표시
        if hasattr(self, 'goal_position') and self.goal_position is not None:
            self.vis_ax.scatter(
                [self.goal_position[0]], [self.goal_position[1]], [-self.goal_position[2]],
                color='green', s=200, marker='*', label='Goal'
            )

        # 시각화 설정
        # 고정 시점 설정 (전체 환경이 보이는 위치)
        self.vis_ax.view_init(elev=30, azim=30)

        # 타이틀 및 축 레이블
        self.vis_ax.set_title(f'Drone Navigation - Frame {frame_count}')
        self.vis_ax.set_xlabel('X (m)')
        self.vis_ax.set_ylabel('Y (m)')
        self.vis_ax.set_zlabel('Z (m)')

        # 범위 설정 (환경 크기에 맞게 조정)
        bound = 20  # 환경 범위를 20m로 가정
        self.vis_ax.set_xlim([bound, -bound])
        self.vis_ax.set_ylim([-bound, bound])
        self.vis_ax.set_zlim([-bound, bound])  # Z축은 대개 높이므로 범위를 줄임

        # 그리드 및 범례
        self.vis_ax.grid(True)
        self.vis_ax.legend(loc='upper right')

        # 화면 업데이트 또는 저장
        if show:
            plt.draw()
            plt.pause(0.001)  # 짧은 일시 중지

        if save_path:
            plt.savefig(f"{save_path}_{frame_count:04d}.png", dpi=200, bbox_inches='tight')

        return self.vis_fig