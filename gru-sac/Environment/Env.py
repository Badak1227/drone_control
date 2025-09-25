import math
import random

import airsim
import numpy as np
import torch
import matplotlib.pyplot as plt

class Config:
    depth_image_height = 42
    depth_image_width = 42
    max_lidar_distance = 20
    max_drone_speed = 5
    goal_threshold = 3
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
        self.goal_position = np.array([50.0, 0.0, -10.0])

        self.goal_distance = 0
        self.prev_goal_distance = 0

        # 에피소드 스텝 카운터
        self.steps = 0

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()  # 반드시 대기  :contentReference[oaicite:14]{index=14}

        self.client.moveToPositionAsync(0, 0, 0, 5).join()

        # 목표 설정 ...
        self.steps = 0
        depth, state, _ = self._get_state()

        # 관측 일관성: step과 동일하게 방향 정규화 (또는 아예 정규화 제거로 통일)
        g = np.linalg.norm(state[3:6]) + 1e-6
        state[3:6] = state[3:6] / g
        self.prev_goal_distance = g  # 초기 prev도 맞춰 둠

        return depth, state

    def step(self, action, dt=0.2):
        # --- 이전 관측으로 블렌딩 벡터 계산 ---
        depth_prev, state_prev, _ = self._get_state()
        goal_dir = state_prev[3:6]
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
        blend = goal_dir * Config.max_drone_speed
        action = 0.2 * blend + 0.8 * action  # :contentReference[oaicite:15]{index=15}

        vx, vy, vz = map(float, action)
        self.client.moveByVelocityAsync(vx, vy, vz, dt).join()  # 반드시 대기  :contentReference[oaicite:16]{index=16}

        # --- 이동 후 새 관측으로 보상/종료 판정 ---
        depth, state, pos = self._get_state()
        reward, done, info = self._compute_reward(depth, state,
                                                  pos)  # 내부에서 prev 업데이트  :contentReference[oaicite:17]{index=17}

        # 관측 일관성 유지: 방향 성분 정규화
        g = np.linalg.norm(state[3:6]) + 1e-6
        state[3:6] = state[3:6] / g

        self.steps += 1
        if self.steps >= Config.max_episode_steps:
            done = True
            info = dict(info or {})
            info["timeout"] = True

        return depth, state, reward, done, info

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
            caution_zone = 5.0

            if min_depth < danger_zone:
                # 매우 위험한 상황 (강한 페널티)
                obstacle_reward = -2.0 * ((danger_zone - min_depth) / danger_zone)
            elif min_depth < caution_zone:
                # 주의 구역 (약한 페널티)
                obstacle_reward = -0.5 * ((caution_zone - min_depth) / (caution_zone - danger_zone))
            else:
                # 안전 구역 (작은 보상)
                obstacle_reward = 0.1

            # 4. 에너지 효율성 보상: 최적 속도 유지 장려
            speed = np.linalg.norm(drone_state[:3])
            optimal_speed = Config.max_drone_speed * 0.7  # 최대 속도의 70%를 최적으로 가정
            efficiency_reward = -0.1 * abs(speed - optimal_speed) / optimal_speed

            # 5. 시간 패널티: 시간이 지날수록 작은 패널티
            time_penalty = -0.001 * self.steps

            # 모든 보상 요소 결합
            reward = progress_reward + obstacle_reward + efficiency_reward + time_penalty

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

            return lidar_to_depth_image(points_list)

        return lidar_to_depth_image([])

    def _get_state(self):
        # 드론 상태와 방향 가져오기
        kinematics = self.client.simGetGroundTruthKinematics()
        position = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val])

        # 라이다 데이터에서 깊이 이미지 가져오기
        depth_image = self._get_lidar_data()

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
        깊이 이미지의 3D 시각화 (학습 중 실시간 모니터링용)
        depth_image: 현재 라이다 깊이 이미지
        frame_count: 현재 프레임 번호 (표시용)
        show: 화면에 표시 여부
        save_path: 저장 경로 (None이면 저장하지 않음)
        """
        if not hasattr(self, 'lidar_fig') or self.lidar_fig is None:
            # 처음 호출 시 그림 초기화
            plt.ion()  # 대화형 모드 활성화
            self.lidar_fig = plt.figure(figsize=(10, 8))
            self.lidar_ax = self.lidar_fig.add_subplot(111, projection='3d')
            self.lidar_scatter = None
            self.drone_point = None

        # 이전 산점도 제거
        if self.lidar_scatter:
            self.lidar_scatter.remove()
        if self.drone_point:
            self.drone_point.remove()

        # 각도 설정
        x_size, y_size = Config.depth_image_width, Config.depth_image_height
        az = np.linspace(-np.pi, np.pi, x_size)
        el = np.linspace(np.pi / 2, -np.pi / 2, y_size)
        AZ, EL = np.meshgrid(az, el)

        # 깊이 값 변환
        R = depth_image * Config.max_lidar_distance

        # 표시할 데이터 수 줄이기 (성능 향상)
        skip = 4  # 4개마다 하나씩 표시

        # 구면 좌표를 3D 직교 좌표로 변환
        X = R[::skip, ::skip] * np.cos(EL[::skip, ::skip]) * np.cos(AZ[::skip, ::skip])
        Y = R[::skip, ::skip] * np.cos(EL[::skip, ::skip]) * np.sin(AZ[::skip, ::skip])
        Z = R[::skip, ::skip] * np.sin(EL[::skip, ::skip])

        # 데이터 평탄화
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        depth_flat = depth_image[::skip, ::skip].flatten()

        mask = depth_flat < 1.0
        X_plot = X_flat[mask]
        Y_plot = Y_flat[mask]
        Z_plot = Z_flat[mask]
        colors = depth_flat[mask]

        self.lidar_scatter = self.lidar_ax.scatter(
            X_plot, Y_plot, Z_plot,
            c=colors,
            cmap='viridis_r',
            s=3,
            alpha=0.8,
            marker='.'
        )

        # 드론 위치 표시
        self.drone_point = self.lidar_ax.scatter([0], [0], [0], color='red', s=50, marker='o')

        # 축 설정
        self.lidar_ax.set_title(f'Training Step: {frame_count}')
        self.lidar_ax.set_xlabel('North (m)')
        self.lidar_ax.set_ylabel('East (m)')
        self.lidar_ax.set_zlabel('Down (m)')

        # 축 범위 설정
        max_vis_range = Config.max_lidar_distance / 2
        self.lidar_ax.set_xlim([-max_vis_range, max_vis_range])
        self.lidar_ax.set_ylim([-max_vis_range, max_vis_range])
        self.lidar_ax.set_zlim([-max_vis_range, max_vis_range])

        # 화면 업데이트 (학습 속도에 영향을 최소화하기 위해 빠르게 처리)
        if show:
            self.lidar_fig.canvas.draw_idle()
            plt.pause(0.001)  # 매우 짧은 일시 중지

        # 프레임 저장
        if save_path:
            plt.savefig(save_path)

        return self.lidar_fig