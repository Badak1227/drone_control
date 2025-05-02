import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class SpeedEstimator:
    """UWB range 변화를 이용한 속도 추정기 (논문 Section III-B)"""

    def __init__(self, delta_t=0.1):
        self.delta_t = delta_t
        self.range_history = []
        self.time_history = []
        self.speed_history = []

    def update(self, range_measurement, time):
        self.range_history.append(range_measurement)
        self.time_history.append(time)

        # 최소 3개의 측정값이 필요
        if len(self.range_history) < 3:
            return 0.0

        # 최근 3개 값만 사용
        r0 = self.range_history[-3]
        r1 = self.range_history[-2]
        r2 = self.range_history[-1]

        t0 = self.time_history[-3]
        t2 = self.time_history[-1]

        # 실제 시간 간격 계산
        actual_dt = (t2 - t0) / 2

        # 식 (6): 속도 계산
        numerator = r2 ** 2 + r0 ** 2 - 2 * r1 ** 2
        denominator = 2 * actual_dt ** 2

        if denominator <= 0 or numerator < 0:
            return 0.0

        speed = np.sqrt(numerator / denominator)
        self.speed_history.append(speed)

        return speed


class ExtendedKalmanFilter:
    """단일 UWB 앵커를 위한 Extended Kalman Filter (논문 식(1), (2))"""

    def __init__(self, initial_state, initial_covariance, anchor_position=[0, 0]):
        self.state = initial_state  # [x, y, theta, v, w]
        self.covariance = initial_covariance
        self.anchor_position = anchor_position
        self.dt = 0.1

        # 프로세스 노이즈
        self.Q = np.eye(5) * 0.01

        # 측정 노이즈
        self.R = np.eye(3)
        self.R[0, 0] = 0.2  # UWB range noise
        self.R[1, 1] = 0.1  # heading noise
        self.R[2, 2] = 0.5  # velocity noise

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt

        x, y, theta, v, w = self.state

        # 상태 전이 모델 (식 1)
        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt

        # 상태 예측
        new_state = np.zeros(5)
        new_state[0] = x + v * np.cos(theta) * dt
        new_state[1] = y + v * np.sin(theta) * dt
        new_state[2] = theta + w * dt
        new_state[3] = v
        new_state[4] = w

        # 공분산 예측
        new_covariance = F @ self.covariance @ F.T + self.Q

        self.state = new_state
        self.covariance = new_covariance

        return self.state

    def update(self, measurements):
        """measurements = [range, heading, speed]"""
        x, y, theta, v, w = self.state

        # 예측된 측정값 계산 (식 2)
        predicted_range = np.sqrt((x - self.anchor_position[0]) ** 2 +
                                  (y - self.anchor_position[1]) ** 2)
        predicted_heading = theta
        predicted_velocity = v

        # 혁신(Innovation)
        innovation = np.array([
            measurements[0] - predicted_range,
            measurements[1] - predicted_heading,
            measurements[2] - predicted_velocity
        ])

        # 관측 행렬의 자코비안
        H = np.zeros((3, 5))
        if predicted_range > 1e-6:
            H[0, 0] = (x - self.anchor_position[0]) / predicted_range
            H[0, 1] = (y - self.anchor_position[1]) / predicted_range
        H[1, 2] = 1
        H[2, 3] = 1

        # 칼만 이득 계산
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # 상태 업데이트
        self.state = self.state + K @ innovation

        # 공분산 업데이트
        I = np.eye(5)
        self.covariance = (I - K @ H) @ self.covariance

        return self.state


def quaternion_to_euler(x, y, z, w):
    """쿼터니언을 오일러 각도(yaw)로 변환"""
    rotation = R.from_quat([x, y, z, w])
    euler = rotation.as_euler('xyz')
    return euler[2]  # yaw


def extract_range_data(ranges_str):
    """ranges 문자열에서 거리 데이터 추출"""
    try:
        # 첫 번째 range 값 추출
        ranges = ranges_str.split(',')
        for r in ranges:
            if 'range:' in r:
                value = float(r.split(':')[1].strip())
                return value
    except:
        return None
    return None


def synchronize_data(flare_data, imu_data, max_time_diff=0.1):
    """UWB와 IMU 데이터 동기화"""
    synced_data = []

    for i, flare_row in flare_data.iterrows():
        flare_time = flare_row['Time']

        # 가장 가까운 IMU 데이터 찾기
        time_diffs = np.abs(imu_data['Time'] - flare_time)
        min_idx = time_diffs.argmin()
        min_diff = time_diffs.iloc[min_idx]

        if min_diff < max_time_diff:
            imu_row = imu_data.iloc[min_idx]

            # UWB range 추출
            range_value = extract_range_data(flare_row['ranges'])

            if range_value is not None:
                synced_data.append({
                    'time': flare_time,
                    'range': range_value,
                    'true_x': flare_row['pos.x'],
                    'true_y': flare_row['pos.y'],
                    'quat_x': imu_row['orientation.x'],
                    'quat_y': imu_row['orientation.y'],
                    'quat_z': imu_row['orientation.z'],
                    'quat_w': imu_row['orientation.w']
                })

    return pd.DataFrame(synced_data)


def main():
    # 데이터 로드
    flare_data = pd.read_csv('rtls_flares.csv')
    imu_data = pd.read_csv('waveshare_sense_hat_b.csv')

    # 데이터 동기화
    synced_data = synchronize_data(flare_data, imu_data)

    # 알고리즘 초기화
    speed_estimator = SpeedEstimator(delta_t=0.1)

    # EKF 초기화
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, theta, v, w]
    initial_covariance = np.eye(5) * 0.1
    ekf = ExtendedKalmanFilter(initial_state, initial_covariance)

    # 결과 저장을 위한 리스트
    estimated_trajectory = []
    true_trajectory = []
    estimated_speeds = []
    range_measurements = []

    # 알고리즘 실행
    for i, row in synced_data.iterrows():
        # UWB range 데이터 처리
        time = row['time']
        range_measurement = row['range']

        # 속도 추정
        speed = speed_estimator.update(range_measurement, time)

        # IMU 방향 데이터 처리
        yaw = quaternion_to_euler(row['quat_x'], row['quat_y'],
                                  row['quat_z'], row['quat_w'])

        # EKF 예측 단계
        ekf.predict()

        # EKF 업데이트 단계
        measurements = [range_measurement, yaw, speed]
        state = ekf.update(measurements)

        # 결과 저장
        estimated_trajectory.append([state[0], state[1]])
        true_trajectory.append([row['true_x'], row['true_y']])
        estimated_speeds.append(speed)
        range_measurements.append(range_measurement)

    # 결과 시각화
    estimated_trajectory = np.array(estimated_trajectory)
    true_trajectory = np.array(true_trajectory)

    # 경로 플롯
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], 'b-', label='Estimated')
    plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'r--', label='True')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.title('Robot Trajectory')
    plt.axis('equal')

    # 속도 플롯
    plt.subplot(132)
    plt.plot(estimated_speeds, 'g-', label='Estimated Speed')
    plt.xlabel('Sample')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.title('Speed Estimation')

    # Range 플롯
    plt.subplot(133)
    plt.plot(range_measurements, 'c-', label='UWB Range')
    plt.xlabel('Sample')
    plt.ylabel('Range (m)')
    plt.legend()
    plt.title('UWB Range Measurements')

    plt.tight_layout()
    plt.show()

    # 에러 계산
    position_error = np.linalg.norm(estimated_trajectory - true_trajectory, axis=1)
    rmse = np.sqrt(np.mean(position_error ** 2))
    print(f"Position RMSE: {rmse:.3f} m")


if __name__ == "__main__":
    main()