import torch
import torch.nn as nn
import torch.nn.functional as F


# 테스트용 설정
class TestConfig:
    hidden_dim = 256
    cnn_features = 64
    gru_hidden_dim = 128
    gru_num_layers = 2


# 간단한 CNN 구현 (테스트용)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 5 * 5, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))


# 데이터 흐름 검증용 클래스
class TestVisualStateSummarizer(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        # CNN for visual features
        self.cnn = SimpleCNN()

        # State processing
        self.state_fc = nn.Linear(state_dim, TestConfig.cnn_features)

        # Combined feature processing
        self.feature_combine = nn.Linear(TestConfig.cnn_features * 2, TestConfig.hidden_dim)

        # GRU for temporal processing
        self.gru = nn.GRU(
            input_size=TestConfig.hidden_dim,
            hidden_size=TestConfig.gru_hidden_dim,
            num_layers=TestConfig.gru_num_layers,
            batch_first=True
        )

    def forward_sequence_with_tracking(self, depth_images, states):
        """순서를 추적하면서 처리하는 함수"""

        batch_size, seq_len = depth_images.shape[:2]

        print(f"입력 shape: {depth_images.shape}")

        # 각 요소에 고유한 식별자 추가 (순서 추적용)
        # 예: 첫 번째 배치, 첫 번째 시퀀스의 왼쪽 상단 부분에 식별자 추가
        for b in range(batch_size):
            for s in range(seq_len):
                depth_images[b, s, 0, 0] = b + s * 0.1

        # Reshape for CNN processing
        depth_reshaped = depth_images.reshape(batch_size * seq_len, *depth_images.shape[2:])
        print(f"Reshape 후 shape: {depth_reshaped.shape}")

        # 각 요소의 식별자 확인
        for i in range(min(5, batch_size * seq_len)):
            print(f"Reshaped index {i}: value[0,0] = {depth_reshaped[i, 0, 0]}")

        # CNN processing
        visual_features = self.cnn(depth_reshaped.unsqueeze(1))
        print(f"CNN 후 shape: {visual_features.shape}")

        # 시퀀스로 복원
        visual_features_reshaped = visual_features.reshape(batch_size, seq_len, -1)
        print(f"복원 후 shape: {visual_features_reshaped.shape}")

        # State processing
        state_features = F.relu(self.state_fc(states))

        # Combine features
        combined = torch.cat([visual_features_reshaped, state_features], dim=-1)
        combined = F.relu(self.feature_combine(combined))

        # GRU processing
        gru_output, new_hidden = self.gru(combined)

        return gru_output, new_hidden


# 테스트 코드
def test_data_flow():
    # 테스트 설정
    batch_size = 2
    seq_len = 3
    height, width = 84, 84  # 예시 이미지 크기
    state_dim = 10

    # 테스트 데이터 생성
    depth_images = torch.randn(batch_size, seq_len, height, width)
    states = torch.randn(batch_size, seq_len, state_dim)

    # 모델 생성
    model = TestVisualStateSummarizer(state_dim)

    # 데이터 흐름 추적
    with torch.no_grad():
        output, hidden = model.forward_sequence_with_tracking(depth_images, states)

    print(f"\n최종 GRU 출력 shape: {output.shape}")
    print(f"Hidden state shape: {hidden.shape}")


# 순서 유지 검증 테스트
def verify_order_preservation():
    batch_size = 4
    seq_len = 5
    height, width = 84, 84
    state_dim = 10

    # 테스트 데이터 생성 (패턴이 있는 데이터)
    depth_images = torch.zeros(batch_size, seq_len, height, width)

    # 각 위치에 고유한 값 할당
    for b in range(batch_size):
        for s in range(seq_len):
            # 식별 가능한 패턴 생성
            value = b * 100 + s
            depth_images[b, s, :, :] = value

    states = torch.randn(batch_size, seq_len, state_dim)

    model = TestVisualStateSummarizer(state_dim)

    print("=== 순서 유지 검증 ===")
    print("\n원본 데이터 패턴:")
    for b in range(batch_size):
        for s in range(seq_len):
            print(f"Batch {b}, Seq {s}: {depth_images[b, s, 0, 0]}")

    # Forward pass
    with torch.no_grad():
        # CNN 처리 전 reshape
        batch_size, seq_len = depth_images.shape[:2]
        depth_reshaped = depth_images.reshape(batch_size * seq_len, *depth_images.shape[2:])

        print("\nReshape 후 패턴:")
        for i in range(batch_size * seq_len):
            print(f"Reshaped index {i}: {depth_reshaped[i, 0, 0]}")

        # Reshape 복원 테스트
        depth_recovered = depth_reshaped.reshape(batch_size, seq_len, *depth_images.shape[2:])

        print("\n복원 후 원본과 비교:")
        for b in range(batch_size):
            for s in range(seq_len):
                original_val = depth_images[b, s, 0, 0]
                recovered_val = depth_recovered[b, s, 0, 0]
                print(
                    f"Batch {b}, Seq {s}: Original={original_val}, Recovered={recovered_val}, Match={original_val == recovered_val}")


# 모든 테스트 실행
if __name__ == "__main__":
    print("=== 기본 데이터 흐름 테스트 ===")
    test_data_flow()

    print("\n" + "=" * 50 + "\n")

    print("=== 순서 유지 검증 테스트 ===")
    verify_order_preservation()