import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------
# 1. 실제 데이터(1차 + 2차) 통계 기반으로 만든 파라미터
#    - 길이 범위(len_min, len_max)는 네가 준 데이터에서 그대로 가져옴
#    - center / amp는 1차 데이터 분포를 기반으로 대략 절반 스케일
# ----------------------------------------------------
MOTION_CONFIG = {
    "circle": {
        "len_min": 193,
        "len_max": 281,
        "x_center": 310.5,
        "x_amp": 134.8,
        "y_center": -273.5,
        "y_amp": 172.8,
        "z_center": -105.0,
        "z_amp": 115.0,
    },
    "horizontal": {
        "len_min": 330,
        "len_max": 633,
        "x_center": 296.0,
        "x_amp": 131.5,
        "y_center": -154.0,
        "y_amp": 243.0,   # Y축이 크게 움직이는 동작
        "z_center": -170.0,
        "z_amp": 34.5,
    },
    "vertical": {
        "len_min": 216,
        "len_max": 529,
        "x_center": 398.0,
        "x_amp": 93.0,
        "y_center": -108.0,
        "y_amp": 66.5,
        "z_center": 98.0,
        "z_amp": 268.0,   # Z축이 크게 움직이는 동작
    },
    "diagonal_left": {
        "len_min": 220,
        "len_max": 431,
        "x_center": 420.0,
        "x_amp": 65.0,
        "y_center": 20.0,
        "y_amp": 123.0,
        "z_center": -19.5,
        "z_amp": 140.2,
    },
    "diagonal_right": {
        "len_min": 250,
        "len_max": 442,
        "x_center": 467.5,
        "x_amp": 46.2,
        "y_center": 3.5,
        "y_amp": 151.8,
        "z_center": 95.5,
        "z_amp": 181.8,
    },
}


def generate_synthetic_trajectory(motion_type: str,
                                  noise_std: float = 3.0,
                                  static_ratio_range=(0.05, 0.12)) -> np.ndarray:
    """
    네 실제 데이터(1차 + 2차) 통계 기반으로
    circle / horizontal / vertical / diagonal_left / diagonal_right
    에 해당하는 3D 궤적(x,y,z)을 생성.

    반환: (T, 3) numpy 배열, 각 행은 [x, y, z]
    """
    cfg = MOTION_CONFIG[motion_type]

    # 길이: 실제 데이터에서 관찰된 min~max 사이에서 랜덤 선택
    total_len = np.random.randint(cfg["len_min"], cfg["len_max"] + 1)

    # 앞부분 static (멈춰있는 구간) 길이
    static_ratio = np.random.uniform(*static_ratio_range)
    static_len = max(5, int(total_len * static_ratio))
    dynamic_len = max(10, total_len - static_len)

    t = np.linspace(0, 1, dynamic_len)

    xc, xa = cfg["x_center"], cfg["x_amp"]
    yc, ya = cfg["y_center"], cfg["y_amp"]
    zc, za = cfg["z_center"], cfg["z_amp"]

    # ----------------------------
    # 동작별 기본 궤적 모양(Base shape)
    # ----------------------------
    if motion_type == "circle":
        # Y-Z 평면에서 타원형 궤적, X는 약간만 출렁
        x = xc + xa * 0.3 * np.cos(2 * np.pi * t)
        y = yc + ya * np.cos(2 * np.pi * t)
        z = zc + za * np.sin(2 * np.pi * t)

    elif motion_type == "horizontal":
        # Y축의 큰 왕복 + X/Z는 약간의 흔들림
        x = xc + xa * 0.1 * np.sin(2 * np.pi * t)
        y = yc + ya * np.sin(2 * np.pi * t)
        z = zc + za * 0.05 * np.sin(4 * np.pi * t)

    elif motion_type == "vertical":
        # Z축의 큰 왕복 + X는 약간의 이동, Y는 거의 일정
        x = xc + xa * 0.15 * np.sin(2 * np.pi * t)
        y = yc + ya * 0.2 * np.sin(2 * np.pi * t + np.pi / 4)
        z = zc + za * np.sin(2 * np.pi * t)

    elif motion_type == "diagonal_left":
        # 왼쪽 아래 ↘ 방향 대각선 + 약간의 커브
        x = xc + xa * (t - 0.5) * 0.8
        y = yc - ya * (t + 0.2) + ya * 0.2 * np.sin(2 * np.pi * t)
        z = zc - za * 0.5 * t + za * 0.2 * np.sin(2 * np.pi * t)

    elif motion_type == "diagonal_right":
        # 오른쪽 아래 ↙ 방향 대각선 + 약간의 커브
        x = xc + xa * (t - 0.5) * 0.8
        y = yc + ya * (t + 0.2) + ya * 0.2 * np.sin(2 * np.pi * t)
        z = zc - za * 0.5 * t + za * 0.2 * np.sin(2 * np.pi * t)

    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

    traj_dyn = np.stack([x, y, z], axis=1)

    # ----------------------------
    # 센서 노이즈 + static 구간 추가
    # ----------------------------
    # dynamic 구간에 노이즈 (2차 dataset과 비슷한 정도)
    traj_dyn_noisy = traj_dyn + np.random.normal(0, noise_std, traj_dyn.shape)

    # static 구간: 시작 위치 근처에서 약간만 떠는 구간
    start_pos = traj_dyn_noisy[0]
    static_noise = np.random.normal(0, noise_std * 0.4, size=(static_len, 3))
    traj_static = start_pos + static_noise

    traj_full = np.vstack([traj_static, traj_dyn_noisy])

    # 실제 데이터처럼 정수 좌표로 반올림
    traj_full = np.round(traj_full).astype(int)

    return traj_full


def save_trajectory_as_csv(traj: np.ndarray, path: Path):
    """
    1_preprocessing.py가 읽을 수 있는 형식으로 저장
    - 각 줄: "x/y/z"
    - header 없음, index 없음
    """
    df = pd.DataFrame(traj)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, header=False, index=False, sep='/')


def main():
    # 생성할 synthetic 샘플 개수 (원하면 여기 숫자만 바꾸면 됨)
    num_per_class = 10

    output_root = Path("synthetic_test")
    motion_types = ["circle", "horizontal", "vertical",
                    "diagonal_left", "diagonal_right"]

    print("Generating synthetic test set...")
    for motion in motion_types:
        for i in range(1, num_per_class + 1):
            traj = generate_synthetic_trajectory(motion)
            out_path = output_root / motion / f"synth_{motion}_{i}.csv"
            save_trajectory_as_csv(traj, out_path)
        print(f"  - {motion}: {num_per_class} samples saved")

    print("\nDone! You can now point DATA_ROOT to 'synthetic_test' and")
    print("use them as an additional test set for your Random Forest pipeline.")


if __name__ == "__main__":
    main()
