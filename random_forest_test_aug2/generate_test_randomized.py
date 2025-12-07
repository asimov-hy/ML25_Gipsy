import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------
# 1. 기본 통계 파라미터 (기준점)
# ----------------------------------------------------
MOTION_CONFIG = {
    "circle": {
        "len_min": 193, "len_max": 281,
        "x_center": 310.5, "x_amp": 134.8,
        "y_center": -273.5, "y_amp": 172.8,
        "z_center": -105.0, "z_amp": 115.0,
    },
    "horizontal": {
        "len_min": 330, "len_max": 633,
        "x_center": 296.0, "x_amp": 131.5,
        "y_center": -154.0, "y_amp": 243.0,
        "z_center": -170.0, "z_amp": 34.5,
    },
    "vertical": {
        "len_min": 216, "len_max": 529,
        "x_center": 398.0, "x_amp": 93.0,
        "y_center": -108.0, "y_amp": 66.5,
        "z_center": 98.0, "z_amp": 268.0,
    },
    "diagonal_left": {
        "len_min": 220, "len_max": 431,
        "x_center": 420.0, "x_amp": 65.0,
        "y_center": 20.0, "y_amp": 123.0,
        "z_center": -19.5, "z_amp": 140.2,
    },
    "diagonal_right": {
        "len_min": 250, "len_max": 442,
        "x_center": 467.5, "x_amp": 46.2,
        "y_center": 3.5, "y_amp": 151.8,
        "z_center": 95.5, "z_amp": 181.8,
    },
}

def apply_random_rotation(traj, angle_range=15):
    """
    3D 궤적을 랜덤하게 회전시킴 (자세 틀어짐 모사)
    """
    # X, Y, Z 축 각각에 대해 -15도 ~ +15도 사이 랜덤 회전
    angles = np.random.uniform(-angle_range, angle_range, size=3)
    r = R.from_euler('xyz', angles, degrees=True)
    
    # 중심점을 기준으로 회전시키기 위해 중심 이동
    center = np.mean(traj, axis=0)
    traj_centered = traj - center
    traj_rotated = r.apply(traj_centered)
    
    return traj_rotated + center

def apply_shape_distortion(t, magnitude=0.1):
    """
    기본 궤적(sin/cos)에 저주파 노이즈를 섞어 '찌그러진' 형태를 만듦
    """
    # sin(3t), cos(4t) 등을 섞어서 완벽한 타원이 아니게 만듦
    distortion = (np.random.uniform(-1, 1) * 0.5 * np.sin(3 * np.pi * t) +
                  np.random.uniform(-1, 1) * 0.3 * np.cos(5 * np.pi * t))
    return distortion * magnitude

def generate_synthetic_trajectory(motion_type: str,
                                  noise_std: float = 3.0,
                                  static_ratio_range=(0.05, 0.12)) -> np.ndarray:
    
    cfg = MOTION_CONFIG[motion_type]

    # [Random 1] 길이 랜덤성 강화
    total_len = np.random.randint(cfg["len_min"], cfg["len_max"] + 1)
    
    # Static 구간 설정
    static_ratio = np.random.uniform(*static_ratio_range)
    static_len = max(5, int(total_len * static_ratio))
    dynamic_len = max(10, total_len - static_len)

    t = np.linspace(0, 1, dynamic_len)

    # [Random 2] 파라미터 랜덤성 (User Variance)
    # 중심점(Center)을 매번 ±30mm 이동 (사람마다 앉는 위치 다름)
    xc = cfg["x_center"] + np.random.uniform(-30, 30)
    yc = cfg["y_center"] + np.random.uniform(-30, 30)
    zc = cfg["z_center"] + np.random.uniform(-30, 30)

    # 움직임 크기(Amplitude)를 0.8배 ~ 1.2배로 랜덤 조절 (팔 길이 차이)
    scale_factor = np.random.uniform(0.8, 1.2)
    xa = cfg["x_amp"] * scale_factor
    ya = cfg["y_amp"] * scale_factor
    za = cfg["z_amp"] * scale_factor

    # [Random 3] Shape Distortion (찌그러짐 추가)
    # 기본 t에 왜곡을 주거나, 좌표 자체에 왜곡 추가
    dist_x = apply_shape_distortion(t) * xa
    dist_y = apply_shape_distortion(t) * ya
    dist_z = apply_shape_distortion(t) * za

    # --- 기본 궤적 생성 ---
    if motion_type == "circle":
        # 위상(Phase)도 랜덤하게 조금 틀어줌
        phase_shift = np.random.uniform(-0.2, 0.2)
        x = xc + xa * 0.3 * np.cos(2 * np.pi * t + phase_shift) + dist_x
        y = yc + ya * np.cos(2 * np.pi * t) + dist_y
        z = zc + za * np.sin(2 * np.pi * t) + dist_z

    elif motion_type == "horizontal":
        x = xc + xa * 0.1 * np.sin(2 * np.pi * t) + dist_x
        # 사인파 대신 약간 비대칭적인 움직임 추가
        y = yc + ya * np.sin(2 * np.pi * t) + dist_y
        z = zc + za * 0.05 * np.sin(4 * np.pi * t) + dist_z

    elif motion_type == "vertical":
        x = xc + xa * 0.15 * np.sin(2 * np.pi * t) + dist_x
        y = yc + ya * 0.2 * np.sin(2 * np.pi * t + np.pi / 4) + dist_y
        z = zc + za * np.sin(2 * np.pi * t) + dist_z

    elif motion_type == "diagonal_left":
        # 선형 이동에도 굴곡 추가
        x = xc + xa * (t - 0.5) * 0.8 + dist_x
        y = yc - ya * (t + 0.2) + ya * 0.2 * np.sin(2 * np.pi * t) + dist_y
        z = zc - za * 0.5 * t + za * 0.2 * np.sin(2 * np.pi * t) + dist_z

    elif motion_type == "diagonal_right":
        x = xc + xa * (t - 0.5) * 0.8 + dist_x
        y = yc + ya * (t + 0.2) + ya * 0.2 * np.sin(2 * np.pi * t) + dist_y
        z = zc - za * 0.5 * t + za * 0.2 * np.sin(2 * np.pi * t) + dist_z

    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

    traj_dyn = np.stack([x, y, z], axis=1)

    # [Random 4] 3D 회전 적용 (자세 틀어짐)
    traj_dyn = apply_random_rotation(traj_dyn, angle_range=15)

    # 센서 노이즈 추가
    # 노이즈 크기도 매번 조금씩 다르게 (2.0 ~ 4.0 사이)
    current_noise = np.random.uniform(noise_std * 0.7, noise_std * 1.3)
    traj_dyn_noisy = traj_dyn + np.random.normal(0, current_noise, traj_dyn.shape)

    # Static 구간 추가
    start_pos = traj_dyn_noisy[0]
    static_noise = np.random.normal(0, current_noise * 0.4, size=(static_len, 3))
    traj_static = start_pos + static_noise

    traj_full = np.vstack([traj_static, traj_dyn_noisy])
    traj_full = np.round(traj_full).astype(int)

    return traj_full

def save_trajectory_as_csv(traj: np.ndarray, path: Path):
    """
    csv 포맷 저장 (구분자는 '/' 유지 - 시각화 도구 호환성 위해)
    """
    df = pd.DataFrame(traj)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 기존 코드와 동일하게 '/' 구분자 사용
    df.to_csv(path, header=False, index=False, sep='/')

def main():
    num_per_class = 10
    output_root = Path("synthetic_test_randomized")
    
    # 모든 클래스 생성
    motion_types = ["circle", "horizontal", "vertical", "diagonal_left", "diagonal_right"]

    print("Generating HIGH VARIANCE synthetic test set...")
    for motion in motion_types:
        for i in range(1, num_per_class + 1):
            traj = generate_synthetic_trajectory(motion)
            out_path = output_root / motion / f"synth_{motion}_{i}.csv"
            save_trajectory_as_csv(traj, out_path)
        print(f"  - {motion}: {num_per_class} samples saved (All different!)")

    print("\nDone! Check 'synthetic_test_randomized' folder.")

if __name__ == "__main__":
    main()