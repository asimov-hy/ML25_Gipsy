import numpy as np
from scipy.stats import pearsonr
# from scipy.signal import welch # 사용되지 않는 모듈은 주석 처리 또는 제거

# ------------------------------------------------------
# 1. 기본 특징 추출 (Basic Features) - 사용은 안하지만 함수 정의는 유지
# ------------------------------------------------------
def extract_basic_features(data):
    # ... (기존 9개 특징 계산 로직) ...
    # 이 함수는 아래의 extract_features에서는 호출되지 않습니다.
    return np.array([0.0] * 9) # 임시로 9개짜리 배열을 반환한다고 가정


# ------------------------------------------------------
# 2. 스마트 특징 추출 (Smart Features) - 17개
# ------------------------------------------------------
def extract_smart_features(data):
    """
    Statistical and kinematic features for complex motion. (17 features)
    """
    data = np.array(data)

    # 1) 위치 통계: 평균 및 분산 (6개)
    mean_x, mean_y, mean_z = np.mean(data, axis=0)
    var_x, var_y, var_z = np.var(data, axis=0)

    # 2) 상관 계수 (3개)
    corr_xy = pearsonr(data[:, 0], data[:, 1])[0]
    corr_yz = pearsonr(data[:, 1], data[:, 2])[0]
    corr_zx = pearsonr(data[:, 2], data[:, 0])[0]

    # 3) 선형성 (Linearity)
    start_point = data[0]
    line_vec = data[-1] - start_point
    
    if np.linalg.norm(line_vec) > 1e-6:
        line_vec = line_vec / np.linalg.norm(line_vec)
        projected_dist = np.dot(data - start_point, line_vec)
        linearity = np.var(projected_dist) / (np.var(np.linalg.norm(data - start_point, axis=1)) + 1e-6)
    else:
        linearity = 0.0

    # 4) 폐쇄성 (Closure)
    closure = np.linalg.norm(data[-1] - data[0])

    # 5) 속도 통계
    velocity = np.linalg.norm(np.diff(data, axis=0), axis=1)
    mean_vel = np.mean(velocity)
    max_vel_smart = np.max(velocity)
    
    # 6) 평균 회전각 (Mean Turning Angle)
    angles = []
    for i in range(len(data)-2):
        a = data[i+1] - data[i]
        b = data[i+2] - data[i+1]
        denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)
        ang = np.arccos(np.clip(np.dot(a,b)/denom, -1, 1)) 
        angles.append(ang)
        
    mean_turn_angle = np.mean(angles) if angles else 0.0

    # 7) 축별 분산 비율 (3개)
    var_ratio_x = var_y / (var_x + 1e-6)
    var_ratio_y = var_z / (var_y + 1e-6)
    var_ratio_z = var_x / (var_z + 1e-6)

    return np.array([
        mean_x, mean_y, mean_z, var_x, var_y, var_z,
        corr_xy, corr_yz, corr_zx,
        linearity, closure,
        mean_vel, max_vel_smart, mean_turn_angle,
        var_ratio_x, var_ratio_y, var_ratio_z
    ])


# ------------------------------------------------------
# 3. 메인 특징 추출 함수 (17개 특징만 반환하도록 수정)
# ------------------------------------------------------

# 17개 스마트 특징의 이름
feature_names_17 = [
    's_mean_x', 's_mean_y', 's_mean_z', 's_var_x', 's_var_y', 's_var_z',
    'corr_xy', 'corr_yz', 'corr_zx',
    'linearity', 'closure',
    'mean_vel', 'max_vel_smart', 'turn_angle_mean',
    'var_ratio_x', 'var_ratio_y', 'var_ratio_z'
]

def extract_features(X):
    """
    X : (N, 100, 3) shaped preprocessed sequences
    return: (N, 17) feature array + feature names (Smart Features Only)
    """
    feature_list = []

    for seq in X:
        # 17개 스마트 특징만 추출
        f_smart = extract_smart_features(seq)
        feature_list.append(f_smart)

    feature_list = np.array(feature_list)
    
    return feature_list, feature_names_17