# ============================================================
# 2_feature_engineering.py  (FINAL MERGED VERSION)
# 기존 feature + Smart feature 모두 포함 (총 26개)
# ============================================================

import numpy as np
from scipy.stats import pearsonr


# ------------------------------------------------------------
# 🧠 SMART FEATURE EXTRACTOR
# ------------------------------------------------------------

def extract_smart_features(data):
    data = np.array(data)

    # 1) 위치 기반 통계
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)

    # 2) 안전한 상관계수 (대각선 방향 구분)
    def safe_corr(a, b):
        return pearsonr(a, b)[0] if np.std(a)>1e-6 and np.std(b)>1e-6 else 0

    corr_xy = safe_corr(data[:,0], data[:,1])
    corr_yz = safe_corr(data[:,1], data[:,2])
    corr_zx = safe_corr(data[:,2], data[:,0])

    # 3) 원 모양 판단 (linearity)
    displacement = np.linalg.norm(data[-1] - data[0])
    step = np.linalg.norm(np.diff(data, axis=0), axis=1)
    total_path = np.sum(step)
    linearity = displacement / (total_path + 1e-6)

    # 4) start-end closure (circle 판단)
    closure = np.linalg.norm(data[0] - data[-1])

    # 5) Variance ratio (축 별 비율)
    total_var = np.sum(var) + 1e-6
    var_ratio = var / total_var

    # 6) 속도 기반 특징
    vel = step
    mean_vel = np.mean(vel)
    max_vel = np.max(vel)

    # 7) 평균 회전각 (circle은 회전 많음)
    angles = []
    for i in range(len(data)-2):
        a = data[i+1] - data[i]
        b = data[i+2] - data[i+1]
        denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)
        ang = np.arccos(np.clip(np.dot(a,b)/denom, -1, 1))
        angles.append(ang)
    mean_turn_angle = np.mean(angles)

    return np.array([
        # 위치 기반 정보
        mean[0], mean[1], mean[2],
        var[0], var[1], var[2],

        # 대각선 방향 관련
        corr_xy, corr_yz, corr_zx,

        # 원/직선 관련
        linearity, closure,

        # 속도 정보
        mean_vel, max_vel,

        # 방향 전환량
        mean_turn_angle,

        # 축 비율
        var_ratio[0], var_ratio[1], var_ratio[2]
    ])


# ------------------------------------------------------------
# 🏗 기존 FEATURE (너의 원래 코드 기반)
# ------------------------------------------------------------

def extract_basic_features(data):

    vel = np.linalg.norm(np.diff(data, axis=0), axis=1)
    max_vel = np.max(vel)

    var_y = np.var(data[:, 1])
    var_z = np.var(data[:, 2])
    ratio_yz = var_y / (var_z + 1e-6)

    mean_x = np.mean(data[:, 0])
    mean_y = np.mean(data[:, 1])

    step = np.linalg.norm(np.diff(data, axis=0), axis=1)
    path_length = np.sum(step)

    dist_se = np.linalg.norm(data[-1] - data[0])
    linearity_ratio = dist_se / (path_length + 1e-6)

    range_x = np.ptp(data[:, 0])
    range_y = np.ptp(data[:, 1])
    range_z = np.ptp(data[:, 2])

    return np.array([
        mean_x,
        max_vel,
        ratio_yz,
        mean_y,
        path_length,
        linearity_ratio,
        range_x,
        range_y,
        range_z
    ])


# ------------------------------------------------------------
# 🎯 최종 FEATURE 합본 (기존 9개 + 스마트 17개 = 총 26개)
# ------------------------------------------------------------

def extract_features(X):
    """
    X : (N, 100, 3) shaped preprocessed sequences
    return: (N, 26) feature array + feature names
    """

    feature_list = []

    for seq in X:
        f_basic = extract_basic_features(seq)            # 기존 feature 9개
        f_smart = extract_smart_features(seq)            # 스마트 feature 17개
        combined = np.concatenate([f_basic, f_smart])    # 총 26개

        feature_list.append(combined)

    feature_list = np.array(feature_list)

    # 이름 매핑
    feature_names = [
        # 기존 feature
        "mean_x", "max_vel", "ratio_yz", "mean_y",
        "path_length", "linearity_ratio",
        "range_x", "range_y", "range_z",

        # 스마트 feature
        "s_mean_x", "s_mean_y", "s_mean_z",
        "s_var_x", "s_var_y", "s_var_z",
        "corr_xy", "corr_yz", "corr_zx",
        "smart_linearity", "closure",
        "mean_vel", "max_vel_smart",
        "turn_angle_mean",
        "var_ratio_x", "var_ratio_y", "var_ratio_z"
    ]

    return feature_list, feature_names
