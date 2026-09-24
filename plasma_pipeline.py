#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血浆血红蛋白浓度预测 + 溶血等级分类 - 完整流程
=================================================
基于 RGB 图像层面批次校正 + 43 维颜色特征 + 多模型训练 + 四组独立测试

用法:
    python plasma_pipeline.py [--mode=regression|classification|dual]
                              [--method=scale_mean|...] [--hues=GY,YG,...]
                              [--levels=7,8,9,10] [--cv=stratified|kfold]

模式:
    regression      (默认) 浓度回归 + Stacking 集成
    classification        仅溶血等级四分类 (VotingClassifier)
    dual                  同时训练回归器 + 分类器, 输出双指标对比

依赖:
    cut_plasma_target_region.py      - 血浆区域自动分割
    model_training_v2_1.py           - ConcentrationModelTrainer
    correct_features_in_OneslefBatch_v1.py - FeatureLevelCorrector (可选)
    splitdata_formodel.txt           - 数据划分定义
    sample_list.txt                  - 样本级列表 (exclude 标记)

输出:
    conc_trained_models/             - 训练好的模型
    results/                         - 评估结果图表和 CSV
"""

import os, sys, warnings, glob, argparse, json
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from skimage import color
from scipy.stats import skew
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─────────── 复用模块路径 ───────────
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)  # 优先使用本地副本

from cut_plasma_target_region import get_plasma_region
from model_training_v2_1 import ConcentrationModelTrainer

# ============================================================
# §1 配置
# ============================================================

DATA_DIR  = "/hdd/home/weifeng_ma/05.projects/01.blood_protein_prediction/blood_imag"
# 公开发布包中图像随仓库提供时, 自动使用仓库内 data/blood_imag (可用 --data_dir= 覆盖)
_LOCAL_DATA = os.path.join(CUR_DIR, "data", "blood_imag")
if os.path.isdir(_LOCAL_DATA):
    DATA_DIR = _LOCAL_DATA
SPLIT_FILE = os.path.join(CUR_DIR, "splitdata_formodel.txt")
SAMPLE_FILE = os.path.join(CUR_DIR, "sample_list.txt")
OUTPUT_DIR = os.path.join(CUR_DIR, "results")
MODEL_DIR  = os.path.join(CUR_DIR, "conc_trained_models")

# 批次校正方法: "scale_mean" | "calib_combat" | "rgb_combat" | "none"
BATCH_CORRECT_METHOD = "scale_mean"

# 校准液色调筛选: ["GY","YG","Y","OY","OR","BR"] 全部 / ["OY","OR","BR"] 仅红色系
CALIB_HUES = ["GY","YG","Y","OY","OR","BR"]

# 校准液色级筛选: None 全部 / [7,8,9,10] 仅高色级 / [0.5,1,2,3] 仅低色级
CALIB_LEVELS = [7, 8, 9, 10]  # None = all

# 特征选择: None | "kbest_mi" | "lasso" | "rfecv"

# === 命令行参数覆盖 (用于网格搜索) ===
RUN_MODE = "regression"  # "regression" 回归 | "classification" 仅等级分类 | "dual" 双模型
CV_METHOD = "stratified"  # 默认分层CV
FIXED_FEATURES_NAME = None   # --fixed_features=model_v4 使用官方冻结特征集 (论文可复现)
FEATURES_CSV_INPUT = None    # --features_csv= 跳过图像特征提取, 使用冻结特征快照
import sys as _sys
for _a in _sys.argv[1:]:
    if _a.startswith('--method='):
        BATCH_CORRECT_METHOD = _a.split('=',1)[1]
        print(f'[ARG] 批次校正: {BATCH_CORRECT_METHOD}')
    elif _a.startswith('--hues='):
        v = _a.split('=',1)[1]
        CALIB_HUES = [h.strip() for h in v.split(',')]
        print(f'[ARG] 色调: {CALIB_HUES}')
    elif _a.startswith('--levels='):
        v = _a.split('=',1)[1]
        CALIB_LEVELS = None if v == 'all' else [float(l.strip()) for l in v.split(',')]
        print(f'[ARG] 色级: {CALIB_LEVELS}')
    elif _a.startswith('--level_name='):
        _level_name = _a.split('=',1)[1]  # for logging only
    elif _a.startswith('--rfecv_est='):
        _rfecv_est_name = _a.split('=',1)[1]
        print(f'[ARG] RFECV estimator: {_rfecv_est_name}')
    elif _a.startswith('--cv='):
        CV_METHOD = _a.split('=',1)[1]
        print(f'[ARG] CV方法: {CV_METHOD}')
    elif _a.startswith('--ridge_alpha='):
        RIDGE_ALPHA = float(_a.split('=',1)[1])
        print(f'[ARG] Ridge alpha: {RIDGE_ALPHA}')
    elif _a.startswith('--mode='):
        RUN_MODE = _a.split('=',1)[1].strip().lower()
        if RUN_MODE not in ('regression', 'classification', 'dual'):
            raise ValueError(f"[ERROR] 未知模式: {RUN_MODE} (可选: regression/classification/dual)")
        print(f'[ARG] 模式: {RUN_MODE}')
    elif _a.startswith('--out_dir='):
        OUTPUT_DIR = os.path.join(CUR_DIR, _a.split('=',1)[1])
        print(f'[ARG] 输出目录: {OUTPUT_DIR}')
    elif _a.startswith('--model_dir='):
        MODEL_DIR = os.path.join(CUR_DIR, _a.split('=',1)[1])
        print(f'[ARG] 模型目录: {MODEL_DIR}')
    elif _a.startswith('--fixed_features='):
        FIXED_FEATURES_NAME = _a.split('=',1)[1]
        print(f'[ARG] 固定特征集: {FIXED_FEATURES_NAME}')
    elif _a.startswith('--features_csv='):
        FEATURES_CSV_INPUT = os.path.join(CUR_DIR, _a.split('=',1)[1])
        print(f'[ARG] 预提取特征 CSV: {FEATURES_CSV_INPUT}')
    elif _a.startswith('--data_dir='):
        DATA_DIR = os.path.abspath(_a.split('=',1)[1])
        print(f'[ARG] 数据目录: {DATA_DIR}')

FEATURE_SELECTION_METHOD = "rfecv"
RIDGE_ALPHA = 1.0  # Ridge 正则化强度, 影响 RFECV 特征选择

# A方案 (2026-08-18): 线性后校正 t = k*pred + b, 在内部 289 样本的 5 折 OOF 预测上拟合,
# 修正高浓度系统性低估 (预测压缩)。holdout 与测试集全程不参与拟合。
USE_CALIBRATION = True

# 血浆目标区域: "Auto" (自动分割) 或 (y, x, h, w) 手动坐标
PLASMA_TARGET_REGION = "Auto"

# 随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 校准液目标区域 (y, x, h, w)
CALIB_TARGET_REGION = (850, 1600, 100, 100)

# 特征名称 (43 维)
FEATURE_NAMES = [
    "rgb_r_median","rgb_g_median","rgb_b_median",
    "rgb_r_std","rgb_g_std","rgb_b_std",
    "rgb_r_skew","rgb_g_skew","rgb_b_skew",
    "hsv_h_median","hsv_s_median","hsv_v_median",
    "hsv_h_std","hsv_s_std","hsv_v_std",
    "hsv_h_skew","hsv_s_skew","hsv_v_skew",
    "Lab_L_median","Lab_a_median","Lab_b_median",
    "Lab_L_std","Lab_a_std","Lab_b_std",
    "Lab_L_skew","Lab_a_skew","Lab_b_skew",
    "CIECAM02_J","CIECAM02_M","CIECAM02_h_cam",
    "yuv_y","yuv_u","yuv_v",
    "cmyk_c","cmyk_m","cmyk_y","cmyk_k",
    "rg_ratio_median","rb_ratio_median","bg_ratio_median",
    "orange_intensity","saturation","yellow_component",
]

# model_V4 官方冻结特征集 (2026-08-19 论文可复现性拍板):
# RFECV 在 289 样本上处于平台区 (Lab_L_median 与 CIECAM02_J 的 |coef| 差仅 ~5e-12),
# 图像特征提取的 ~1e-13 浮点噪声即可让两者的淘汰次序翻转 (实测 10 个噪声种子翻转 6 次),
# 导致两次运行的特征集互换 1 个、下游数字微变 (ΔR² ±0.003)。
# 论文模型的特征集冻结为该 15 个 (08-18 官方运行的选择), 保证重跑与发表数字一致。
MODEL_V4_OFFICIAL_FEATURES = [
    "rgb_g_std", "rgb_b_std", "hsv_h_skew", "Lab_L_median", "Lab_a_median",
    "Lab_L_std", "Lab_a_std", "Lab_b_std", "Lab_a_skew", "Lab_b_skew",
    "CIECAM02_M", "yuv_y", "yuv_u", "rg_ratio_median", "orange_intensity",
]


class FixedSelector:
    """固定特征集选择器: 与 sklearn selector 同接口 (transform / get_support)。
    论文可复现性: RFECV 结果冻结后使用, 跳过运行时选择, 避免平台区浮点翻转。"""
    def __init__(self, indices):
        self._indices = np.asarray(list(indices), dtype=int)

    def transform(self, X):
        return X[:, self._indices]

    def get_support(self, indices=False):
        mask = np.zeros(len(FEATURE_NAMES), dtype=bool)
        mask[self._indices] = True
        return np.where(mask)[0] if indices else mask

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"输出目录: {OUTPUT_DIR}")
print(f"模型目录: {MODEL_DIR}")
print(f"运行模式: {RUN_MODE}")
print(f"批次校正: {BATCH_CORRECT_METHOD}")
print(f"特征选择: {FEATURE_SELECTION_METHOD}")

# ============================================================
# §2 颜色特征提取函数
# ============================================================

def extract_YUV_and_cmyk_features(region_rgb):
    """YUV + CMYK 特征 (7维)"""
    yuv = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2YUV)
    yuv_median = np.median(yuv, axis=(0, 1))

    region_float = region_rgb.astype(np.float32) / 255.0
    k = 1 - np.max(region_float, axis=2)
    c = (1 - region_float[:,:,0] - k) / (1 - k + 1e-6)
    m = (1 - region_float[:,:,1] - k) / (1 - k + 1e-6)
    y_cmyk = (1 - region_float[:,:,2] - k) / (1 - k + 1e-6)

    cmyk_medians = [np.median(c), np.median(m), np.median(y_cmyk), np.median(k)]
    return np.concatenate([yuv_median, cmyk_medians])


def extract_color_contrast_features(region_rgb):
    """颜色对比度特征 (3维): R/G, R/B, B/G"""
    r = region_rgb[:,:,0].astype(np.float32)
    g = region_rgb[:,:,1].astype(np.float32)
    b = region_rgb[:,:,2].astype(np.float32)
    eps = 1e-6
    return np.array([
        np.median(np.clip(r / (g + eps), 0, 1000)),
        np.median(np.clip(r / (b + eps), 0, 1000)),
        np.median(np.clip(b / (g + eps), 0, 1000)),
    ])


def extract_plasma_specific_features(region_rgb):
    """血浆特化特征 (3维): orange_intensity, saturation, yellow_component"""
    r, g, b = region_rgb[:,:,0], region_rgb[:,:,1], region_rgb[:,:,2]
    orange_intensity = np.median(r * 0.6 + g * 0.2 + b * 0.2)
    intensity = (r + g + b) / 3.0
    saturation = 1 - (np.minimum(np.minimum(r, g), b) / (intensity + 1e-6))
    avg_saturation = np.median(saturation)
    yellow_component = np.median(g / (r + 1e-6))
    return np.array([orange_intensity, avg_saturation, yellow_component])


def segment_and_extract_feature(image, target_region="Auto"):
    """
    从血浆图像提取 43 维颜色特征。
    输入: image (BGR, numpy array)
    """
    # 1. 选取目标区域
    if target_region == "Auto":
        result = get_plasma_region(image)
        if result is None or result[0] is None:
            # 自动分割失败 → 使用图像中心区域
            h, w = image.shape[:2]
            x, y, ww, hh = int(w*0.2), int(h*0.2), int(w*0.6), int(h*0.6)
            if ww <= 0 or hh <= 0:
                ww, hh = w, h
                x, y = 0, 0
        else:
            _, _, _, x, y, ww, hh = result
    else:
        y, x, hh, ww = target_region

    # 安全检查
    if ww <= 0 or hh <= 0 or y < 0 or x < 0 or y+hh > image.shape[0] or x+ww > image.shape[1]:
        # fallback: use whole image
        y, x, hh, ww = 0, 0, image.shape[0], image.shape[1]

    target_region_image = image[y:y+hh, x:x+ww]
    target_rgb = cv2.cvtColor(target_region_image, cv2.COLOR_BGR2RGB)

    # 2. RGB 特征
    rgb_median = np.median(target_rgb, axis=(0, 1))
    rgb_std = np.std(target_rgb, axis=(0, 1))
    rgb_skew = skew(target_rgb, axis=(0, 1))

    # 3. HSV 特征
    hsv = color.rgb2hsv(target_rgb / 255.0)
    hsv_median = np.median(hsv, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))
    hsv_skew = skew(hsv, axis=(0, 1))

    # 4. Lab 特征
    lab = color.rgb2lab(target_rgb / 255.0)
    lab_median = np.median(lab, axis=(0, 1))
    lab_std = np.std(lab, axis=(0, 1))
    lab_skew = skew(lab, axis=(0, 1))

    # 5. CIECAM02 近似
    J = lab_median[0]
    M = np.sqrt(lab_median[1]**2 + lab_median[2]**2)
    h_cam = np.arctan2(lab_median[2], lab_median[1])

    # 6. 其他特征
    yuv_cmyk = extract_YUV_and_cmyk_features(target_rgb)
    contrast = extract_color_contrast_features(target_rgb)
    plasma = extract_plasma_specific_features(target_rgb)

    # 7. 拼接
    features = np.concatenate([
        rgb_median, rgb_std, rgb_skew,
        hsv_median, hsv_std, hsv_skew,
        lab_median, lab_std, lab_skew,
        [J, M, h_cam],
        yuv_cmyk,
        contrast,
        plasma,
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    return features.reshape(1, -1)


# ============================================================
# §3 批次校正
# ============================================================

def compute_calib_mean_for_batch(data_dir, batch_id, target_region):
    """
    计算单个 batch 的校准图 RGB 均值。
    返回 calib_mean (3,) BGR 格式，或 None。
    """
    calib_dir = os.path.join(data_dir, batch_id, "calibration")
    if not os.path.isdir(calib_dir):
        print(f"  [WARN] {batch_id}: 无 calibration 目录")
        return None

    bgr_vals = []
    hues = sorted(os.listdir(calib_dir))
    for hue in hues:
        hue_dir = os.path.join(calib_dir, hue)
        if not os.path.isdir(hue_dir):
            continue
        if hue not in CALIB_HUES:
            continue
        for img_file in sorted(glob.glob(os.path.join(hue_dir, "*.jpg"))):
            if CALIB_LEVELS is not None:
                fname = os.path.splitext(os.path.basename(img_file))[0]
                try:
                    lvl = float(fname.split('-')[1])
                    if lvl not in CALIB_LEVELS:
                        continue
                except (ValueError, IndexError):
                    pass
            img = cv2.imread(img_file)
            if img is None:
                continue
            y, x, h, w = target_region
            if y+h > img.shape[0] or x+w > img.shape[1]:
                continue
            patch = img[y:y+h, x:x+w]
            bgr_vals.append(np.mean(patch, axis=(0, 1)))  # (B, G, R)

    if len(bgr_vals) == 0:
        print(f"  [WARN] {batch_id}: 无有效校准图像")
        return None

    return np.mean(bgr_vals, axis=0)  # (3,) BGR


def compute_correction_factors(data_dir, train_batch_ids, all_batch_ids, target_region):
    """
    计算所有批次的 RGB 校正因子。
    仅用 train_batch_ids 的校准图计算 reference。
    返回: {batch_id: factor_array (3,) BGR}, reference_RGB (3,)
    """
    print("\n" + "=" * 60)
    print("计算批次校正因子")
    print("=" * 60)

    # Step 1: 每个 batch 计算 calib_mean
    calib_means = {}
    for batch_id in all_batch_ids:
        mean_val = compute_calib_mean_for_batch(data_dir, batch_id, target_region)
        if mean_val is not None:
            calib_means[batch_id] = mean_val
            print(f"  {batch_id}: B={mean_val[0]:.1f} G={mean_val[1]:.1f} R={mean_val[2]:.1f}")
        else:
            print(f"  {batch_id}: FAILED - 将不做校正")

    # Step 2: 仅用 train_set 计算 reference
    train_means = [calib_means[b] for b in train_batch_ids if b in calib_means]
    if len(train_means) == 0:
        print("[ERROR] 无有效训练集校准数据")
        return {}, None

    reference_RGB = np.mean(train_means, axis=0)
    print(f"\n  Reference (train only): B={reference_RGB[0]:.1f} G={reference_RGB[1]:.1f} R={reference_RGB[2]:.1f}")

    # Step 3: 计算每个 batch 的校正因子
    factors = {}
    eps = 1e-6
    for batch_id in all_batch_ids:
        if batch_id in calib_means:
            factors[batch_id] = reference_RGB / (calib_means[batch_id] + eps)
        else:
            factors[batch_id] = np.ones(3)  # 无校准数据 → 不做校正

    return factors, reference_RGB


def compute_rgb_combat_params(data_dir, train_batch_ids, all_batch_ids, target_region):
    """
    RGB 层面的 ComBat: 用校准图每个通道的均值和标准差做 z-score 归一化。
    corrected_pixel = (raw - batch_mean_RGB) / batch_std_RGB × ref_std_RGB + ref_mean_RGB

    返回:
        combat_params: {batch_id: {'mean': (3,), 'std': (3,)}}
        ref_mean, ref_std: (3,)  train 批次的参考值
    """
    print("\n" + "=" * 60)
    print("计算 rgb_combat 校正参数 (RGB层面, 基于校准图像素值)")
    print("=" * 60)

    TARGET = target_region
    batch_pixels = {}  # {batch_id: [所有像素值 (flat), (N_pixels, 3)]}

    for batch_id in all_batch_ids:
        calib_dir = os.path.join(data_dir, batch_id, "calibration")
        if not os.path.isdir(calib_dir):
            print(f"  [WARN] {batch_id}: 无校准目录")
            continue
        pixels = []
        for hue in sorted(os.listdir(calib_dir)):
            if hue not in CALIB_HUES:
                continue
            hue_dir = os.path.join(calib_dir, hue)
            if not os.path.isdir(hue_dir):
                continue
            for img_file in sorted(glob.glob(os.path.join(hue_dir, "*.jpg"))):
                img = cv2.imread(img_file)
                if CALIB_LEVELS is not None:
                    fname = os.path.splitext(os.path.basename(img_file))[0]
                    try:
                        lvl = float(fname.split('-')[1])
                        if lvl not in CALIB_LEVELS:
                            continue
                    except (ValueError, IndexError):
                        pass
                if img is None:
                    continue
                y, x, h, w = TARGET
                if y+h > img.shape[0] or x+w > img.shape[1]:
                    continue
                patch = img[y:y+h, x:x+w].astype(np.float32)  # (100, 100, 3) BGR
                pixels.append(patch.reshape(-1, 3))  # (10000, 3)

        if len(pixels) == 0:
            print(f"  [WARN] {batch_id}: 无有效校准图像")
            continue
        batch_pixels[batch_id] = np.concatenate(pixels, axis=0)  # (66×10000, 3)
        bmean = np.mean(batch_pixels[batch_id], axis=0)
        bstd  = np.std(batch_pixels[batch_id], axis=0)
        print(f"  {batch_id}: mean_BGR=[{bmean[0]:.1f},{bmean[1]:.1f},{bmean[2]:.1f}] std=[{bstd[0]:.1f},{bstd[1]:.1f},{bstd[2]:.1f}]")

    # 仅用 train 计算参考值
    train_pixels = np.concatenate(
        [batch_pixels[b] for b in train_batch_ids if b in batch_pixels], axis=0)
    ref_mean = np.mean(train_pixels, axis=0)  # (3,)
    ref_std  = np.std(train_pixels, axis=0)   # (3,)
    ref_std  = np.clip(ref_std, 1e-6, None)
    print(f"\n  Reference BGR mean: [{ref_mean[0]:.1f},{ref_mean[1]:.1f},{ref_mean[2]:.1f}]")
    print(f"  Reference BGR std:  [{ref_std[0]:.1f},{ref_std[1]:.1f},{ref_std[2]:.1f}]")

    combat_params = {}
    for batch_id in all_batch_ids:
        if batch_id in batch_pixels:
            bp = batch_pixels[batch_id]
            combat_params[batch_id] = {
                'mean': np.mean(bp, axis=0),
                'std':  np.clip(np.std(bp, axis=0), 1e-6, None),
            }
        else:
            combat_params[batch_id] = None

    return combat_params, ref_mean, ref_std


def apply_rgb_combat(img_bgr, batch_mean, batch_std, ref_mean, ref_std):
    """对图像应用 RGB 层面的 ComBat 校正"""
    img_f = img_bgr.astype(np.float32)
    corrected = (img_f - batch_mean) / batch_std * ref_std + ref_mean
    return np.clip(corrected, 0, 255).astype(np.uint8)


def compute_calib_combat_params(data_dir, train_batch_ids, all_batch_ids):
    """
    用校准图像提取 43 维特征，计算 per-batch 的 mean/std 校正参数 (ComBat 风格)。

    返回:
        combat_params: {batch_id: {'mean': (43,), 'std': (43,)}}
        ref_mean: (43,)  train 批次的校准特征均值
        ref_std:  (43,)  train 批次的校准特征 std
    """
    print("\n" + "=" * 60)
    print("计算 calib_combat 校正参数 (基于校准图特征)")
    print("=" * 60)

    # Step 1: 每个 batch 提取校准图特征
    TARGET = (850, 1600, 100, 100)
    batch_calib_feats = {}  # {batch_id: (N_calib, 43)}

    for batch_id in all_batch_ids:
        calib_dir = os.path.join(data_dir, batch_id, "calibration")
        if not os.path.isdir(calib_dir):
            print(f"  [WARN] {batch_id}: 无 calibration 目录")
            continue

        feats = []
        for hue in sorted(os.listdir(calib_dir)):
            if hue not in CALIB_HUES:
                continue
            hue_dir = os.path.join(calib_dir, hue)
            if not os.path.isdir(hue_dir):
                continue
            for img_file in sorted(glob.glob(os.path.join(hue_dir, "*.jpg"))):
                img = cv2.imread(img_file)
                if img is None:
                    continue
                y, x, h, w = TARGET
                if y+h > img.shape[0] or x+w > img.shape[1]:
                    continue
                try:
                    f = segment_and_extract_feature(img, target_region=TARGET)
                    feats.append(f.flatten())
                except Exception:
                    pass

        if len(feats) == 0:
            print(f"  [WARN] {batch_id}: 无有效校准特征")
            continue
        feats_arr = np.array(feats)  # (N, 43)
        # 过滤含 NaN 的特征
        valid = ~np.any(np.isnan(feats_arr), axis=1)
        if valid.sum() < len(feats_arr):
            print(f"    [WARN] {batch_id}: {len(feats_arr) - valid.sum()} 张校准图含 NaN 已排除")
        batch_calib_feats[batch_id] = feats_arr[valid]
        print(f"  {batch_id}: {valid.sum()} 张有效校准图")

    # Step 2: 仅用 train 批次计算 reference mean/std
    train_feats = np.concatenate(
        [batch_calib_feats[b] for b in train_batch_ids if b in batch_calib_feats], axis=0)
    ref_mean = np.mean(train_feats, axis=0)  # (43,)
    ref_std  = np.std(train_feats, axis=0)   # (43,)
    ref_std = np.clip(ref_std, 1e-6, None)   # 防止零除
    print(f"\n  Reference mean (43维): [{ref_mean[0]:.1f} ... {ref_mean[-1]:.3f}]")
    print(f"  Reference std  (43维): [{ref_std[0]:.1f} ... {ref_std[-1]:.3f}]")

    # Step 3: 每个 batch 计算 mean/std
    combat_params = {}
    eps = 1e-6
    for batch_id in all_batch_ids:
        if batch_id in batch_calib_feats:
            bf = batch_calib_feats[batch_id]
            bmean = np.mean(bf, axis=0)
            bstd  = np.std(bf, axis=0)
            bstd  = np.clip(bstd, 1e-6, None)
            combat_params[batch_id] = {'mean': bmean, 'std': bstd}
        else:
            combat_params[batch_id] = None  # 无数据，不做校正

    return combat_params, ref_mean, ref_std


def apply_calib_combat(features, batch_ids, combat_params, ref_mean, ref_std):
    """
    对特征矩阵应用 ComBat 风格校正: (x - mean_batch) / std_batch * ref_std + ref_mean
    """
    corrected = features.copy()
    for bid in np.unique(batch_ids):
        mask = batch_ids == bid
        if combat_params.get(bid) is None:
            continue
        bmean = combat_params[bid]['mean']
        bstd  = combat_params[bid]['std']
        corrected[mask] = (features[mask] - bmean) / bstd * ref_std + ref_mean
    return corrected


def apply_correction(img_bgr, factor):
    """对图像应用 RGB 校正: corrected = raw * factor (逐像素)"""
    if np.allclose(factor, 1.0):
        return img_bgr
    corrected = img_bgr.astype(np.float32) * factor.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


# ============================================================
# §4 数据加载
# ============================================================

def load_plasma_data(data_dir, sample_df, correction_factors,
                     combat_params=None, ref_mean=None, ref_std=None,
                     rgb_combat_params=None, rgb_ref_mean=None, rgb_ref_std=None):
    """
    加载血浆图像，应用 RGB 校正，提取特征，可选特征级/像素级校正。

    Args:
        data_dir: 数据根目录
        sample_df: sample_list.txt 的 DataFrame (只含 exclude==0 的行)
        correction_factors: {batch_id: factor_array}  (scale_mean)
        combat_params: calib_combat 参数 (特征级)
        ref_mean, ref_std: calib_combat 参考值 (特征级)
        rgb_combat_params: rgb_combat 参数 {batch_id: {'mean':(3,),'std':(3,)}}
        rgb_ref_mean, rgb_ref_std: rgb_combat 参考值 (3,)

    Returns:
        features (N, 43), concentrations (N,), batch_ids (N,), sample_names (N,)
    """
    print("\n" + "=" * 60)
    print("加载血浆数据")
    print("=" * 60)
    if rgb_combat_params is not None:
        print("  RGB 校正: rgb_combat (像素级 z-score)")
    if combat_params is not None:
        print("  特征级校正: calib_combat")

    all_feats, all_conc, all_bids, all_snames = [], [], [], []
    failed = []
    batch_counts = {}

    for _, row in sample_df.iterrows():
        batch_id = row['BatchID']
        sample_id = str(int(row['SampleID']))
        conc = row['Concentration']
        sample_name = row['SampleName']

        plasma_dir = os.path.join(data_dir, batch_id, "plasma", sample_id)
        jpg_files = glob.glob(os.path.join(plasma_dir, "*.jpg"))
        if len(jpg_files) == 0:
            failed.append((sample_name, "无JPG图像"))
            continue
        img_path = jpg_files[0]

        img = cv2.imread(img_path)
        if img is None:
            failed.append((sample_name, "无法读取图像"))
            continue

        # RGB 校正
        if rgb_combat_params is not None and rgb_combat_params.get(batch_id) is not None:
            p = rgb_combat_params[batch_id]
            img = apply_rgb_combat(img, p['mean'], p['std'], rgb_ref_mean, rgb_ref_std)
        else:
            factor = correction_factors.get(batch_id, np.ones(3))
            img = apply_correction(img, factor)

        # 特征提取
        try:
            feats = segment_and_extract_feature(img, PLASMA_TARGET_REGION)
        except Exception as e:
            failed.append((sample_name, f"特征提取失败: {e}"))
            continue

        f = feats.flatten()
        if np.any(np.isnan(f)):
            failed.append((sample_name, "特征含NaN"))
            continue
        all_feats.append(f)
        all_conc.append(conc)
        all_bids.append(batch_id)
        all_snames.append(sample_name)
        batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1

    features = np.array(all_feats)  # (N, 43)
    concentrations = np.array(all_conc)
    batch_ids = np.array(all_bids)
    sample_names = np.array(all_snames)

    # calib_combat 特征级校正 (ComBat 风格, 基于校准图特征)
    if combat_params is not None:
        features = apply_calib_combat(features, batch_ids, combat_params, ref_mean, ref_std)
        print("\n  [calib_combat] 特征级校正已应用")

    print(f"\n  成功加载: {len(features)} 样本")
    print(f"  失败: {len(failed)} 样本")
    if failed:
        for name, reason in failed[:10]:
            print(f"    {name}: {reason}")
        if len(failed) > 10:
            print(f"    ... 共 {len(failed)} 个失败")

    print(f"\n  各批次样本数:")
    for bid in sorted(batch_counts.keys()):
        print(f"    {bid}: {batch_counts[bid]}")

    return features, concentrations, batch_ids, sample_names


# ============================================================
# §5 数据划分
# ============================================================

def split_by_config(features, concentrations, batch_ids, sample_names, sample_df):
    """
    按 splitData 字段划分 train / test_set1~4。
    返回 dict: {"train": (X, y, names), "test_set1": (...), ...}
    """
    split_map = {}
    for _, row in sample_df.iterrows():
        split_map[row['SampleName']] = row['splitData']

    splits = {}
    for split_name in ['train_set', 'test_set1', 'test_set2', 'test_set3', 'test_set4']:
        mask = np.array([split_map.get(n) == split_name for n in sample_names])
        if mask.sum() == 0:
            print(f"  [WARN] {split_name}: 无样本!")
            continue
        splits[split_name] = (
            features[mask],
            concentrations[mask],
            sample_names[mask],
            batch_ids[mask],
        )

    print(f"\n数据划分:")
    for k, (X, y, _, _) in splits.items():
        print(f"  {k}: {len(X)} 样本, conc range [{y.min():.1f}, {y.max():.1f}]")

    return splits


# ============================================================
# §6 评估
# ============================================================

def get_hemolysis_grade(v):
    if v <= 20: return "None"
    elif v <= 50: return "Mild"
    elif v <= 100: return "Moderate"
    else: return "Severe"


GRADE_NAMES = ['None', 'Mild', 'Moderate', 'Severe']

def get_grade_int(v):
    """浓度 → 溶血等级整数标签 (0=None, 1=Mild, 2=Moderate, 3=Severe)"""
    if v <= 20: return 0
    elif v <= 50: return 1
    elif v <= 100: return 2
    else: return 3


def evaluate_split(model, selector, X, y, sample_names, batch_ids, split_name, output_dir, calib=None):
    """对单个测试集进行评估，生成图表和CSV
    calib: (k, b) 线性后校正参数, None 表示不校正
    """
    print(f"\n{'='*60}")
    print(f"评估: {split_name}")
    print(f"{'='*60}")

    # 特征选择
    if selector is not None:
        X_sel = selector.transform(X)
    else:
        X_sel = X

    # 预测
    y_pred = model.predict(X_sel)
    y_pred = np.clip(y_pred, 0, None)
    y_pred_raw = y_pred.copy()
    if calib is not None:
        if isinstance(calib, dict) and calib.get('type') == 'piecewise':
            _bp = calib['breakpoint']
            _lo = y_pred < _bp
            y_pred[_lo] = calib['low']['k'] * y_pred[_lo] + calib['low']['b']
            y_pred[~_lo] = calib['high']['k'] * y_pred[~_lo] + calib['high']['b']
        else:
            y_pred = calib[0] * y_pred + calib[1]
        y_pred = np.clip(y_pred, 0, None)

    # 整体指标
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²: {r2:.4f}")

    # Per-batch 指标
    print(f"\n  Per-batch 指标:")
    for bid in np.unique(batch_ids):
        m = batch_ids == bid
        if m.sum() < 3:
            print(f"    {bid}: 样本太少 ({m.sum()})")
            continue
        b_r2 = r2_score(y[m], y_pred[m])
        b_rmse = np.sqrt(mean_squared_error(y[m], y_pred[m]))
        print(f"    {bid}: R²={b_r2:.4f}, RMSE={b_rmse:.3f}, n={m.sum()}")

    # 溶血等级评估
    y_true_grade = [get_hemolysis_grade(v) for v in y]
    y_pred_grade = [get_hemolysis_grade(v) for v in y_pred]
    label_order = ["None", "Mild", "Moderate", "Severe"]
    cm = confusion_matrix(y_true_grade, y_pred_grade, labels=label_order)
    acc = accuracy_score(y_true_grade, y_pred_grade)

    # 二分类溶血判定 (阈值 50 mg/dL): 论文报告口径 (2026-08-18 改, 文献可见溶血界值 50 mg/dL)
    yt_bin = (y > 50.0).astype(int)
    yp_bin = (y_pred > 50.0).astype(int)
    acc_bin = accuracy_score(yt_bin, yp_bin)
    print(f"\n  溶血等级准确率(4级): {acc:.2%}   |   溶血判定准确率(>50 mg/dL): {acc_bin:.2%}")

    # 散点图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    max_val = max(y.max(), y_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5)
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(batch_ids))))
    for i, bid in enumerate(np.unique(batch_ids)):
        m = batch_ids == bid
        ax1.scatter(y[m], y_pred[m], c=[colors[i]], label=bid.split('_')[-1], alpha=0.7, s=30)
    ax1.set_xlabel('True Concentration')
    ax1.set_ylabel('Predicted Concentration')
    ax1.set_title(f'{split_name}: R²={r2:.3f}, RMSE={rmse:.1f}')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_order)
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title(f'Hemolysis Grade (Acc={acc:.2%})')

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"eval_{split_name}.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fpath}")

    # CSV
    result_df = pd.DataFrame({
        'SampleName': sample_names,
        'BatchID': batch_ids,
        'conc_true': y,
        'conc_pred': y_pred,
        'conc_pred_raw': y_pred_raw,
        'true_grade': y_true_grade,
        'pred_grade': y_pred_grade,
    })
    csv_path = os.path.join(output_dir, f"predictions_{split_name}.csv")
    result_df.to_csv(csv_path, index=False, sep='\t')
    print(f"  CSV已保存: {csv_path}")

    return {'split': split_name, 'R²': r2, 'RMSE': rmse, 'Accuracy': acc_bin, 'Accuracy4': acc, 'n': len(y)}


# ============================================================
# §6-v2 分类器训练与评估
# ============================================================

def train_hemolysis_classifier(X_df, y_conc, selected_indices, output_dir,
                               cv_method="stratified", seed=RANDOM_SEED):
    """
    训练溶血等级分类器 (VotingClassifier: RF + XGBoost + MLP)。

    参数:
        X_df: 原始 43 维特征 DataFrame (index=sample_names)
        y_conc: 浓度值数组 (用于生成等级标签和分层)
        selected_indices: 回归 RFECV 选出的特征索引 (复用, 不做分类专用 RFECV)
        output_dir: 模型保存目录

    返回:
        classifier: 训练好的 VotingClassifier
        clf_selector: 特征选择器 (基于 selected_indices 的简单 selector)
        meta: dict, 含 holdout 指标和特征列表
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report
    from xgboost import XGBClassifier
    import joblib

    y_conc = np.asarray(y_conc, dtype=float)
    y_grade = np.array([get_grade_int(v) for v in y_conc])
    conc_bins = pd.cut(y_conc, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)

    sel_names = [FEATURE_NAMES[i] for i in selected_indices]
    print(f"\n{'='*60}")
    print(f"训练溶血等级分类器")
    print(f"{'='*60}")
    print(f"  复用回归特征: {len(sel_names)} features")
    print(f"  {sel_names}")
    for g in range(4):
        print(f"  {GRADE_NAMES[g]}: {(y_grade == g).sum()}")

    # ── 简单 selector: 基于回归选出的特征索引 ──
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso
    class SimpleSelector:
        """基于固定特征索引的选择器, 兼容 transform 接口"""
        def __init__(self, indices):
            self.indices_ = indices
        def transform(self, X):
            return X[:, self.indices_]
        def get_support(self, indices=False):
            mask = np.zeros(X_df.shape[1], dtype=bool)
            mask[list(self.indices_)] = True
            if indices:
                return np.where(mask)[0]
            return mask
    clf_selector = SimpleSelector(selected_indices)

    # ── 80/20 holdout (与回归相同的分层策略) ──
    X_raw = X_df.values.astype(float)
    _X_idx = X_df.index.to_numpy()
    X_idx_tr, X_idx_ho, yg_tr, yg_ho, yc_tr, yc_ho, idx_tr, idx_ho = train_test_split(
        X_raw, y_grade, y_conc, _X_idx,
        test_size=0.2, random_state=seed, stratify=conc_bins)

    X_tr = clf_selector.transform(X_idx_tr)
    X_ho = clf_selector.transform(X_idx_ho)

    print(f"\n  Train: {len(X_tr)}, Holdout: {len(X_ho)}")
    print(f"  Holdout grade dist: {np.bincount(yg_ho)}")

    # ── CV folds ──
    tr_bins = pd.cut(yc_tr, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
    if cv_method == "stratified":
        _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr_cv = list(_skf.split(X_tr, tr_bins))
    else:
        from sklearn.model_selection import KFold
        tr_cv = list(KFold(n_splits=5, shuffle=False).split(X_tr))

    # ── RF Classifier ──
    rf_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=seed))])
    rf_clf = GridSearchCV(rf_clf,
        {'clf__n_estimators': [50, 100], 'clf__max_depth': [5, 10, None]},
        cv=tr_cv, scoring='accuracy', n_jobs=-1).fit(X_tr, yg_tr).best_estimator_
    print(f"  RF best: {rf_clf.named_steps['clf'].get_params()}")

    # ── XGBoost Classifier ──
    xgb_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(random_state=seed, eval_metric='mlogloss'))])
    xgb_clf = GridSearchCV(xgb_clf,
        {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]},
        cv=tr_cv, scoring='accuracy', n_jobs=-1).fit(X_tr, yg_tr).best_estimator_

    # ── MLP Classifier ──
    mlp_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=5000, random_state=seed))])
    mlp_clf = GridSearchCV(mlp_clf,
        {'clf__hidden_layer_sizes': [(50,), (100,)], 'clf__alpha': [0.0001, 0.001]},
        cv=tr_cv, scoring='accuracy', n_jobs=-1).fit(X_tr, yg_tr).best_estimator_

    # ── Voting: soft voting, 权重 = 1/holdout_accuracy ──
    model_map = [('rf', rf_clf), ('xgb', xgb_clf), ('mlp', mlp_clf)]
    accs = {}
    for name, m in model_map:
        accs[name] = accuracy_score(yg_ho, m.predict(X_ho))
        print(f"  {name} holdout acc: {accs[name]:.4f}")
    inv = {n: 1.0 / max(a, 0.01) for n, a in accs.items()}
    total = sum(inv.values())
    weights = [inv[n] / total for n in ['rf', 'xgb', 'mlp']]

    voting = VotingClassifier(
        estimators=[('rf', rf_clf), ('xgb', xgb_clf), ('mlp', mlp_clf)],
        voting='soft', weights=weights)
    voting.fit(X_tr, yg_tr)
    ho_acc = accuracy_score(yg_ho, voting.predict(X_ho))
    print(f"  Voting holdout acc: {ho_acc:.4f}  weights: {dict(zip(['rf','xgb','mlp'], [f'{w:.3f}' for w in weights]))}")

    # ── 保存模型 ──
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(voting, os.path.join(output_dir, 'hemolysis_classifier.pkl'))
    joblib.dump(rf_clf, os.path.join(output_dir, 'hemolysis_rf_clf.pkl'))
    joblib.dump(xgb_clf, os.path.join(output_dir, 'hemolysis_xgb_clf.pkl'))
    joblib.dump(mlp_clf, os.path.join(output_dir, 'hemolysis_mlp_clf.pkl'))
    print(f"  模型已保存到: {output_dir}")

    meta = {
        'holdout_acc': ho_acc,
        'per_model_accs': accs,
        'weights': dict(zip(['rf', 'xgb', 'mlp'], weights)),
        'selected_features': sel_names,
        'n_features': len(selected_indices),
        'holdout_ids': idx_ho,
    }
    return voting, clf_selector, meta


def evaluate_classifier_split(clf, clf_selector, X, y, sample_names, batch_ids,
                              split_name, output_dir):
    """对单个测试集评估分类器: accuracy, confusion matrix, per-grade P/R/F1"""
    print(f"\n{'='*60}")
    print(f"分类器评估: {split_name}")
    print(f"{'='*60}")

    if clf_selector is not None:
        X_sel = clf_selector.transform(X)
    else:
        X_sel = X

    y_true = np.array([get_grade_int(v) for v in y])
    y_pred = clf.predict(X_sel)
    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy: {acc:.4f}")

    # Per-batch accuracy
    print(f"\n  Per-batch 指标:")
    for bid in np.unique(batch_ids):
        m = batch_ids == bid
        if m.sum() < 3:
            print(f"    {bid}: 样本太少 ({m.sum()})")
            continue
        b_acc = accuracy_score(y_true[m], y_pred[m])
        print(f"    {bid}: Acc={b_acc:.4f}, n={m.sum()}")

    # Classification report
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=GRADE_NAMES, labels=[0, 1, 2, 3], zero_division=0))

    # Confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GRADE_NAMES)
    disp.plot(ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title(f'{split_name} Classifier (Acc={acc:.2%})')

    # Per-grade P/R bar chart
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    x = np.arange(4)
    w = 0.25
    ax2.bar(x - w, p, w, label='Precision', color='#2196F3')
    ax2.bar(x, r, w, label='Recall', color='#4CAF50')
    ax2.bar(x + w, f1, w, label='F1', color='#FF9800')
    ax2.set_xticks(x)
    ax2.set_xticklabels(GRADE_NAMES)
    ax2.set_ylabel('Score')
    ax2.set_title(f'{split_name} Per-Grade Metrics')
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"eval_clf_{split_name}.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {fpath}")

    # CSV
    result_df = pd.DataFrame({
        'SampleName': sample_names,
        'BatchID': batch_ids,
        'conc_true': y,
        'true_grade': [GRADE_NAMES[g] for g in y_true],
        'pred_grade': [GRADE_NAMES[g] for g in y_pred],
    })
    csv_path = os.path.join(output_dir, f"predictions_clf_{split_name}.csv")
    result_df.to_csv(csv_path, index=False, sep='\t')
    print(f"  CSV已保存: {csv_path}")

    return {
        'split': split_name,
        'Accuracy': acc,
        'Precision_macro': np.mean(p),
        'Recall_macro': np.mean(r),
        'F1_macro': np.mean(f1),
        'n': len(y),
    }


def evaluate_dual_split(model, clf, selector, clf_selector,
                         X, y, sample_names, batch_ids, split_name, output_dir):
    """dual 模式: 同时评估回归器 + 分类器, 输出合并结果"""
    # 回归评估
    res_reg = evaluate_split(model, selector, X, y, sample_names, batch_ids,
                              split_name, output_dir)
    # 分类器评估
    res_clf = evaluate_classifier_split(clf, clf_selector, X, y, sample_names, batch_ids,
                                         split_name, output_dir)

    # 合并 CSV (各自应用各自的 feature selector)
    X_reg = selector.transform(X) if selector is not None else X
    X_clf = clf_selector.transform(X) if clf_selector is not None else X
    yp_reg = np.clip(model.predict(X_reg), 0, None)
    yp_clf = clf.predict(X_clf)
    y_true_grade = np.array([get_grade_int(v) for v in y])

    dual_df = pd.DataFrame({
        'SampleName': sample_names,
        'BatchID': batch_ids,
        'conc_true': y,
        'conc_pred': yp_reg,
        'true_grade': [GRADE_NAMES[g] for g in y_true_grade],
        'reg_grade': [GRADE_NAMES[get_grade_int(v)] for v in yp_reg],
        'pred_grade_clf': [GRADE_NAMES[g] for g in yp_clf],
    })
    csv_path = os.path.join(output_dir, f"predictions_dual_{split_name}.csv")
    dual_df.to_csv(csv_path, index=False, sep='\t')
    print(f"  Dual CSV已保存: {csv_path}")

    # 4 宫格图: 回归散点 | 回归 CM | 分类器 CM | 分类器 P/R
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # (0,0) 回归散点
    max_val = max(y.max(), yp_reg.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.5)
    axes[0, 0].scatter(y, yp_reg, alpha=0.6, s=20)
    axes[0, 0].set_xlabel('True'); axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title(f'{split_name} Regression: R²={res_reg["R²"]:.3f}, RMSE={res_reg["RMSE"]:.1f}')
    axes[0, 0].grid(True, alpha=0.3)

    # (0,1) 回归 CM
    cm_reg = confusion_matrix(
        [get_hemolysis_grade(v) for v in y],
        [get_hemolysis_grade(v) for v in yp_reg],
        labels=["None", "Mild", "Moderate", "Severe"])
    ConfusionMatrixDisplay(cm_reg, display_labels=["None","Mild","Moderate","Severe"]).plot(
        ax=axes[0, 1], cmap='Blues', values_format='d')
    axes[0, 1].set_title(f'Reg Grade (Acc={res_reg["Accuracy"]:.2%})')

    # (1,0) 分类器 CM
    cm_clf = confusion_matrix(y_true_grade, yp_clf, labels=[0, 1, 2, 3])
    ConfusionMatrixDisplay(cm_clf, display_labels=GRADE_NAMES).plot(
        ax=axes[1, 0], cmap='Blues', values_format='d')
    axes[1, 0].set_title(f'Classifier (Acc={res_clf["Accuracy"]:.2%})')

    # (1,1) 对比柱状图
    x = np.arange(4)
    w = 0.35
    from sklearn.metrics import precision_recall_fscore_support as prfs
    p_reg, r_reg, f1_reg, _ = prfs(
        [get_grade_int(v) for v in y],
        [get_grade_int(v) for v in yp_reg],
        labels=[0, 1, 2, 3], zero_division=0)
    p_clf, r_clf, f1_clf, _ = prfs(y_true_grade, yp_clf, labels=[0, 1, 2, 3], zero_division=0)
    axes[1, 1].bar(x - w/2, f1_reg, w, label='Reg F1', color='#2196F3')
    axes[1, 1].bar(x + w/2, f1_clf, w, label='Clf F1', color='#FF9800')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(GRADE_NAMES)
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Per-Grade F1: Regression vs Classifier')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.05)

    plt.tight_layout()
    fpath = os.path.join(output_dir, f"eval_dual_{split_name}.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Dual图表已保存: {fpath}")

    return {
        'split': split_name, 'n': len(y),
        'R²': res_reg['R²'], 'RMSE': res_reg['RMSE'],
        'Reg_Acc': res_reg['Accuracy'], 'Clf_Acc': res_clf['Accuracy'],
        'Clf_P': res_clf['Precision_macro'], 'Clf_R': res_clf['Recall_macro'],
        'Clf_F1': res_clf['F1_macro'],
    }


# ============================================================
# §7 主流程
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"开始时间: {timestamp}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ── 7.1 读取数据划分配置 ──
    split_config = pd.read_csv(SPLIT_FILE, sep="\t")
    sample_list = pd.read_csv(SAMPLE_FILE, sep="\t")

    # 只取可用样本
    usable = sample_list[sample_list['exclude'] == 0].copy()
    print(f"\n可用样本: {len(usable)} (排除 {len(sample_list) - len(usable)} 个)")

    train_batch_ids = split_config[split_config['splitData'] == 'train_set']['BatchID-lab'].tolist()
    all_batch_ids = sorted(set(usable['BatchID'].tolist()))
    print(f"训练批次: {len(train_batch_ids)} 个")
    print(f"全部批次: {len(all_batch_ids)} 个")

    # ── 7.2 计算校正因子 ──
    combat_params, ref_mean, ref_std = None, None, None
    rgb_combat_params, rgb_ref_mean, rgb_ref_std = None, None, None

    if BATCH_CORRECT_METHOD == "none":
        correction_factors = {b: np.ones(3) for b in all_batch_ids}
        reference_RGB = None
    elif BATCH_CORRECT_METHOD == "scale_mean":
        correction_factors, reference_RGB = compute_correction_factors(
            DATA_DIR, train_batch_ids, all_batch_ids, CALIB_TARGET_REGION)
    elif BATCH_CORRECT_METHOD == "calib_combat":
        correction_factors = {b: np.ones(3) for b in all_batch_ids}
        combat_params, ref_mean, ref_std = compute_calib_combat_params(
            DATA_DIR, train_batch_ids, all_batch_ids)
    elif BATCH_CORRECT_METHOD == "rgb_combat":
        correction_factors = {b: np.ones(3) for b in all_batch_ids}
        rgb_combat_params, rgb_ref_mean, rgb_ref_std = compute_rgb_combat_params(
            DATA_DIR, train_batch_ids, all_batch_ids, CALIB_TARGET_REGION)
    else:
        raise ValueError(f"未知校正方法: {BATCH_CORRECT_METHOD}")

    # 保存校正因子
    factors_df = pd.DataFrame([
        {'BatchID': b, 'factor_B': f[0], 'factor_G': f[1], 'factor_R': f[2]}
        for b, f in correction_factors.items()
    ])
    factors_df.to_csv(os.path.join(OUTPUT_DIR, "correction_factors.csv"), index=False)
    print(f"\n校正因子已保存: correction_factors.csv")

    # ── 7.3 加载血浆数据 ──
    if FEATURES_CSV_INPUT is not None:
        # 论文可复现模式: 跳过图像特征提取, 使用冻结的特征快照 (按 sample_list 顺序对齐)
        _feat_in = pd.read_csv(FEATURES_CSV_INPUT, sep='\t')
        _feat_in = _feat_in.set_index('SampleName').reindex(usable['SampleName'].values)
        assert not _feat_in[FEATURE_NAMES].isna().any().any(), '特征快照存在缺失值'
        features = _feat_in[FEATURE_NAMES].values.astype(float)
        concentrations = _feat_in['Concentration'].values.astype(float)
        batch_ids = usable['BatchID'].values
        sample_names = usable['SampleName'].values
        print(f"[可复现模式] 特征从 {FEATURES_CSV_INPUT} 加载: {features.shape}")
    else:
        features, concentrations, batch_ids, sample_names = load_plasma_data(
            DATA_DIR, usable, correction_factors,
            combat_params=combat_params, ref_mean=ref_mean, ref_std=ref_std,
            rgb_combat_params=rgb_combat_params, rgb_ref_mean=rgb_ref_mean, rgb_ref_std=rgb_ref_std)

    # 保存特征矩阵
    feat_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    feat_df['SampleName'] = sample_names
    feat_df['BatchID'] = batch_ids
    feat_df['Concentration'] = concentrations
    feat_df.to_csv(os.path.join(OUTPUT_DIR, f"features_{timestamp}.csv"), index=False, sep='\t')
    print(f"\n特征矩阵已保存: features_{timestamp}.csv ({features.shape})")

    # ── 7.4 划分训练/测试 ──
    splits = split_by_config(features, concentrations, batch_ids, sample_names, usable)

    if 'train_set' not in splits:
        print("[ERROR] 无训练集!")
        return

    X_train, y_train, names_train, bids_train = splits['train_set']

    # ── 7.5 特征选择 ──
    # 两路各自独立: 回归用 RFECV(Ridge, r2), 分类用 RFECV(RandomForestClassifier, accuracy)
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.svm import LinearSVR
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier as RFC_feat
    from sklearn.feature_selection import SelectFromModel, RFECV

    selector = None          # 回归专用 selector
    selected_indices = None  # 回归专用特征索引
    clf_selector = None      # 分类器专用 selector

    # ── 回归特征选择 (所有 mode 都需要, 分类器复用) ──
    if FEATURE_SELECTION_METHOD is not None:
        print(f"\n{'='*60}")
        print(f"[回归] 特征选择: {FEATURE_SELECTION_METHOD}")
        print(f"{'='*60}")

        # B方案 (2026-08-17): 特征选择排除内部 holdout。
        # 先按 trainer 相同参数拆 80/20, 选择只用内部 289 个样本,
        # holdout (73) 全程不参与选择, 避免选择泄漏
        # (旧做法在全 362 上选, holdout R² 被高估约 +0.008)。
        from sklearn.model_selection import train_test_split
        _sel_bins = pd.cut(y_train, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
        X_sel, _, y_sel, _ = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=_sel_bins)
        print(f"  特征选择样本: {len(y_sel)} (内部 80%, holdout 已排除)")

        if FIXED_FEATURES_NAME is not None:
            # 论文可复现模式: 跳过 RFECV 运行时选择, 使用冻结特征集
            if FIXED_FEATURES_NAME == 'model_v4':
                _fixed_names = MODEL_V4_OFFICIAL_FEATURES
            else:
                _fixed_names = [n.strip() for n in FIXED_FEATURES_NAME.split(',')]
            _fixed_idx = [FEATURE_NAMES.index(n) for n in _fixed_names]
            selector = FixedSelector(_fixed_idx)
            print(f"  [固定特征] 跳过 RFECV, 使用冻结特征集 ({len(_fixed_idx)} 个): {_fixed_names}")
        elif FEATURE_SELECTION_METHOD == 'kbest_mi':
            selector = SelectKBest(mutual_info_regression, k=27)
            selector.fit(X_sel, y_sel)
        elif FEATURE_SELECTION_METHOD == 'kbest_f':
            from sklearn.feature_selection import f_regression
            selector = SelectKBest(f_regression, k=27)
            selector.fit(X_sel, y_sel)
        elif FEATURE_SELECTION_METHOD == 'lasso':
            lasso = Lasso(alpha=0.01, max_iter=50000, random_state=RANDOM_SEED)
            lasso.fit(X_sel, y_sel)
            selector = SelectFromModel(lasso, prefit=True, threshold='mean')
        elif FEATURE_SELECTION_METHOD == 'rfecv':
            est_map = {
                'ridge': Ridge(alpha=RIDGE_ALPHA),
                'lasso': Lasso(alpha=0.01, max_iter=5000),
                'elasticnet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
                'linearsvr': LinearSVR(C=1.0, max_iter=5000, dual='auto'),
                'rf': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
                'et': ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
            }
            _est = est_map.get(globals().get('_rfecv_est_name', 'ridge'), Ridge(alpha=1.0))
            if CV_METHOD == "stratified":
                from sklearn.model_selection import StratifiedKFold
                _conc_bins = pd.cut(y_sel, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
                _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
                _rfecv_cv = list(_skf.split(X_sel, _conc_bins))
            else:
                from sklearn.model_selection import KFold
                _rfecv_cv = KFold(n_splits=5, shuffle=False)
            selector = RFECV(
                estimator=_est,
                cv=_rfecv_cv, scoring='r2',
                min_features_to_select=5,
                n_jobs=-1,
            )
            selector.fit(X_sel, y_sel)

        selected_indices = selector.get_support(indices=True)
        selected_names = [FEATURE_NAMES[i] for i in selected_indices]
        print(f"  原始特征: {len(FEATURE_NAMES)}")
        print(f"  选择特征: {len(selected_indices)}")
        print(f"  选择的特征: {selected_names}")

    # ── 7.6 训练模型 ──
    model = None
    classifier = None
    clf_selector = None
    calibration = None

    # ── 回归训练 (regression / dual) ──
    if RUN_MODE in ('regression', 'dual'):
        # 应用回归特征选择
        if selector is not None:
            X_train_reg = selector.transform(X_train)
        else:
            X_train_reg = X_train

        n_feat = X_train_reg.shape[1]
        print(f"\n{'='*60}")
        print(f"[回归] 训练模型 (特征数: {n_feat})")
        print(f"{'='*60}")

        reg_sel_names = [FEATURE_NAMES[i] for i in selected_indices] if selected_indices is not None else FEATURE_NAMES
        X_train_df = pd.DataFrame(X_train_reg, columns=reg_sel_names, index=names_train)

        trainer = ConcentrationModelTrainer(output_dir=MODEL_DIR)
        results = trainer.train_models(
            X_train_df, y_train,
            true_hemolysis_grade=[get_hemolysis_grade(v) for v in y_train],
            extended_models=True,
            feature_selection_method=None,
            cv_method=CV_METHOD
        )

        # 最优模型按 5 折 CV RMSE 选择 (2026-08-18 审计: 不用 holdout, holdout 仅做最终验证)
        best_result = min(results.items(), key=lambda x: x[1]['cv_rmse'])
        best_model_name = best_result[0]
        print(f"\n最佳回归模型: {best_model_name}, CV RMSE={best_result[1]['cv_rmse']:.4f}, "
              f"holdout RMSE={best_result[1]['test_rmse']:.4f}, R²={best_result[1]['test_r2']:.4f}")

        model = trainer.load_model(best_model_name)

        # ── 校正: model_V4 分段线性 E2 (嵌套 OOF, holdout/test 不参与拟合) ──
        # 高段 = v3 全局线 (全 OOF 拟合); 低段 (pred<40) 单独拟合, 修正低端抬升
        calibration = None
        if USE_CALIBRATION and FEATURE_SELECTION_METHOD is not None:
            from sklearn.base import clone
            from sklearn.model_selection import StratifiedKFold
            from scipy.stats import linregress
            # BUGFIX (2026-08-18 审计): OOF 必须用 selector 变换后的 15 特征,
            # 与部署模型的输入一致 (此前直接用了 43 维 X_sel, 系数拟合对象错位)
            _X_cal = selector.transform(X_sel) if selector is not None else X_sel
            _cal_bins = pd.cut(y_sel, bins=[-np.inf, 50, 150, 400, 700, np.inf], labels=False)
            _cal_cv = list(StratifiedKFold(n_splits=5, shuffle=True,
                                           random_state=RANDOM_SEED).split(_X_cal, _cal_bins))
            _oof = np.zeros(len(y_sel))
            for _tr, _va in _cal_cv:
                _m = clone(model)
                _m.fit(_X_cal[_tr], y_sel[_tr])
                _oof[_va] = _m.predict(_X_cal[_va])
            _k, _b, _r, _, _ = linregress(_oof, y_sel)
            _lo_mask = _oof < 40.0
            _k_lo, _b_lo, _, _, _ = linregress(_oof[_lo_mask], y_sel[_lo_mask])
            calibration = {
                'type': 'piecewise', 'breakpoint': 40.0,
                'low': {'k': float(_k_lo), 'b': float(_b_lo),
                        'fit_on': f'{int(_lo_mask.sum())} OOF samples with pred<40'},
                'high': {'k': float(_k), 'b': float(_b), 'fit_on': 'all 289 OOF samples'},
            }
            print(f"\n[校正 model_V4] 高段: t = {_k:.4f}×pred + {_b:.1f} (OOF R²={_r**2:.4f})")
            print(f"[校正 model_V4] 低段(pred<40): t = {_k_lo:.4f}×pred + {_b_lo:.1f} (n={int(_lo_mask.sum())})")
            with open(os.path.join(MODEL_DIR, 'calibration.json'), 'w') as _f:
                json.dump({**calibration, 'oof_r2': float(_r ** 2),
                           'model': best_model_name}, _f, indent=2)

            # 校正后的 holdout 指标 (仅打印, 不用于拟合)
            _hv = pd.read_csv(f'{best_model_name}_conc_predict_validation_results.csv', sep='\t')
            _p_cal = np.clip(_k * _hv['conc_pred'].values + _b, 0, None)
            _hv_rmse = float(np.sqrt(mean_squared_error(_hv['conc_true'], _p_cal)))
            _hv_r2 = float(r2_score(_hv['conc_true'], _p_cal))
            print(f"[校正] holdout (校正后): RMSE={_hv_rmse:.2f}, R²={_hv_r2:.4f}")

    # ── 分类器训练 (classification / dual) ──
    if RUN_MODE in ('classification', 'dual'):
        # 构建原始 43 维特征 DataFrame
        X_train_43_df = pd.DataFrame(X_train, columns=FEATURE_NAMES, index=names_train)
        # 分类器直接使用全部 43 维特征, 不做特征筛选
        clf_feat_indices = list(range(len(FEATURE_NAMES)))
        classifier, clf_selector, clf_meta = train_hemolysis_classifier(
            X_train_43_df, y_train, clf_feat_indices, MODEL_DIR, cv_method=CV_METHOD)
        print(f"\n分类器 holdout Acc: {clf_meta['holdout_acc']:.4f}")
        print(f"分类器特征数: {clf_meta['n_features']}")
        print(f"分类器特征: {clf_meta['selected_features']}")

    # ── 7.7 评估所有测试集 ──
    print(f"\n{'='*60}")
    print(f"评估所有测试集")
    print(f"{'='*60}")

    eval_results = []
    for split_name in ['test_set1', 'test_set2', 'test_set3', 'test_set4']:
        if split_name not in splits:
            continue
        X_t, y_t, names_t, bids_t = splits[split_name]
        if len(X_t) == 0:
            continue

        if RUN_MODE == 'regression':
            res = evaluate_split(model, selector, X_t, y_t, names_t, bids_t, split_name, OUTPUT_DIR,
                                 calib=calibration)
        elif RUN_MODE == 'classification':
            res = evaluate_classifier_split(classifier, clf_selector, X_t, y_t, names_t, bids_t, split_name, OUTPUT_DIR)
        else:  # dual
            res = evaluate_dual_split(model, classifier, selector, clf_selector,
                                       X_t, y_t, names_t, bids_t, split_name, OUTPUT_DIR)
        eval_results.append(res)

    # ── 7.8 对比汇总 ──
    print(f"\n{'='*60}")
    print(f"最终对比")
    print(f"{'='*60}")

    summary_df = pd.DataFrame(eval_results)
    print(summary_df.to_string(index=False))

    # 按模式保存 summary
    if RUN_MODE == 'classification':
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_clf.csv"), index=False, sep='\t')
    elif RUN_MODE == 'dual':
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False, sep='\t')  # 兼容
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_dual.csv"), index=False, sep='\t')
    else:
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False, sep='\t')

    # 对比柱状图
    if RUN_MODE == 'classification':
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.bar(summary_df['split'], summary_df['Accuracy'], color=['#2196F3','#FF9800','#4CAF50','#F44336'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy by Test Set')
        ax.set_ylim(0, 1)
        for i, v in enumerate(summary_df['Accuracy']):
            ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=10)
    elif RUN_MODE == 'dual':
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes[0, 0].bar(summary_df['split'], summary_df['R²'], color=['#2196F3','#FF9800','#4CAF50','#F44336'])
        axes[0, 0].set_ylabel('R²'); axes[0, 0].set_title('Regression R²')
        for i, v in enumerate(summary_df['R²']):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        axes[0, 1].bar(summary_df['split'], summary_df['RMSE'], color=['#2196F3','#FF9800','#4CAF50','#F44336'])
        axes[0, 1].set_ylabel('RMSE'); axes[0, 1].set_title('Regression RMSE')
        for i, v in enumerate(summary_df['RMSE']):
            axes[0, 1].text(i, v + 5, f'{v:.1f}', ha='center', fontsize=10)

        x = np.arange(len(summary_df)); w = 0.35
        axes[1, 0].bar(x - w/2, summary_df['Reg_Acc'], w, label='Reg (离散化)', color='#2196F3')
        axes[1, 0].bar(x + w/2, summary_df['Clf_Acc'], w, label='Classifier', color='#FF9800')
        axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(summary_df['split'])
        axes[1, 0].set_ylabel('Accuracy'); axes[1, 0].set_title('Grade Accuracy: Reg vs Classifier')
        axes[1, 0].legend(); axes[1, 0].set_ylim(0, 1.05)

        axes[1, 1].bar(x - w/2, summary_df['Clf_F1'], w, color='#4CAF50')
        axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(summary_df['split'])
        axes[1, 1].set_ylabel('F1 (macro)'); axes[1, 1].set_title('Classifier F1 (macro)')
        axes[1, 1].set_ylim(0, 1.05)
        for i, v in enumerate(summary_df['Clf_F1']):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(summary_df['split'], summary_df['R²'], color=['#2196F3','#FF9800','#4CAF50','#F44336'])
        axes[0].set_ylabel('R²')
        axes[0].set_title('R² by Test Set')
        axes[0].axhline(y=0, color='black', linewidth=0.5)
        for i, v in enumerate(summary_df['R²']):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        axes[1].bar(summary_df['split'], summary_df['RMSE'], color=['#2196F3','#FF9800','#4CAF50','#F44336'])
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE by Test Set')
        for i, v in enumerate(summary_df['RMSE']):
            axes[1].text(i, v + 5, f'{v:.1f}', ha='center', fontsize=10)

    fname = "summary_comparison.png"
    if RUN_MODE == 'classification':
        fname = "summary_clf_comparison.png"
    elif RUN_MODE == 'dual':
        fname = "summary_dual_comparison.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n全部完成! 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
