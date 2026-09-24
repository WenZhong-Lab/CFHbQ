#!/bin/bash
# ============================================================
# model_V4 一键复现 (冻结特征路线, 无需重新处理图像, 分钟级)
#   python plasma_pipeline.py --fixed_features=model_v4 --features_csv=results/features_20260818_170937.csv
#   产物应与 results/ 中官方产物一致:
#     independent test 平均 R2 = 0.9241, internal validation R2 = 0.8915 (RMSE 74.0)
# 用法: bash reproduce_model_v4.sh [输出目录]   (默认 model_v4_repro)
# 从图像完整重算特征: python plasma_pipeline.py   (自动使用 ./data/blood_imag)
# ============================================================
set -e
cd "$(dirname "$0")"

OUT_DIR="${1:-model_v4_repro}"
FEAT_SNAPSHOT="results/features_20260818_170937.csv"

echo "=== 冻结特征流水线 (输出: $OUT_DIR) ==="
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
(cd "$OUT_DIR" && python ../plasma_pipeline.py \
    --out_dir="$OUT_DIR" \
    --model_dir="$OUT_DIR" \
    --fixed_features=model_v4 \
    --features_csv="$FEAT_SNAPSHOT" \
    > run.log 2>&1)
echo "完成: $OUT_DIR/summary.csv, calibration.json, predictions_test_*.csv"
