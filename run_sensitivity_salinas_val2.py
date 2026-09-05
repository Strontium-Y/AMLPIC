"""
Parameter Sensitivity Analysis - Salinas as Validation Set
"""

import os
import sys
import subprocess
import time
import shutil
import json
import re
import numpy as np
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.getcwd()
TRAIN_SCRIPT = "com_train_hybrid.py"

EPOCHS = 10
TASKS_PER_EPOCH = 50
UPDATE_STEP = 20

TEST_SEEDS = [42, 123, 456, 789, 1011]
NUM_TESTS = 50
TEST_INNER_LR = 0.03
TEST_UPDATE_STEP = 20

VALIDATION_DATASET = "Salinas"

INNER_LR_VALUES = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
DA_ALPHA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def run_training(param_name, param_value, results_dir):
    print(f"\n[RUN] Training: {param_name}={param_value}")
    backup = TRAIN_SCRIPT + ".backup"
    if not os.path.exists(backup):
        shutil.copy2(TRAIN_SCRIPT, backup)
    with open(TRAIN_SCRIPT, "r") as f:
        content = f.read()
    content = re.sub(r'inner_lr\s*=\s*[\d.]+', f'inner_lr = {param_value}', content)
    content = re.sub(r'da_alpha\s*=\s*[\d.]+', f'da_alpha = {param_value}', content)
    content = re.sub(r'epochs\s*=\s*\d+', f'epochs = {EPOCHS}', content)
    content = re.sub(r'tasks_per_epoch\s*=\s*\d+', f'tasks_per_epoch = {TASKS_PER_EPOCH}', content)
    content = re.sub(r'results_dir\s*=\s*"[^"]+"', f'results_dir = "{results_dir}"', content)
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(content)
    start = time.time()
    result = subprocess.run(["python", TRAIN_SCRIPT], capture_output=False, text=True)
    elapsed = time.time() - start
    if os.path.exists(backup):
        shutil.copy2(backup, TRAIN_SCRIPT)
        os.remove(backup)
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {param_name}={param_value}")
        return False
    print(f"[DONE] Completed in {elapsed / 60:.1f} min")
    return True


def run_test(model_path):
    """使用 com_test_adaptivePINN_1.py 进行快速测试"""
    print(f"[TEST] Testing model: {model_path}")

    cmd = ["python", "com_test_adaptivePINN_1.py", model_path, "Salinas"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Test failed with return code {result.returncode}")
        if result.stderr:
            print(f"[ERROR] stderr: {result.stderr}")
        return None

    output = result.stdout.strip()
    if not output:
        print(f"[ERROR] No output from test script")
        return None

    if "ERROR" in output:
        print(f"[ERROR] Test script reported error: {output}")
        return None

    try:
        lines = output.split('\n')
        last_line = lines[-1]
        parts = last_line.split(',')
        if len(parts) == 6:
            return {
                'OA_mean': float(parts[0]),
                'OA_std': float(parts[1]),
                'AA_mean': float(parts[2]),
                'AA_std': float(parts[3]),
                'Kappa_mean': float(parts[4]),
                'Kappa_std': float(parts[5]),
            }
        else:
            print(f"[ERROR] Unexpected output format: {last_line}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to parse output: {e}")
        print(f"[ERROR] Output: {output}")
        return None


def main():
    print("=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"Validation Set: {VALIDATION_DATASET}")
    print(f"Epochs: {EPOCHS}, Tasks per Epoch: {TASKS_PER_EPOCH}")
    print(f"Seeds: {len(TEST_SEEDS)}, Tests per Seed: {NUM_TESTS}")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {'inner_lr': {}, 'da_alpha': {}}

    print(f"\n" + "=" * 80)
    print(f"(b) Inner Learning Rate Sensitivity (Validation on {VALIDATION_DATASET})")
    print("=" * 80)
    for lr in INNER_LR_VALUES:
        rdir = f"sensitivity_LR_{lr}"
        mpath = os.path.join(rdir, "best_model.pth")

        if not os.path.exists(mpath):
            mpath = os.path.join(rdir, "final_model.pth")
            if not os.path.exists(mpath):
                print(f"[WARN] No model found for inner_lr={lr}, skipping")
                continue

        print(f"\n[PROCESS] inner_lr={lr}")
        r = run_test(mpath)
        if r is not None:
            results['inner_lr'][lr] = r
            print(f"  alpha={lr}: OA={r['OA_mean'] * 100:.2f} +/- {r['OA_std'] * 100:.2f}%")
        else:
            print(f"  [WARN] Failed to test inner_lr={lr}")

    print(f"\n" + "=" * 80)
    print(f"(c) DA Loss Weight Sensitivity (Validation on {VALIDATION_DATASET})")
    print("=" * 80)
    for da in DA_ALPHA_VALUES:
        rdir = f"sensitivity_DA_{da}"
        mpath = os.path.join(rdir, "best_model.pth")

        if not os.path.exists(mpath):
            mpath = os.path.join(rdir, "final_model.pth")
            if not os.path.exists(mpath):
                print(f"[WARN] No model found for da_alpha={da}, skipping")
                continue

        print(f"\n[PROCESS] da_alpha={da}")
        r = run_test(mpath)
        if r is not None:
            results['da_alpha'][da] = r
            print(f"  lambda_DA={da}: OA={r['OA_mean'] * 100:.2f} +/- {r['OA_std'] * 100:.2f}%")
        else:
            print(f"  [WARN] Failed to test da_alpha={da}")

    with open("sensitivity_results_salinas_val.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n[INFO] Results saved to sensitivity_results_salinas_val.json")

    print("\n" + "=" * 80)
    print("SUMMARY: Parameter Sensitivity Analysis (Validation on Salinas)")
    print("=" * 80)

    if results['inner_lr']:
        print("\n(b) Inner Learning Rate (alpha):")
        print(f"{'alpha':<12} {'OA (%)':<20}")
        print("-" * 35)
        for lr in sorted(results['inner_lr'].keys()):
            r = results['inner_lr'][lr]
            print(f"{lr:<12} {r['OA_mean'] * 100:.2f} +/- {r['OA_std'] * 100:.2f}")
        best_lr = max(results['inner_lr'].keys(), key=lambda x: results['inner_lr'][x]['OA_mean'])
        print(f"\n  Optimal alpha: {best_lr}")
    else:
        print("\n[WARN] No inner_lr results available")

    if results['da_alpha']:
        print("\n(c) DA Loss Weight (lambda_DA):")
        print(f"{'lambda_DA':<12} {'OA (%)':<20}")
        print("-" * 35)
        for da in sorted(results['da_alpha'].keys()):
            r = results['da_alpha'][da]
            print(f"{da:<12} {r['OA_mean'] * 100:.2f} +/- {r['OA_std'] * 100:.2f}")
        best_da = max(results['da_alpha'].keys(), key=lambda x: results['da_alpha'][x]['OA_mean'])
        print(f"\n  Optimal lambda_DA: {best_da}")
    else:
        print("\n[WARN] No da_alpha results available")

    print("\n" + "=" * 80)
    if results['inner_lr'] and results['da_alpha']:
        print("SELECTED HYPERPARAMETERS:")
        print(f"  inner_lr (training) = {best_lr}")
        print(f"  da_alpha = {best_da}")
        print(f"  update_step = {UPDATE_STEP}")
    else:
        print("[WARN] Some results missing. Check test script manually.")
    print("=" * 80)
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()