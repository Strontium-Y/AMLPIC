"""
Parameter Sensitivity Analysis - Update Steps T (Validation on Salinas)
Using current optimal configuration: inner_lr=0.001, da_alpha=5.0
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

# Using current optimal configuration
INNER_LR = 0.001
DA_ALPHA = 5.0
EPOCHS = 10
TASKS_PER_EPOCH = 50

UPDATE_STEP_VALUES = [25, 30]

TEST_SEEDS = [42, 123, 456, 789, 1011]
NUM_TESTS = 50
TEST_INNER_LR = 0.03
TEST_UPDATE_STEP = 20
VALIDATION_DATASET = "Salinas"


def run_training(update_step, results_dir):
    print(f"\n[RUN] Training: update_step={update_step}")
    backup = TRAIN_SCRIPT + ".backup"
    if not os.path.exists(backup):
        shutil.copy2(TRAIN_SCRIPT, backup)
    with open(TRAIN_SCRIPT, "r") as f:
        content = f.read()
    content = re.sub(r'inner_lr\s*=\s*[\d.]+', f'inner_lr = {INNER_LR}', content)
    content = re.sub(r'da_alpha\s*=\s*[\d.]+', f'da_alpha = {DA_ALPHA}', content)
    content = re.sub(r'update_step\s*=\s*\d+', f'update_step = {update_step}', content)
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
        print(f"[ERROR] Training failed for update_step={update_step}")
        return False
    print(f"[DONE] Completed in {elapsed / 60:.1f} min")
    return True


def run_test(model_path):
    print(f"[TEST] Testing model: {model_path}")
    cmd = ["python", "com_test_adaptivePINN_1.py", model_path, VALIDATION_DATASET]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if not output or "ERROR" in output:
        return None
    try:
        parts = output.split('\n')[-1].split(',')
        if len(parts) == 6:
            return {
                'OA_mean': float(parts[0]), 'OA_std': float(parts[1]),
                'AA_mean': float(parts[2]), 'AA_std': float(parts[3]),
                'Kappa_mean': float(parts[4]), 'Kappa_std': float(parts[5]),
            }
    except Exception:
        return None
    return None


def main():
    print("=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS - UPDATE STEP T")
    print(f"Validation Set: {VALIDATION_DATASET}")
    print(f"Fixed: inner_lr={INNER_LR}, da_alpha={DA_ALPHA}")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    for T in UPDATE_STEP_VALUES:
        rdir = f"sensitivity_T_{T}"
        mpath = os.path.join(rdir, "best_model.pth")
        if not os.path.exists(mpath):
            mpath = os.path.join(rdir, "final_model.pth")
            if not os.path.exists(mpath):
                print(f"[INFO] Training for T={T}...")
                if not run_training(T, rdir):
                    continue
                mpath = os.path.join(rdir, "best_model.pth")
                if not os.path.exists(mpath):
                    mpath = os.path.join(rdir, "final_model.pth")

        print(f"\n[PROCESS] Testing T={T}")
        r = run_test(mpath)
        if r:
            results[T] = r
            print(f"  T={T}: OA={r['OA_mean'] * 100:.2f} +/- {r['OA_std'] * 100:.2f}%")

    with open("sensitivity_results_T_salinas.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("SUMMARY: Update Step Sensitivity (Validation on Salinas)")
    print("=" * 80)
    print(f"{'T':<10} {'OA (%)':<20}")
    print("-" * 35)
    for T in sorted(results.keys()):
        r = results[T]
        print(f"{T:<10} {r['OA_mean'] * 100:.2f} +/- {r['OA_std'] * 100:.2f}")
    if results:
        best_T = max(results.keys(), key=lambda x: results[x]['OA_mean'])
        print(f"\n  Optimal T: {best_T}")
    print("=" * 80)


if __name__ == "__main__":
    main()