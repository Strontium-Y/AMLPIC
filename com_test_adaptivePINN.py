import torch
import numpy as np
import random
import os
import time
from sklearn.metrics import confusion_matrix
from task_sampler import TaskSampler
from new_SSFTT_MAML_DA_RL_AdaptivePINN_hybrid import MAML_SSFTT_DA_RL_AdaptivePINN
from com_data_utils import HSI_Dataset as TargetDataset

# ========== Configuration ==========
MODEL_PATH = "com_cls_result_hybrid/best_model.pth"   # Path to pretrained model
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Random seeds for computing standard deviations across multiple runs
SEED_LIST = [42, 123, 456, 789, 1011]

# Dataset configurations: path to .mat files and number of classes
DATASETS = {
    'salinas': {
        'path': ('data/Salinas.mat', 'data/Salinas_gt.mat'),
        'num_classes': 16,
    },
    'pavia_university': {
        'path': ('data/PaviaU.mat', 'data/PaviaU_gt.mat'),
        'num_classes': 9,
    },
    'pavia_centre': {
        'path': ('data/Pavia.mat', 'data/Pavia_gt.mat'),
        'num_classes': 9,
    },
    'houston': {
        'path': ('data/Houston/Houstondata.mat', 'data/Houston/Houstonlabel.mat'),
        'num_classes': 15,
    },
    'botswana': {
        'path': ('data/Botswana.mat', 'data/Botswana_gt.mat'),
        'num_classes': 14,
    },
    'ksc': {
        'path': ('data/KSC.mat', 'data/KSC_gt.mat'),
        'num_classes': 13,
    },
}

# Meta-testing parameters
NUM_TESTS = 50          # Number of test tasks per seed per dataset
N_WAY = 5               # Number of classes per task
K_SHOT = 5              # Number of support samples per class
K_QUERY = 15            # Number of query samples per class
INNER_LR = 0.03         # Inner-loop learning rate for adaptation (consistent with training)
UPDATE_STEP = 20        # Number of inner-loop gradient steps

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_metrics(conf_mat):
    """Compute OA, AA, Kappa, and per-class accuracy from confusion matrix."""
    OA = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    class_acc = np.diag(conf_mat) / (np.sum(conf_mat, axis=1) + 1e-8)
    AA = np.mean(class_acc)
    total = np.sum(conf_mat)
    sum_po = np.sum(np.diag(conf_mat))
    sum_pe = np.sum(np.sum(conf_mat, axis=0) * np.sum(conf_mat, axis=1)) / (total * total)
    Kappa = (sum_po / total - sum_pe) / (1 - sum_pe) if (1 - sum_pe) != 0 else 0
    return OA, AA, Kappa, class_acc

def main():
    """Run cross-dataset few-shot testing with multiple random seeds and report statistics."""
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Testing with {len(SEED_LIST)} random seeds: {SEED_LIST}\n")

    # Load the pretrained model once; it will be reused for all datasets and seeds
    model = MAML_SSFTT_DA_RL_AdaptivePINN(
        num_classes=16,
        inner_lr=INNER_LR,
        update_step=UPDATE_STEP,
        use_agents=True,
        use_domain_adapter=True,
        use_pinn=True,
        init_pinn_weight=0.5,
        spec_dim=200,
        dim=128,
        depth=2,
        heads=8,
        mlp_dim=256,
        dropout=0.1
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}\n")

    # results[dataset_name][seed] = (OA, AA, Kappa, class_acc_array)
    results = {name: {} for name in DATASETS}

    total_start = time.time()
    total_tasks = 0

    # Iterate over each random seed to get statistics across runs
    for seed in SEED_LIST:
        set_seed(seed)
        print(f"\n========== Running with Seed = {seed} ==========")

        for name, cfg in DATASETS.items():
            data_path, gt_path = cfg['path']
            if not os.path.exists(data_path):
                print(f"  Data file not found: {data_path}. Skipping {name}.")
                continue

            # Load target dataset (different from training set)
            try:
                dataset = TargetDataset(data_path, gt_path, patch_size=13, target_bands=200, use_standardize=True)
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
                continue

            # Create task sampler for few-shot episodes
            sampler = TaskSampler(dataset, N_WAY, K_SHOT, K_QUERY)
            # Adjust classifier output dimension to match current dataset
            model.set_num_classes(cfg['num_classes'])

            all_preds = []
            all_labels = []
            task_times = []

            # Run multiple test tasks for this dataset and seed
            for i in range(NUM_TESTS):
                task_start = time.time()
                support_x, support_y, query_x, query_y = sampler.sample_task()
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                query_x = query_x.to(device)
                query_y = query_y.to(device)

                # Adapt the model on the support set (both feature extractor and classifier)
                beta = 0.0
                fast_feat_weights, fast_cls_weights = model.adapt_on_task_with_classifier(
                    support_x, support_y, beta=beta, update_classifier=True, use_agent=False
                )

                # Save original parameters to restore after query evaluation
                orig_feat = [p.detach().clone() for p in model.feature_extractor.parameters()]
                orig_cls = [p.detach().clone() for p in model.classifier.parameters()]

                # Apply the adapted (fast) weights
                with torch.no_grad():
                    for p, fw in zip(model.feature_extractor.parameters(), fast_feat_weights):
                        p.data = fw
                    for p, cw in zip(model.classifier.parameters(), fast_cls_weights):
                        p.data = cw

                # Evaluate on query set
                with torch.no_grad():
                    query_feats = model.feature_extractor(query_x)
                    logits = model.classifier(query_feats)
                    preds = logits.argmax(1).cpu().numpy()

                # Restore original weights for next task
                with torch.no_grad():
                    for p, orig in zip(model.feature_extractor.parameters(), orig_feat):
                        p.data = orig
                    for p, orig in zip(model.classifier.parameters(), orig_cls):
                        p.data = orig

                all_preds.extend(preds)
                all_labels.extend(query_y.cpu().numpy())

                task_times.append(time.time() - task_start)
                total_tasks += 1

                if (i+1) % 10 == 0:
                    print(f"  {name.upper()} - {i+1}/{NUM_TESTS} tasks done")

            # Compute evaluation metrics for this seed
            n_classes = cfg['num_classes']
            conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
            OA, AA, Kappa, class_acc = compute_metrics(conf_mat)

            # Store results
            results[name][seed] = (OA, AA, Kappa, class_acc)

            print(f"  {name.upper()} | Seed {seed} | OA={OA:.4f}, AA={AA:.4f}, Kappa={Kappa:.4f}")

    # After all seeds, compute mean and standard deviation across seeds
    total_time = time.time() - total_start
    avg_task_time = total_time / total_tasks if total_tasks > 0 else 0

    print("\n" + "="*80)
    print("FINAL RESULTS (Mean ± Std over {} random seeds)".format(len(SEED_LIST)))
    print("="*80)

    # Write detailed results to a text file
    with open("detailed_test_results_std.txt", "w") as f:
        f.write("Cross-dataset 5-shot results (mean ± std)\n")
        f.write("Model: {}\n".format(MODEL_PATH))
        f.write("Seeds: {}\n".format(SEED_LIST))
        f.write("Total testing time: {:.2f} seconds\n".format(total_time))
        f.write("Average time per task: {:.4f} seconds\n\n".format(avg_task_time))

        for name, seed_dict in results.items():
            if not seed_dict:
                continue

            # Gather metrics from all seeds for this dataset
            oa_list = []
            aa_list = []
            kappa_list = []
            class_acc_list = []

            for seed, (OA, AA, Kappa, class_acc) in seed_dict.items():
                oa_list.append(OA)
                aa_list.append(AA)
                kappa_list.append(Kappa)
                class_acc_list.append(class_acc)

            # Compute mean and std for overall metrics
            oa_mean = np.mean(oa_list)
            oa_std = np.std(oa_list)
            aa_mean = np.mean(aa_list)
            aa_std = np.std(aa_list)
            kappa_mean = np.mean(kappa_list)
            kappa_std = np.std(kappa_list)

            # Per-class accuracy: stack across seeds and compute mean/std per class
            class_acc_array = np.stack(class_acc_list, axis=0)  # (num_seeds, n_classes)
            class_mean = np.mean(class_acc_array, axis=0)
            class_std = np.std(class_acc_array, axis=0)

            # Print to console
            print(f"\n{name.upper()}:")
            print(f"  OA = {oa_mean:.4f} ± {oa_std:.4f}")
            print(f"  AA = {aa_mean:.4f} ± {aa_std:.4f}")
            print(f"  Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}")
            for c, (m, s) in enumerate(zip(class_mean, class_std)):
                print(f"    Class {c}: {m:.4f} ± {s:.4f}")

            # Write to file
            f.write(f"\n{name.upper()}:\n")
            f.write(f"  OA = {oa_mean:.4f} ± {oa_std:.4f}\n")
            f.write(f"  AA = {aa_mean:.4f} ± {aa_std:.4f}\n")
            f.write(f"  Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}\n")
            for c, (m, s) in enumerate(zip(class_mean, class_std)):
                f.write(f"  Class_{c}: {m:.4f} ± {s:.4f}\n")
            f.write("\n")

        # Also write individual seed results for reference
        f.write("\n--- Individual seed results ---\n")
        for name, seed_dict in results.items():
            if not seed_dict:
                continue
            f.write(f"\n{name.upper()}:\n")
            for seed, (OA, AA, Kappa, _) in seed_dict.items():
                f.write(f"  Seed {seed}: OA={OA:.4f}, AA={AA:.4f}, Kappa={Kappa:.4f}\n")

    print("\nDetailed results (including per-class std) saved to 'detailed_test_results_std.txt'")
    print(f"Total testing time: {total_time:.2f} seconds")
    print(f"Average time per task: {avg_task_time:.4f} seconds")

if __name__ == "__main__":
    main()