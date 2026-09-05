import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import random
from wo_SSFTT_MAML_DA_RL_AdaptivePINN_hybrid import MAML_SSFTT_DA_RL_AdaptivePINN
from com_data_utils import HSI_Dataset as TargetDataset


MODEL_PATH = "com_cls_result_full/best_model.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "botswana"


PERPLEXITY = 30
N_COMPONENTS = 2
N_ITER = 1000
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 100


DATASET_PATHS = {
    'salinas': ('data/Salinas.mat', 'data/Salinas_gt.mat', 16),
    'pavia_university': ('data/PaviaU.mat', 'data/PaviaU_gt.mat', 9),
    'pavia_centre': ('data/Pavia.mat', 'data/Pavia_gt.mat', 9),
    'houston': ('data/Houston/Houstondata.mat', 'data/Houston/Houstonlabel.mat', 15),
    'botswana': ('data/Botswana.mat', 'data/Botswana_gt.mat', 14),
    'ksc': ('data/KSC.mat', 'data/KSC_gt.mat', 13),
}


BOTSWANA_CLASS_NAMES = [
    'Background',  # 0
    'Water',  # 1
    'Hippo grass',  # 2
    'Floodplain grasses 1',  # 3
    'Floodplain grasses 2',  # 4
    'Reeds',  # 5
    'Riparian',  # 6
    'Firescar',  # 7
    'Island interior',  # 8
    'Acacia woodlands',  # 9
    'Acacia shrublands',  # 10
    'Acacia grasslands',  # 11
    'Short mopane',  # 12
    'Mixed mopane',  # 13
    'Exposed soils'  # 14
]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model_config():
    return {
        "use_pinn": True,
        "use_agents": True,
        "use_domain_adapter": True
    }


def extract_features_all_classes(model, dataloader, n_classes, device, samples_per_class=100):
    model.eval()
    all_features = []
    all_labels = []

    print("正在提取所有样本的特征...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            output = model.feature_extractor(x)
            if isinstance(output, (tuple, list)):
                features = output[0]
            else:
                features = output

            all_features.append(features.cpu().numpy())
            all_labels.append(y.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  已处理 {batch_idx + 1} 个批次")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"总样本数: {features.shape[0]}, 特征维度: {features.shape[1]}")

    unique_labels = np.unique(labels)
    sampled_indices = []
    sampling_info = []

    for cls in unique_labels:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) > samples_per_class:
            sampled = np.random.choice(cls_indices, samples_per_class, replace=False)
        else:
            sampled = cls_indices
        sampled_indices.extend(sampled)
        sampling_info.append((cls, len(cls_indices), len(sampled)))

    print("\n采样详情:")
    for cls, total, sampled_count in sampling_info:
        print(f"  Class {cls}: {total} 个样本 → 采样 {sampled_count} 个")

    sampled_indices = np.array(sampled_indices)
    features = features[sampled_indices]
    labels = labels[sampled_indices]

    print(f"\n最终采样: {features.shape[0]} 个样本, {features.shape[1]} 维")
    print(f"类别分布: {Counter(labels)}")

    return features, labels


def main():
    set_seed(RANDOM_STATE)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_NAME}")


    if DATASET_NAME not in DATASET_PATHS:
        print(f"Unknown dataset: {DATASET_NAME}")
        return

    data_path, gt_path, n_classes = DATASET_PATHS[DATASET_NAME]
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    try:
        dataset = TargetDataset(data_path, gt_path, patch_size=13, target_bands=200, use_standardize=True)
        print(f"Loaded dataset: {len(dataset)} samples, {n_classes} classes")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )


    config = get_model_config()
    model = MAML_SSFTT_DA_RL_AdaptivePINN(
        num_classes=16,
        inner_lr=0.03,
        update_step=20,
        use_agents=config["use_agents"],
        use_domain_adapter=config["use_domain_adapter"],
        use_pinn=config["use_pinn"],
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


    model.set_num_classes(n_classes)
    features, labels = extract_features_all_classes(model, dataloader, n_classes, device, SAMPLES_PER_CLASS)


    print("\nRunning t-SNE (may take 1-3 minutes)...")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tsne = TSNE(
        n_components=N_COMPONENTS,
        perplexity=PERPLEXITY,
        max_iter=N_ITER,
        random_state=RANDOM_STATE,
        init='pca',
        verbose=1
    )
    features_2d = tsne.fit_transform(features_scaled)
    print(f"t-SNE completed: {features_2d.shape}")


    class_names = []
    if DATASET_NAME == 'botswana':
        class_names = BOTSWANA_CLASS_NAMES
    else:
        try:
            if hasattr(dataset, 'classes') and isinstance(dataset.classes, list):
                if isinstance(dataset.classes[0], str):
                    class_names = dataset.classes
                else:
                    class_names = [f'Class {i}' for i in range(n_classes + 1)]
            else:
                class_names = [f'Class {i}' for i in range(n_classes + 1)]
        except:
            class_names = [f'Class {i}' for i in range(n_classes + 1)]


    plt.figure(figsize=(16, 12))

    cmap = plt.cm.tab20
    unique_labels = np.unique(labels)

    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        color_idx = int(cls) % 20

        label_text = class_names[int(cls)] if int(cls) < len(class_names) else f'Class {cls}'

        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[cmap(color_idx / 20)],
            label=label_text,
            s=12,
            alpha=0.7,
            edgecolors='none'
        )

    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    plt.title(
        f'AMLPIC t-SNE Visualization\n'
        f'Dataset: {DATASET_NAME.upper()} ({n_classes} classes, {SAMPLES_PER_CLASS} samples/class)',
        fontsize=14
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    save_path = f'tsne_amplic_{DATASET_NAME}_full.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")

    pdf_path = f'tsne_amlpic_{DATASET_NAME}_full.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")

    plt.show()

    print("\n" + "=" * 60)
    print(f"t-SNE Statistics for {DATASET_NAME.upper()}")
    print("=" * 60)
    print(f"Total samples: {features.shape[0]}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Perplexity: {PERPLEXITY}")
    print(f"Iterations: {N_ITER}")

    from scipy.spatial.distance import pdist
    class_centers = []
    for cls in unique_labels:
        mask = labels == cls
        center = features_2d[mask].mean(axis=0)
        class_centers.append(center)

    if len(class_centers) > 1:
        avg_inter_dist = np.mean(pdist(np.array(class_centers)))
        print(f"Average inter-class distance: {avg_inter_dist:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()