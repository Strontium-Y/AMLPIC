import random
import torch

class TaskSampler:
    """从数据集中采样 N-way K-shot 任务"""
    def __init__(self, dataset, n_way, k_shot, k_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        # 按类别索引样本
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            lbl = label.item() if torch.is_tensor(label) else label
            if lbl not in self.class_to_indices:
                self.class_to_indices[lbl] = []
            self.class_to_indices[lbl].append(idx)

    def sample_task(self):
        # 选择 n_way 个类别（确保每个类别有足够样本）
        available_classes = [c for c, idxs in self.class_to_indices.items()
                             if len(idxs) >= self.k_shot + self.k_query]
        if len(available_classes) < self.n_way:
            raise ValueError(f"Not enough classes with sufficient samples: need {self.n_way}, have {len(available_classes)}")
        classes = random.sample(available_classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for cls in classes:
            indices = self.class_to_indices[cls]
            selected = random.sample(indices, self.k_shot + self.k_query)
            support_idx = selected[:self.k_shot]
            query_idx = selected[self.k_shot:]

            for idx in support_idx:
                x, _ = self.dataset[idx]
                support_x.append(x)
                support_y.append(cls)
            for idx in query_idx:
                x, _ = self.dataset[idx]
                query_x.append(x)
                query_y.append(cls)

        support_x = torch.stack(support_x)  # (n_way*k_shot, 1, bands, h, w)
        support_y = torch.tensor(support_y).long()
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y).long()

        return support_x, support_y, query_x, query_y