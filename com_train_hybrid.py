import torch
import torch.optim as optim
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import os
from new_SSFTT_MAML_DA_RL_AdaptivePINN_hybrid import MAML_SSFTT_DA_RL_AdaptivePINN
from data_utils import HSI_Dataset as SourceDataset
from data_utils_Salinas import HSI_Dataset as TargetDataset
from task_sampler import TaskSampler


def main():
    # ========== Record training start time ==========
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========== Experiment configuration ==========
    num_classes = 16
    n_way = 5
    k_shot = 5
    k_query = 15
    meta_lr = 1e-4  # Outer-loop learning rate
    inner_lr = 0.001  # Inner-loop learning rate
    update_step = 10  # Number of inner-loop steps (aligned with optimal test steps)
    epochs = 10  # Number of training epochs
    tasks_per_epoch = 100  # Number of tasks per epoch (total episodes = epochs * tasks_per_epoch)

    use_domain_adapter = True
    use_agents = True
    da_alpha = 1.0
    use_pinn = True
    init_pinn_weight = 0.5

    domain_adapter_type = 'binary'
    hidden_dim = 64

    spec_dim = 200
    dim = 128
    depth = 2
    heads = 8
    mlp_dim = 256
    dropout = 0.1
    patch_size = 13

    # ========== Data loading ==========
    source_full = SourceDataset("data/Indian_pines_corrected.mat",
                                "data/Indian_pines_gt.mat", patch_size=patch_size)
    coords = source_full.coords
    split_row = int(0.7 * 145)
    train_mask = coords < split_row
    train_indices = np.where(train_mask)[0].tolist()
    source_dataset = torch.utils.data.Subset(source_full, train_indices)
    source_sampler = TaskSampler(source_dataset, n_way, k_shot, k_query)

    target_dataset = TargetDataset("data/Salinas.mat", "data/Salinas_gt.mat", patch_size=patch_size)
    target_sampler = TaskSampler(target_dataset, n_way, k_shot, k_query)

    print(f"Source domain samples: {len(source_dataset)}, Target domain samples: {len(target_dataset)}")

    # ========== Create hybrid model ==========
    model = MAML_SSFTT_DA_RL_AdaptivePINN(
        num_classes=num_classes,
        inner_lr=inner_lr,
        update_step=update_step,
        use_agents=use_agents,
        use_domain_adapter=use_domain_adapter,
        da_alpha=da_alpha,
        use_pinn=use_pinn,
        init_pinn_weight=init_pinn_weight,
        domain_adapter_type=domain_adapter_type,
        hidden_dim=hidden_dim,
        spec_dim=spec_dim,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Optimizer
    all_params = list(model.feature_extractor.parameters()) + model.agent_params
    all_params += list(model.classifier.parameters())
    if use_domain_adapter:
        all_params += list(model.domain_classifier.parameters())
    all_params += list(model.weight_agent.parameters())
    all_params += list(model.pinn.parameters())
    optimizer = optim.Adam(all_params, lr=meta_lr)

    # ========== Result saving ==========
    results_dir = "com_cls_result_hybrid"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "training_log.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "cls_loss", "pinn_loss", "beta", "acc"])

    epoch_cls_losses = []
    epoch_pinn_losses = []
    epoch_betas = []
    epoch_accs = []
    best_acc = 0.0

    # ========== Training loop ==========
    for epoch in range(1, epochs + 1):
        model.train()
        total_cls_loss = 0.0
        total_pinn_loss = 0.0
        total_beta = 0.0
        total_acc = 0.0

        for _ in range(tasks_per_epoch):
            s_support_x, s_support_y, s_query_x, s_query_y = source_sampler.sample_task()
            s_support_x = s_support_x.to(device)
            s_support_y = s_support_y.to(device)
            s_query_x = s_query_x.to(device)
            s_query_y = s_query_y.to(device)

            t_support_x, t_support_y, t_query_x, t_query_y = target_sampler.sample_task()
            t_support_x = t_support_x.to(device)
            t_support_y = t_support_y.to(device)
            t_query_x = t_query_x.to(device)
            t_query_y = t_query_y.to(device)

            optimizer.zero_grad()

            cls_loss, acc, s_feats, pinn_loss, beta = model(
                s_support_x, s_support_y, s_query_x, s_query_y,
                epoch=epoch, total_epochs=epochs, return_feats=True
            )
            _, _, t_feats, _, _ = model(
                t_support_x, t_support_y, t_query_x, t_query_y,
                epoch=epoch, total_epochs=epochs, return_feats=True
            )
            da_loss = model.compute_domain_loss(s_feats, t_feats)

            total_loss = cls_loss + da_alpha * da_loss + beta * pinn_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_pinn_loss += pinn_loss.item()
            total_beta += beta.item()
            total_acc += acc.item()

        avg_cls = total_cls_loss / tasks_per_epoch
        avg_pinn = total_pinn_loss / tasks_per_epoch
        avg_beta = total_beta / tasks_per_epoch
        avg_acc = total_acc / tasks_per_epoch
        epoch_cls_losses.append(avg_cls)
        epoch_pinn_losses.append(avg_pinn)
        epoch_betas.append(avg_beta)
        epoch_accs.append(avg_acc)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_cls, avg_pinn, avg_beta, avg_acc])

        print(f"Epoch {epoch:3d} | Cls {avg_cls:.4f} | PINN {avg_pinn:.6f} | β {avg_beta:.4f} | Acc {avg_acc:.4f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            print(f"  -> New best model saved (acc={best_acc:.4f})")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(results_dir, f"model_epoch{epoch}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(results_dir, "final_model.pth"))

    # ========== Calculate total training time ==========
    total_time = time.time() - start_time
    total_hours = total_time / 3600
    print(f"Total training time: {total_hours:.2f} hours ({total_time:.2f} seconds)")

    # ========== Save training meta information (parameters, hyperparameters, time, etc.) ==========
    meta_path = os.path.join(results_dir, "train_meta.txt")
    with open(meta_path, 'w') as f:
        f.write("========== Training Meta Information ==========\n")
        f.write(f"Total parameters: {total_params / 1e6:.2f}M\n")
        f.write(f"Total training time: {total_hours:.2f} hours ({total_time:.2f} seconds)\n")
        f.write("\n----- Hyperparameters -----\n")
        f.write(f"epochs = {epochs}\n")
        f.write(f"tasks_per_epoch = {tasks_per_epoch}\n")
        f.write(f"total_episodes = {epochs * tasks_per_epoch}\n")
        f.write(f"meta_lr = {meta_lr}\n")
        f.write(f"inner_lr = {inner_lr}\n")
        f.write(f"update_step = {update_step}\n")
        f.write(f"n_way = {n_way}\n")
        f.write(f"k_shot = {k_shot}\n")
        f.write(f"k_query = {k_query}\n")
        f.write(f"use_domain_adapter = {use_domain_adapter}\n")
        f.write(f"use_agents = {use_agents}\n")
        f.write(f"da_alpha = {da_alpha}\n")
        f.write(f"use_pinn = {use_pinn}\n")
        f.write(f"init_pinn_weight = {init_pinn_weight}\n")
        f.write(f"spec_dim = {spec_dim}\n")
        f.write(f"dim = {dim}\n")
        f.write(f"depth = {depth}\n")
        f.write(f"heads = {heads}\n")
        f.write(f"mlp_dim = {mlp_dim}\n")
        f.write(f"dropout = {dropout}\n")
        f.write(f"patch_size = {patch_size}\n")
        f.write("==============================================\n")

    print(f"Model saved to {os.path.join(results_dir, 'final_model.pth')}")
    print(f"Training meta info saved to {meta_path}")

    # ========== Plot training curves ==========
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.plot(range(1, epochs + 1), epoch_cls_losses, label='Cls Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Classification Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(range(1, epochs + 1), epoch_pinn_losses, label='PINN Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Physics Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(range(1, epochs + 1), epoch_betas, label='Adaptive β', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('β')
    plt.title('Adaptive PINN Weight')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(range(1, epochs + 1), epoch_accs, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()