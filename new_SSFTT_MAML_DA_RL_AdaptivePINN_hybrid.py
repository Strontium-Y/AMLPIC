import torch
import torch.nn as nn
import torch.nn.functional as F
from agents import MultiLayerAgentContainer
from SSFTTnet_200_1_DA import SSFTTnet
from mamba_branch import MambaBranch


class PINNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.absorption_coeff = nn.Parameter(torch.tensor(0.1))
        self.source_term = nn.Parameter(torch.tensor(0.5))

    def normalize_features(self, features):
        if features.dim() == 4:
            feat_min = features.min(dim=1, keepdim=True)[0]
            feat_max = features.max(dim=1, keepdim=True)[0]
        else:
            feat_min = features.min(dim=1, keepdim=True)[0]
            feat_max = features.max(dim=1, keepdim=True)[0]
        features_norm = (features - feat_min) / (feat_max - feat_min + 1e-8)
        return features_norm

    def forward(self, features):
        features_norm = self.normalize_features(features)
        if features_norm.dim() == 4:
            I = features_norm
            dI_dlambda = I[:, 1:, :, :] - I[:, :-1, :, :]
            I_mid = (I[:, 1:, :, :] + I[:, :-1, :, :]) / 2
        else:
            I = features_norm
            dI_dlambda = I[:, 1:] - I[:, :-1]
            I_mid = (I[:, 1:] + I[:, :-1]) / 2
        residual = dI_dlambda + self.absorption_coeff * I_mid - self.source_term
        pinn_loss = torch.mean(residual ** 2)
        return pinn_loss


class AdaptiveWeightAgent(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.state_dim = 3
        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, loss_cls, pinn_loss, epoch, total_epochs):
        epoch_progress = torch.tensor(epoch / total_epochs, device=loss_cls.device, dtype=loss_cls.dtype)
        loss_cls_norm = torch.sigmoid(loss_cls - 0.5)
        pinn_loss_norm = torch.sigmoid(pinn_loss - 0.5)
        state = torch.stack([loss_cls_norm, pinn_loss_norm, epoch_progress])
        state = state.view(1, -1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        beta = torch.sigmoid(self.fc3(x))
        return beta.squeeze()


class SharedFeatureExtractor(nn.Module):
    def __init__(self, spec_dim=200, dim=128, depth=2, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.ssftt_branch = SSFTTnet(
            num_classes=dim,
            spec_dim=spec_dim,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        self.mamba_branch = MambaBranch(
            spec_dim=spec_dim,
            patch_size=13,
            dim=dim,
            d_state=64,
            d_conv=4,
            expand_factor=2
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.feature_dim = dim

    def forward(self, x):
        ssftt_out = self.ssftt_branch(x)
        if isinstance(ssftt_out, tuple):
            feat_ssftt = ssftt_out[1]
        else:
            feat_ssftt = ssftt_out
        feat_mamba = self.mamba_branch(x)
        fused = torch.cat([feat_ssftt, feat_mamba], dim=-1)
        fused = self.fusion(fused)
        return fused


class DynamicClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


class MAML_SSFTT_DA_RL_AdaptivePINN(nn.Module):
    def __init__(self, num_classes=16, inner_lr=0.03, update_step=5,
                 use_agents=True, use_domain_adapter=True, da_alpha=1.0,
                 use_pinn=True, init_pinn_weight=0.5,
                 domain_adapter_type='binary', hidden_dim=64,
                 spec_dim=200, dim=128, depth=2, heads=8, mlp_dim=256, dropout=0.1):
        super().__init__()

        self.use_agents = use_agents
        self.use_domain_adapter = use_domain_adapter
        self.use_pinn = use_pinn
        self.da_alpha = da_alpha
        self.inner_lr = inner_lr
        self.update_step = update_step
        self.init_pinn_weight = init_pinn_weight

        self.feature_extractor = SharedFeatureExtractor(
            spec_dim=spec_dim, dim=dim, depth=depth,
            heads=heads, mlp_dim=mlp_dim, dropout=dropout
        )
        self.feature_dim = dim

        self.classifier = DynamicClassifier(self.feature_dim, num_classes)
        self.current_num_classes = num_classes

        self.pinn = PINNModule()
        self.weight_agent = AdaptiveWeightAgent(hidden_dim=32)

        if self.use_domain_adapter:
            if domain_adapter_type == 'binary':
                self.domain_classifier = nn.Sequential(
                    nn.Linear(self.feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 1)
                )
        else:
            self.domain_classifier = None

        if use_agents:
            param_shapes = [p.data.shape for p in self.feature_extractor.parameters()]
            try:
                self.agent_container = MultiLayerAgentContainer(param_shapes, hidden_dim=hidden_dim)
                self.agent_params = list(self.agent_container.parameters())
                if len(self.agent_params) == 0:
                    self.agent_container = None
                    self.agent_params = []
            except Exception:
                self.agent_container = None
                self.agent_params = []
        else:
            self.agent_container = None
            self.agent_params = []

    def set_num_classes(self, num_classes):
        if num_classes != self.current_num_classes:
            device = next(self.classifier.parameters()).device
            self.classifier = DynamicClassifier(self.feature_dim, num_classes).to(device)
            self.current_num_classes = num_classes

    def compute_domain_loss(self, src_features, tgt_features):
        if not self.use_domain_adapter or self.domain_classifier is None:
            return torch.tensor(0.0, device=src_features.device)
        src_pred = self.domain_classifier(src_features)
        tgt_pred = self.domain_classifier(tgt_features)
        src_labels = torch.ones_like(src_pred)
        tgt_labels = torch.zeros_like(tgt_pred)
        loss_da = F.binary_cross_entropy_with_logits(
            torch.cat([src_pred, tgt_pred], dim=0),
            torch.cat([src_labels, tgt_labels], dim=0)
        )
        return loss_da

    def adapt_on_task(self, support_x, support_y, beta):
        original_params = [p.detach().clone() for p in self.feature_extractor.parameters()]
        flat_params_list = [p.detach().flatten() for p in self.feature_extractor.parameters()]

        self.feature_extractor.train()

        for step in range(self.update_step):
            features = self.feature_extractor(support_x)
            logits = self.classifier(features)
            loss_cls = F.cross_entropy(logits, support_y)
            loss_pinn = self.pinn(features) if self.use_pinn else 0.0
            total_loss = loss_cls + beta * loss_pinn

            grads = torch.autograd.grad(total_loss, self.feature_extractor.parameters(),
                                        create_graph=False, allow_unused=True)

            flat_grads_list = []
            for g, fp in zip(grads, flat_params_list):
                if g is not None:
                    flat_grads_list.append(g.detach().flatten())
                else:
                    flat_grads_list.append(torch.zeros_like(fp))

            if self.use_agents and self.agent_container is not None:
                try:
                    valid_grads = [g for g in flat_grads_list if g is not None]
                    grad_norm = torch.norm(torch.cat(valid_grads)) if valid_grads else torch.tensor(0.0,
                                                                                                    device=loss_cls.device)
                    loss_val = total_loss.detach()
                    current_lr = torch.tensor(self.inner_lr).to(total_loss.device)

                    corrected_grads_list, masks, new_lr = self.agent_container(
                        flat_params_list, flat_grads_list, grad_norm, loss_val, current_lr
                    )

                    for i, (fp, cg) in enumerate(zip(flat_params_list, corrected_grads_list)):
                        flat_params_list[i] = fp - new_lr * cg

                    with torch.no_grad():
                        for p, fp_new in zip(self.feature_extractor.parameters(), flat_params_list):
                            p.data = fp_new.view(p.shape)

                except Exception:
                    with torch.no_grad():
                        for p, g in zip(self.feature_extractor.parameters(), flat_grads_list):
                            if g is not None:
                                p.data -= self.inner_lr * g.view(p.shape)
            else:
                with torch.no_grad():
                    for p, g in zip(self.feature_extractor.parameters(), flat_grads_list):
                        if g is not None:
                            p.data -= self.inner_lr * g.view(p.shape)

        fast_weights = [p.detach().clone() for p in self.feature_extractor.parameters()]

        with torch.no_grad():
            for p, orig in zip(self.feature_extractor.parameters(), original_params):
                p.data = orig

        return fast_weights

    def adapt_on_task_with_classifier(self, support_x, support_y, beta=0.0,
                                      update_classifier=True, use_agent=False):
        orig_feat_params = [p.detach().clone() for p in self.feature_extractor.parameters()]
        if update_classifier:
            orig_cls_params = [p.detach().clone() for p in self.classifier.parameters()]
        else:
            orig_cls_params = None

        feat_flat = [p.detach().flatten() for p in self.feature_extractor.parameters()]
        if update_classifier:
            cls_flat = [p.detach().flatten() for p in self.classifier.parameters()]
        else:
            cls_flat = []

        self.feature_extractor.train()
        if update_classifier:
            self.classifier.train()

        for step in range(self.update_step):
            features = self.feature_extractor(support_x)
            logits = self.classifier(features)
            loss_cls = F.cross_entropy(logits, support_y)

            if self.use_pinn and beta > 0:
                loss_pinn = self.pinn(features)
                total_loss = loss_cls + beta * loss_pinn
            else:
                total_loss = loss_cls

            params_to_update = list(self.feature_extractor.parameters())
            if update_classifier:
                params_to_update += list(self.classifier.parameters())

            grads = torch.autograd.grad(total_loss, params_to_update,
                                        create_graph=False, allow_unused=True)

            num_feat_params = len(list(self.feature_extractor.parameters()))
            feat_grads = grads[:num_feat_params]
            cls_grads = grads[num_feat_params:] if update_classifier else []

            flat_feat_grads = [g.detach().flatten() if g is not None else torch.zeros_like(f)
                               for g, f in zip(feat_grads, feat_flat)]
            if update_classifier:
                flat_cls_grads = [g.detach().flatten() if g is not None else torch.zeros_like(c)
                                  for g, c in zip(cls_grads, cls_flat)]
            else:
                flat_cls_grads = []

            lr = self.inner_lr
            with torch.no_grad():
                for i, (fp, g) in enumerate(zip(feat_flat, flat_feat_grads)):
                    feat_flat[i] = fp - lr * g
                if update_classifier:
                    for i, (cp, g) in enumerate(zip(cls_flat, flat_cls_grads)):
                        cls_flat[i] = cp - lr * g

            for p, fp in zip(self.feature_extractor.parameters(), feat_flat):
                p.data = fp.view(p.shape)
            if update_classifier:
                for p, cp in zip(self.classifier.parameters(), cls_flat):
                    p.data = cp.view(p.shape)

        fast_feat_weights = [p.detach().clone() for p in self.feature_extractor.parameters()]
        if update_classifier:
            fast_cls_weights = [p.detach().clone() for p in self.classifier.parameters()]
        else:
            fast_cls_weights = None

        with torch.no_grad():
            for p, orig in zip(self.feature_extractor.parameters(), orig_feat_params):
                p.data = orig
            if update_classifier:
                for p, orig in zip(self.classifier.parameters(), orig_cls_params):
                    p.data = orig

        return fast_feat_weights, fast_cls_weights

    def forward(self, support_x, support_y, query_x, query_y, epoch=0, total_epochs=100, return_feats=False):
        with torch.no_grad():
            features_ref = self.feature_extractor(support_x)
            logits_ref = self.classifier(features_ref)
            loss_cls_ref = F.cross_entropy(logits_ref, support_y)
            loss_pinn_ref = self.pinn(features_ref) if self.use_pinn else torch.tensor(0.0, device=loss_cls_ref.device)
            beta = self.weight_agent(loss_cls_ref, loss_pinn_ref, epoch, total_epochs)

        fast_weights = self.adapt_on_task(support_x, support_y, beta)

        original_params = [p.detach().clone() for p in self.feature_extractor.parameters()]
        with torch.no_grad():
            for p, fw in zip(self.feature_extractor.parameters(), fast_weights):
                p.data = fw

        query_features = self.feature_extractor(query_x)
        query_logits = self.classifier(query_features)
        loss_cls = F.cross_entropy(query_logits, query_y)
        acc = (query_logits.argmax(1) == query_y).float().mean()
        loss_pinn = self.pinn(query_features) if self.use_pinn else 0.0

        with torch.no_grad():
            for p, orig in zip(self.feature_extractor.parameters(), original_params):
                p.data = orig

        if return_feats:
            return loss_cls, acc, query_features, loss_pinn, beta
        return loss_cls, acc, loss_pinn, beta

    def test_cross_domain(self, support_x, support_y, query_x, query_y, epoch=0, total_epochs=100):
        beta = self.init_pinn_weight
        fast_weights = self.adapt_on_task(support_x, support_y, beta)
        original_params = [p.detach().clone() for p in self.feature_extractor.parameters()]
        with torch.no_grad():
            for p, fw in zip(self.feature_extractor.parameters(), fast_weights):
                p.data = fw
        with torch.no_grad():
            query_features = self.feature_extractor(query_x)
            query_logits = self.classifier(query_features)
            acc = (query_logits.argmax(1) == query_y).float().mean().item()
        with torch.no_grad():
            for p, orig in zip(self.feature_extractor.parameters(), original_params):
                p.data = orig
        return acc