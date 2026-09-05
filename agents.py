import torch
import torch.nn as nn
import torch.nn.functional as F


class LRAgent(nn.Module):
    """Learning rate agent: outputs a new learning rate based on gradient norm, loss value, and previous LR."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, grad_norm, loss_val, prev_lr):
        # Inputs are scalars but support batched processing; unsqueeze to form feature vectors
        x = torch.cat([grad_norm.unsqueeze(-1), loss_val.unsqueeze(-1), prev_lr.unsqueeze(-1)], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output is clamped to [0, 0.2] to avoid excessively large learning rates
        lr = self.sigmoid(self.fc3(x)) * 0.2
        return lr.squeeze(-1)   # returns [batch] or scalar


class GradLayerAgent(nn.Module):
    """Gradient correction agent for a single layer: applies element-wise scaling to the flattened gradient vector."""

    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        # Input is the flattened current gradient (simplified; no previous layer gradient used)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, in_features)

    def forward(self, flat_grad):
        # flat_grad: [batch, dim]
        x = F.relu(self.fc1(flat_grad))
        x = F.relu(self.fc2(x))
        scaling = torch.sigmoid(self.fc3(x)) * 2.0   # scaling factor in [0, 2]
        return scaling


class ParamLayerAgent(nn.Module):
    """Parameter importance agent for a single layer: computes element-wise importance mask from parameters and gradients."""

    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        # Concatenate flattened parameter and gradient -> input dimension = in_features * 2
        self.fc1 = nn.Linear(in_features * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, in_features)

    def forward(self, flat_param, flat_grad):
        # Concatenate parameter and gradient along the feature dimension
        x = torch.cat([flat_param, flat_grad], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        importance = torch.sigmoid(self.fc3(x))   # importance weights in [0, 1]
        return importance


class MultiLayerAgentContainer(nn.Module):
    """Container managing agents for all layers, handling gradient correction, masking, and learning rate adjustment."""

    def __init__(self, model_parameters_shapes, hidden_dim=64):
        """
        Args:
            model_parameters_shapes: list of shapes (torch.Size) or number of elements (int) for each layer
        """
        super().__init__()
        self.num_layers = len(model_parameters_shapes)
        self.grad_agents = nn.ModuleList()   # one gradient correction agent per layer
        self.param_agents = nn.ModuleList()  # one importance agent per layer
        for shape in model_parameters_shapes:
            # Get total number of elements for this layer
            if hasattr(shape, 'numel'):
                dim = shape.numel()
            else:
                dim = shape
            self.grad_agents.append(GradLayerAgent(dim, hidden_dim))
            self.param_agents.append(ParamLayerAgent(dim, hidden_dim))
        # Global learning rate agent
        self.lr_agent = LRAgent(hidden_dim)

    def forward(self, flat_params_list, flat_grads_list, grad_norm, loss_val, prev_lr):
        """
        Args:
            flat_params_list: list of flattened parameter tensors per layer, each [batch, dim]
            flat_grads_list:  list of flattened gradient tensors per layer
            grad_norm:        global gradient norm (scalar or [batch])
            loss_val:         current loss value (scalar or [batch])
            prev_lr:          previous learning rate (scalar or [batch])
        Returns:
            corrected_grads: list of gradients after correction and masking
            masks:           importance masks for each layer (for monitoring or later use)
            lr:              updated learning rate
        """
        corrected_grads = []
        masks = []
        # Process each layer
        for i, (flat_param, flat_grad) in enumerate(zip(flat_params_list, flat_grads_list)):
            # 1. Gradient correction scaling
            scaling = self.grad_agents[i](flat_grad)
            corrected_grad = flat_grad * scaling
            # 2. Parameter importance mask (soft masking via multiplication)
            importance = self.param_agents[i](flat_param, flat_grad)
            masked_grad = corrected_grad * importance
            corrected_grads.append(masked_grad)
            masks.append(importance)
        # 3. Global learning rate adjustment
        lr = self.lr_agent(grad_norm, loss_val, prev_lr)
        return corrected_grads, masks, lr