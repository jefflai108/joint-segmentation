import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DynamicWeightedBCEWithLogitsLoss(nn.Module):
    """
    A custom loss function for binary classification tasks that dynamically adjusts
    the weights for positive samples based on their prevalence in each batch. This
    approach helps to handle imbalanced datasets by calculating the weights within
    the loss function at runtime, ensuring that the loss is sensitive to the frequency
    of positive samples in each batch.

    The loss calculated is the Binary Cross-Entropy (BCE) Loss with Logits, which
    combines a sigmoid layer and the BCELoss in one single class. This version adapts
    the positive weights based on the batch content dynamically, which is particularly
    useful in cases where the dataset is large and/or highly imbalanced.

    Reference:
    - GitHub issue discussion on dynamic loss functions: https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305

    Methods:
        forward(logits, targets):
            Computes the BCEWithLogitsLoss using dynamically calculated positive weights.

    Args:
        logits (Tensor): The logits predicted by the model. Shape: [batch_size, *]
            These are raw, unnormalized scores output by the last layer of a model.
        targets (Tensor): The ground truth binary labels corresponding to the logits.
            Shape should match that of logits.

    Returns:
        Tensor: The computed loss as a scalar tensor, which aggregates the loss across
        all instances in the batch considering the dynamically adjusted positive weights.
    """
    def __init__(self, ignore_index=None, alpha=0.1):
        super(DynamicWeightedBCEWithLogitsLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of logits from the model (before sigmoid activation).
            targets: Binary targets tensor.

        Returns:
            Calculated loss with dynamic positive weights.
        """
        if self.ignore_index is not None:
            # only consider positions that are not PAD_TOKEN
            mask = targets != self.ignore_index
            logits = logits[mask]
            targets = targets[mask]

        # Calculate the count of positive samples per batch
        positive_counts = targets.sum(dim=0)
        total_counts = targets.size(0)

        # Avoid division by zero and calculate pos_weight
        pos_weight = (total_counts - positive_counts) / (positive_counts + 1e-5)
        pos_weight = pos_weight.to(logits.device)  # Ensure pos_weight is on the same device as logits

        # Apply label smoothing
        smoothed_targets = targets * (1 - self.alpha) + 0.5 * self.alpha
    
        # Calculate BCEWithLogitsLoss with dynamic pos_weight
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets.float(), pos_weight=pos_weight)

# Example usage
if __name__ == "__main__":
    # Assume logits and targets are torch tensors of appropriate shape
    logits = torch.randn(100, 32)  # Example logits
    targets = torch.randint(0, 2, (100, 32)).float()  # Example binary targets
    targets[-10:, :] = 1024 # padding 
    loss_func = DynamicWeightedBCEWithLogitsLoss(1024)

    # they should be the same 
    loss = loss_func(logits, targets)
    print("Calculated Loss:", loss.item())
    loss = loss_func(logits[:-2], targets[:-2])
    print("Calculated Loss:", loss.item())
