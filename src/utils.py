import torch

PAD_TOKEN = 1024

def calculate_boundary_metrics(logits, targets, threshold=0.5, ignore_index=PAD_TOKEN):
    """
    Calculate F1 score, R-value, and over-segmentation score from logits directly. 
    The accuracies calculated should be the same as that with calculate_accuracy()
    
    Args:
    - logits (Tensor): Logits output from the model (without sigmoid activation).
    - targets (Tensor): Ground truth binary labels.
    - threshold (float): Decision threshold for binary classification.
    - ignore_index (int, optional): Index that should be ignored in the calculation.
    
    Returns:
    - Tuple containing F1 score, R-value, and OS score.
    """
    with torch.no_grad():
        predictions = (torch.sigmoid(logits) > threshold).float()
        
        # Apply the mask to filter out ignored indices
        if ignore_index is not None:
            mask = (targets != ignore_index)
            predictions = predictions * mask
            targets = targets * mask

        true_positives = (predictions * targets).sum()
        predicted_positives = predictions.sum()
        actual_positives = targets.sum()
        correct_predictions = ((predictions == targets) & mask).sum() # important, only count corrects in MASK regions 
        
        # Calculate precision and recall
        precision = true_positives / predicted_positives if predicted_positives != 0 else torch.tensor(0.0, device=logits.device)
        recall = true_positives / actual_positives if actual_positives != 0 else torch.tensor(0.0, device=logits.device)

        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) != 0 else torch.tensor(0.0, device=logits.device)

        # Calculate Over-segmentation (OS)
        os = recall / precision - 1 if precision != 0 else torch.tensor(-float('inf'), device=logits.device)

        # Calculate R-value
        r1 = torch.sqrt((1 - recall) ** 2 + os ** 2)
        r2 = (-os + recall - 1) / torch.sqrt(torch.tensor(2.0, device=logits.device))
        rvalue = 1 - (torch.abs(r1) + torch.abs(r2)) / 2
    
        # Calculate Accuracy
        total_elements = mask.sum() if ignore_index is not None else targets.numel()
        accuracy = correct_predictions / total_elements if total_elements != 0 else torch.tensor(0.0)

        return f1_score.item(), rvalue.item(), os.item(), accuracy.item()

def calculate_accuracy(logits, targets, threshold=0.5, ignore_index=PAD_TOKEN):
    """
    Calculate binary classification accuracy from logits directly.

    Args:
    - logits (Tensor): Logits output from the model (without sigmoid activation).
    - targets (Tensor): Ground truth binary labels.
    - mask (Tensor, optional): Mask for valid data positions.

    Returns:
    - accuracy (float): The accuracy of predictions.
    """
    with torch.no_grad():
        predictions = (torch.sigmoid(logits) > threshold).float()
        correct = (predictions == targets).float()  # Compare with ground truths

        if ignore_index is not None:
            mask = targets != ignore_index
            correct = correct * mask  # Apply mask
            total = mask.sum()
        else:
            total = torch.tensor(targets.numel(), device=targets.device)

        accuracy = correct.sum() / total if total > 0 else torch.tensor(0.0, device=targets.device)
    return accuracy.item()

class SegmentationStats:
    """
    A class to track and accumulate segmentation statistics over multiple instances,
    such as utterances, and calculate average metrics. 

    Attributes:
        f1_accum (float): Accumulator for F1 score values.
        os_accum (float): Accumulator for over-segmentation score values.
        rval_accum (float): Accumulator for R-value (a combined metric of precision and recall) values.
        n (int): Counter for the number of instances statistics have been recorded for.

    Methods:
        update(stats): Update the accumulated stats with a new set of statistics.
        get_average_stats(): Calculate and return the average statistics across all recorded instances.
        print_stats(boundary_type, output_dir): Print the average statistics to the terminal and save to a file.
    """
    def __init__(self, logger):
        self.acc_accum = 0 
        self.f1_accum = 0
        self.os_accum = 0
        self.rval_accum = 0
        self.n = 0
        self.logger = logger 

    def update(self, stats):
        """Update the accumulated stats with new stats."""
        self.acc_accum += stats[0]
        self.f1_accum += stats[1]
        self.os_accum += stats[2]
        self.rval_accum += stats[3]
        self.n += 1

    def get_average_stats(self):
        """Calculate and return the average stats."""
        if self.n == 0:
            return [0, 0, 0, 0]  # Avoid division by zero
        return [
            self.acc_accum / self.n, 
            self.f1_accum / self.n,
            self.os_accum / self.n,
            self.rval_accum / self.n,
        ]

    def print_stats(self, boundary_type):
        """Print the average stats for a given boundary type and save to a file."""
        avg_stats = self.get_average_stats()
        stats_output = (
            f"{boundary_type} boundaries:\n"
            f"Accuracy: {avg_stats[0]*100:.2f}%\t"
            f"F-score: {avg_stats[1]*100:.2f}%\t"
            f"OS: {avg_stats[2]*100:.2f}%\t"
            f"R-value: {avg_stats[3]*100:.2f}%\n"
            + "-" * 75
        )
        self.logger.info(stats_output)

    def reset(self): 
        self.acc_accum = 0 
        self.f1_accum = 0
        self.os_accum = 0
        self.rval_accum = 0
        self.n = 0

