import os, re 

import torch 
import editdistance
import matplotlib.pyplot as plt

PAD_TOKEN = 1024
SOS_TOKEN = 1025 
EOS_TOKEN = 1026 

def deduplicate_segment(segment):
    """De-duplicate consecutive elements in a segment."""
    if segment.size(0) > 1:
        unique_mask = torch.cat([segment[:-1] != segment[1:], torch.tensor([True], dtype=torch.bool)])
        return segment[unique_mask]
    return segment

def preprocess_and_deduplicate(seq, stop_token=EOS_TOKEN):
    """Process and deduplicate each sequence in the batch."""
    processed_batch = []
    for sequence in seq:
        sequence = torch.tensor(sequence) if not isinstance(sequence, torch.Tensor) else sequence
        stop_idx = (sequence == stop_token).nonzero(as_tuple=True)[0]
        truncated_seq = sequence[:stop_idx[0].item()] if len(stop_idx) > 0 else sequence
        deduplicated_seq = deduplicate_segment(truncated_seq)
        processed_batch.append(deduplicated_seq.tolist())
    return processed_batch

def batch_unit_edit_distance(reference_batch, hypothesis_batch):
    total_unit_edit_dist = 0
    for reference, hypothesis in zip(reference_batch, hypothesis_batch):
        reference_processed = preprocess_and_deduplicate([reference])
        hypothesis_processed = preprocess_and_deduplicate([hypothesis])
        
        # Calculate edit distance for each pair in the batch
        edit_dist = editdistance.eval(reference_processed[0], hypothesis_processed[0])
        unit_edit_dist = edit_dist / max(len(reference_processed[0]), 1)  # Avoid division by zero
        total_unit_edit_dist += unit_edit_dist
    return total_unit_edit_dist

def filter_padded_rows(X, Y, padding_x=0.0, padding_y=PAD_TOKEN):
    """
    Removes rows in the batch dimension where both the entire row of X and Y are padded.

    Parameters:
    - X (torch.Tensor): A 3D tensor with padding value of 0.0.
    - Y (torch.Tensor): A 2D tensor with padding value of PAD_TOKEN.
    - padding_x (float): Padding value for X.
    - padding_y (int): Padding value for Y.

    Returns:
    - torch.Tensor: Filtered X without fully padded rows.
    - torch.Tensor: Filtered Y without fully padded rows.
    """
    # Ensure X and Y have the same batch size
    if X.size(0) != Y.size(0):
        raise ValueError("X and Y must have the same size in the batch dimension.")

    # Identify rows in Y that are fully padded
    Y_padded_rows = (Y == padding_y).all(dim=1)

    # Identify rows in X that are fully padded
    X_padded_rows = (X == padding_x).all(dim=2).all(dim=1)

    # Find rows where both X and Y are fully padded
    both_padded = Y_padded_rows & X_padded_rows

    # Filter out fully padded rows from both X and Y
    X_filtered = X[~both_padded]
    Y_filtered = Y[~both_padded]

    return X_filtered, Y_filtered

def trim_pad_tokens(tensor, pad_token=PAD_TOKEN):
    """
    Trims trailing PAD tokens from a 2D tensor along its last dimension.
    
    Parameters:
    - tensor: A 2D PyTorch tensor from which to remove trailing PAD tokens.
    - pad_token: The token used for padding sequences.
    
    Returns:
    - A 2D PyTorch tensor with trailing PAD tokens removed from each sequence.
    """
    # Creating a mask for non-PAD tokens
    mask = tensor != pad_token
    
    # Calculating lengths of non-PAD sequences for each row
    lengths = mask.sum(dim=1)
    
    # Finding the maximum length among non-PAD sequences
    max_length = lengths.max().item()
    
    # Creating a new tensor to hold trimmed sequences
    trimmed_tensor = torch.full((tensor.shape[0], max_length), pad_token, dtype=tensor.dtype)
    
    for i, length in enumerate(lengths):
        trimmed_tensor[i, :length] = tensor[i, :length]
    
    return trimmed_tensor

def find_top_modelckpt(exp_dir): 
    # Regex to extract epoch number and loss from the file name
    pattern = re.compile(r'model_epoch_(\d+)_loss_([0-9.]+)\.pth')

    # List all files and filter out checkpoint files, extracting epoch and loss
    checkpoints = []
    for filename in os.listdir(exp_dir):
        match = pattern.search(filename)
        if match:
            epoch, loss = match.groups()
            checkpoints.append((filename, int(epoch), float(loss)))

    # Sort the checkpoints based on loss
    checkpoints.sort(key=lambda x: x[2])

    # Select the top 5 checkpoints with the lowest loss
    top_checkpoints = checkpoints[:5]

    return top_checkpoints

def average_top_modelckpt(exp_dir, model, averaged_checkpoint_path): 
    top_checkpoints = find_top_modelckpt(exp_dir)
    
    # Initialize an average state dictionary
    avg_state_dict = {}

    # Load each of the top 5 checkpoints and accumulate their state dictionaries
    for filename, _, _ in top_checkpoints:
        checkpoint_path = os.path.join(exp_dir, filename)
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        
        if not avg_state_dict:  # If avg_state_dict is empty, initialize it with zero tensors
            for key, tensor in model_state_dict.items():
                avg_state_dict[key] = torch.zeros_like(tensor)
        
        # Accumulate the model parameters
        for key, tensor in model_state_dict.items():
            avg_state_dict[key] += tensor

    # Divide each parameter by the number of checkpoints to get the average
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(top_checkpoints)

    torch.save({
        'epoch': max(epoch for _, epoch, _ in top_checkpoints),  # Use the latest epoch
        'model_state_dict': avg_state_dict,
        'optimizer_state_dict': checkpoint['optimizer_state_dict']  # Optionally include an averaged or reset optimizer state dict
    }, averaged_checkpoint_path)

def calculate_accuracy(output_reshaped, tgt_reshaped, pad_token=PAD_TOKEN):
    """
    Calculates the prediction accuracy by comparing the predicted labels against the true labels, 
    excluding any instances where the target is a PAD_TOKEN.
    
    Parameters:
    - output_reshaped (Tensor): The model's output logits reshaped to (N, num_classes), where N is 
      the total number of samples in the batch, and num_classes is the number of possible labels.
    - tgt_reshaped (Tensor): The ground truth labels reshaped to a vector of length N.
    
    Returns:
    - accuracy (float): The proportion of correct predictions over the total number of non-pad predictions.
    """
    pred = torch.argmax(output_reshaped, dim=1)  # Get the index of the max log-probability.
    valid_indices = tgt_reshaped != pad_token  # Identify non-PAD_TOKEN elements.
    
    # Filter out PAD_TOKEN predictions and targets.
    valid_preds = pred[valid_indices]
    valid_targets = tgt_reshaped[valid_indices]
    
    # Count how many predictions match the target labels.
    correct = valid_preds.eq(valid_targets).sum().item()
    
    # Calculate the total number of non-pad labels.
    total = valid_targets.size(0)
    
    # Calculate accuracy, avoiding division by zero.
    accuracy = correct / total if total > 0 else 0

    return accuracy, valid_preds, valid_targets

def plot_pred_label_distribution(predictions, ground_truths): 
    # Sort the predictions and ground truths by label for consistency
    predictions_sorted = sorted(predictions, key=lambda x: x[0])
    ground_truths_sorted = sorted(ground_truths, key=lambda x: x[0])

    # Unpack the labels and counts
    pred_labels, pred_counts = zip(*predictions_sorted)
    gt_labels, gt_counts = zip(*ground_truths_sorted)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both distributions
    ax.bar(pred_labels, pred_counts, width=0.4, align='center', alpha=0.6, label='Predictions')
    ax.bar(gt_labels, gt_counts, width=0.4, align='edge', alpha=0.6, label='Ground Truths')

    # Adding labels and title
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Prediction and Ground Truth Distributions')
    ax.legend()

    # Show plot
    plt.xticks(rotation=90) # Rotate labels to avoid overlap
    plt.show()
    plt.savefig('shit.png')

