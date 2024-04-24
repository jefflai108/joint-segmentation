import random 
import time
import logging, os 
import argparse

import numpy as np
import torch

from src.model import JointSegmentationModel
from src.data import get_train_loader, get_eval_loader, DatasetConfig
from src.utils import SegmentationStats

PAD_TOKEN = 1024

def configure_logging(save_dir):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)  # Set the logger level

    # Ensure no duplicate handlers are added
    if not logger.handlers:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

        # File handler
        file_handler = logging.FileHandler(os.path.join(save_dir, 'training.log'), 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Word codebook learning.")

	# Paths
    parser.add_argument("--train_token_file_path", type=str, required=True, help="Path to the train token file.")
    parser.add_argument("--dev_token_file_path", type=str, required=True, help="Path to the dev token file.")
    parser.add_argument("--train_phone_label_file_path", type=str, required=True, help="Path to the train phone label file.")
    parser.add_argument("--dev_phone_label_file_path", type=str, required=True, help="Path to the dev phone label file.")
    parser.add_argument("--train_syllable_label_file_path", type=str, required=True, help="Path to the train syllable label file.")
    parser.add_argument("--dev_syllable_label_file_path", type=str, required=True, help="Path to the dev syllable label file.")
    parser.add_argument("--train_word_label_file_path", type=str, required=True, help="Path to the train word label file.")
    parser.add_argument("--dev_word_label_file_path", type=str, required=True, help="Path to the dev word label file.")
    parser.add_argument('--save_dir', type=str, default='exp/debug/', help='Directory to save model checkpoints and logs')

    # Batch size 
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models.")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers in the transformer.")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network model.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--max_seq_length", type=int, default=120, help="Maximum sequence length.")
    parser.add_argument("--label_smoothing_alpha", type=float, default=0.0,
                        help="Label smoothing parameter between 0 and 1. 0 disables label smoothing.")
    parser.add_argument('--prediction_layers', nargs='+', type=int, default=[2, 4, 6],
                        help='List of encoder layers to use for predictions, separated by space')
    parser.add_argument('--prediction_loss_weights', nargs='+', type=float, default=[0.25, 0.25, 0.50],
                        help='List of weights for prediction losses, corresponding to each prediction layer, separated by space')

    # Training specifics
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--log_interval', type=int, default=100, help='Log loss every this many intervals')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--gradient_acc_steps', type=int, default=2, help='number of training steps accumulated')

    return parser.parse_args()

def adjust_learning_rate(optimizer, step, total_steps, peak_lr, end_lr=1e-6):
    warmup_steps = int(total_steps * 0.2)  # 20% of total steps for warm-up
    decay_steps = int(total_steps * 0.8)  # 80% of total steps for decay

    if step < warmup_steps:
        lr = peak_lr * step / warmup_steps
    elif step < warmup_steps + decay_steps:
        step_into_decay = step - warmup_steps
        lr = peak_lr * (end_lr / peak_lr) ** (step_into_decay / decay_steps)
    else:
        lr = end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def tune_decision_threshold(dev_loader, model, device, threshold_range=0.1, max_batches=100, prediction_loss_weights=[0.25, 0.25, 0.50]):
    """
    Dynamically adjusts the decision threshold of the model based on weighted F1 scores.

    Parameters:
    - dev_loader (DataLoader): DataLoader for development dataset.
    - model (Model): The model being tuned.
    - threshold_range (float): Range around the current threshold to explore.
    - max_batches (int): Maximum number of batches to process for each threshold.
    - prediction_loss_weights (list of float): Weights for F1 scores of different prediction tasks.
    """
	# Start with the current threshold
    current_threshold = model.decision_threshold
    
    # Create a range of thresholds around the current one within the specified range
    decision_thresholds = torch.arange(max(0, current_threshold - threshold_range), 
                                       min(1, current_threshold + threshold_range) + 0.025, 
                                       0.025)  

    max_f1 = 0
    optimal_threshold = current_threshold  # Initialize with the current threshold as baseline

    for threshold in decision_thresholds:
        weighted_f1 = 0 
        count = 0

        for batch_idx, (uttids, tokens, phone_labels, syllable_labels, word_labels, src_mask, src_key_padding_mask) in enumerate(dev_loader):
            if batch_idx >= max_batches:
                break  # Limit the number of batches processed

            with torch.no_grad():
                # Move tensors to the right device
                tokens, phone_labels, syllable_labels, word_labels = tokens.to(device), phone_labels.to(device), syllable_labels.to(device), word_labels.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)

                # make the input [sequence_length, batch_size] instead 
                tokens = tokens.transpose(0, 1) 
                phone_labels = phone_labels.transpose(0, 1)
                syllable_labels = syllable_labels.transpose(0, 1)
                word_labels = word_labels.transpose(0, 1)

                # Compute model predictions for the current batch
                phone_prediction, syllable_prediction, word_prediction = model.compute_predictions(tokens, src_mask, src_key_padding_mask)
                # Calculate boundary metrics at the current threshold
                f1_scores, _, _, _ = model.compute_boundary_metrics(phone_prediction, syllable_prediction, word_prediction, 
                                                                    phone_labels, syllable_labels, word_labels, decision_threshold=threshold)

                # Calculate weighted F1 score for this batch
                batch_weighted_f1 = sum(f * w for f, w in zip(f1_scores, prediction_loss_weights))
                weighted_f1 += batch_weighted_f1
                count += 1

        # Calculate average F1 score across all tasks and samples
        avg_f1 = weighted_f1 / count if count > 0 else 0

        if avg_f1 > max_f1:
            max_f1 = avg_f1
            optimal_threshold = threshold

    # Update the model's decision threshold if a better one is found
    model.set_decision_threshold(optimal_threshold)
    return optimal_threshold
   
def train(model, train_dataloader, dev_dataloader, current_epoch, epochs, logger, log_interval, peak_lr, gradient_acc_steps, decision_threshold, prediction_loss_weights, save_dir, device):
    total_steps = epochs * len(train_dataloader)
    current_step = current_epoch * len(train_dataloader)
    best_loss  = float('inf')
    no_improve_epochs = 0
    patience = 5  # Number of epochs to wait before early stopping
    
    for epoch in range(current_epoch, epochs):
        def _train(current_step):
            model.train()

            phn_stats_tracker = SegmentationStats(logger)
            syllable_stats_tracker = SegmentationStats(logger)
            word_stats_tracker = SegmentationStats(logger)
            total_loss = 0 
            log_loss, log_phone_f1, log_syllable_f1, log_word_f1 = 0, 0, 0, 0
            for batch_idx, (uttids, tokens, phone_labels, syllable_labels, word_labels, src_mask, src_key_padding_mask) in enumerate(train_dataloader):
                current_step += 1
                adjust_learning_rate(model.optimizer, current_step, total_steps, peak_lr)

                # Move tensors to the right device
                tokens, phone_labels, syllable_labels, word_labels = tokens.to(device), phone_labels.to(device), syllable_labels.to(device), word_labels.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)

                # make the input [sequence_length, batch_size] instead 
                tokens = tokens.transpose(0, 1) 
                phone_labels = phone_labels.transpose(0, 1)
                syllable_labels = syllable_labels.transpose(0, 1)
                word_labels = word_labels.transpose(0, 1)

                # Forward pass
                loss, accuracies, f1s, rvals, oss = model.train_step(tokens, phone_labels, syllable_labels, word_labels, src_mask, src_key_padding_mask, current_step, gradient_acc_steps)
                log_loss += loss
                log_phone_f1 += f1s[0]
                log_syllable_f1 += f1s[1]
                log_word_f1 += f1s[2]

                # continuous accumulation
                total_loss += loss  
                phn_stats_tracker.update((accuracies[0], f1s[0], oss[0], rvals[0]))
                syllable_stats_tracker.update((accuracies[1], f1s[1], oss[1], rvals[1]))
                word_stats_tracker.update((accuracies[2], f1s[2], oss[2], rvals[2]))
        
                # Log loss every log_interval steps
                if (batch_idx + 1) % log_interval == 0:
                    log_avg_loss = log_loss / log_interval
                    log_avg_phone_f1 = log_phone_f1 / log_interval
                    log_avg_syllable_f1 = log_syllable_f1 / log_interval
                    log_avg_word_f1 = log_word_f1 / log_interval

                    current_lr = model.optimizer.param_groups[0]['lr']
                    #logger.info(f"Epoch: {epoch}, Step: {batch_idx+1}, Avg Loss: {log_avg_loss:.4f}, Avg Phone Acc: {log_avg_phone_acc:.4f}, Avg Syllable Acc: {log_avg_syllable_acc:.4f}, Avg Word Acc: {log_avg_word_acc:.4f}, LR: {current_lr:.5f}")
                    logger.info(f"Epoch: {epoch}, Step: {batch_idx+1}, Avg Loss: {log_avg_loss:.4f}, Avg Phone F1: {log_avg_phone_f1:.4f}, Avg Syllable F1: {log_avg_syllable_f1:.4f}, Avg Word F1: {log_avg_word_f1:.4f}, LR: {current_lr:.5f}")
                    log_loss, log_phone_f1, log_syllable_f1, log_word_f1 = 0, 0, 0, 0
    
                # Visualize model predictions AND Dynamically adjust decision threhsold every 10 * log_interval 
                if (batch_idx + 1) % (log_interval * 10) == 0:
                    with torch.no_grad(): 
                        # adjust decision threshold on a subset (200 batches) of dev_dataloader --> comment this out for now 
                        #decision_threshold = tune_decision_threshold(dev_dataloader, model, device, 0.1, 200, prediction_loss_weights)
                        #logger.info(f"*** Updated decision threshold to {decision_threshold} based on dev evaluation. ***")

                        # print model predictions to terminal 
                        dummy_phone_logits, dummy_syllable_logits, dummy_word_logits = model.compute_predictions(tokens, src_mask, src_key_padding_mask)
                        dummy_phone_predictions = (torch.sigmoid(dummy_phone_logits) > decision_threshold).float()
                        dummy_syllable_predictions = (torch.sigmoid(dummy_syllable_logits) > decision_threshold).float()
                        dummy_word_predictions = (torch.sigmoid(dummy_word_logits) > decision_threshold).float()

                        sample_size = min(5, tokens.shape[1]) # Ensure we do not exceed the batch size
                        random_indices = np.random.choice(tokens.shape[1], size=sample_size, replace=False)
                        for idx in random_indices:
                            sampled_phone_gt = phone_labels[:, idx].cpu().numpy()
                            sampled_phone_pred = dummy_phone_predictions[:, idx].cpu().numpy().astype(int)
                            phone_mask = sampled_phone_gt != PAD_TOKEN

                            sampled_syllable_gt = syllable_labels[:, idx].cpu().numpy()
                            sampled_syllable_pred = dummy_syllable_predictions[:, idx].cpu().numpy().astype(int)
                            syllable_mask = sampled_syllable_gt != PAD_TOKEN

                            sampled_word_gt = word_labels[:, idx].cpu().numpy()
                            sampled_word_pred = dummy_word_predictions[:, idx].cpu().numpy().astype(int)
                            word_mask = sampled_word_gt != PAD_TOKEN

                            logger.info(f"Sampled Phone Prediction: {sampled_phone_pred[phone_mask]}, Ground Truth: {sampled_phone_gt[phone_mask]}")
                            logger.info(f"Sampled Syllable Prediction: {sampled_syllable_pred[syllable_mask]}, Ground Truth: {sampled_syllable_gt[syllable_mask]}")
                            logger.info(f"Sampled Word Prediction: {sampled_word_pred[word_mask]}, Ground Truth: {sampled_word_gt[word_mask]}")

            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'===> Epoch {epoch} Complete: Avg Loss: {avg_loss:.4f}\n')
            phn_stats_tracker.print_stats('Phone')
            syllable_stats_tracker.print_stats('Syllable')
            word_stats_tracker.print_stats('Word')
            phn_stats_tracker.reset()
            syllable_stats_tracker.reset()
            word_stats_tracker.reset()

            return avg_loss, current_step

        def _val():
            model.eval()

            eval_phn_stats_tracker = SegmentationStats(logger)
            eval_syllable_stats_tracker = SegmentationStats(logger)
            eval_word_stats_tracker = SegmentationStats(logger)
            total_loss = 0 
            with torch.no_grad():
                # adjust decision threshold on the full set of dev_dataloader 
                decision_threshold = tune_decision_threshold(dev_dataloader, model, device, 0.1, len(dev_dataloader), prediction_loss_weights)
                logger.info(f"*** Updated decision threshold to {decision_threshold} based on dev evaluation. *** ")

                for uttids, tokens, phone_labels, syllable_labels, word_labels, src_mask, src_key_padding_mask in dev_dataloader:
                    # Move tensors to the right device
                    tokens, phone_labels, syllable_labels, word_labels = tokens.to(device), phone_labels.to(device), syllable_labels.to(device), word_labels.to(device)
                    src_key_padding_mask = src_key_padding_mask.to(device)

                    # make the input [sequence_length, batch_size] instead 
                    tokens = tokens.transpose(0, 1) 
                    phone_labels = phone_labels.transpose(0, 1)
                    syllable_labels = syllable_labels.transpose(0, 1)
                    word_labels = word_labels.transpose(0, 1)

                    loss, accuracies, f1s, rvals, oss = model.eval_step(tokens, phone_labels, syllable_labels, word_labels, src_mask, src_key_padding_mask)
                    total_loss += loss
                    eval_phn_stats_tracker.update((accuracies[0], f1s[0], oss[0], rvals[0]))
                    eval_syllable_stats_tracker.update((accuracies[1], f1s[1], oss[1], rvals[1]))
                    eval_word_stats_tracker.update((accuracies[2], f1s[2], oss[2], rvals[2]))
            
            avg_loss = total_loss / len(dev_dataloader)
            logger.info(f'===> Validation set: Avg loss: {avg_loss:.4f}.')
            eval_phn_stats_tracker.print_stats('Phone')
            eval_syllable_stats_tracker.print_stats('Syllable')
            eval_word_stats_tracker.print_stats('Word')
            eval_phn_stats_tracker.reset()
            eval_syllable_stats_tracker.reset()
            eval_word_stats_tracker.reset()

            return avg_loss

        train_loss, current_step = _train(current_step)
        val_loss = _val()
        
        # Inside the training loop, after calculating the validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0  # Reset counter as there is improvement
            best_model_save_path = os.path.join(save_dir, f'best_loss_model.pth')
            model.save_model(epoch, best_model_save_path)
        else:
            no_improve_epochs += 1

        # Saving model at each epoch
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}_loss_{val_loss:.4f}.pth')
        model.save_model(epoch, model_save_path)
        logger.info(f'Model saved to {model_save_path}')
        
        if no_improve_epochs >= patience:
            logger.info(f'Early stopping triggered after {patience} epochs without improvement.')
            break  # Break out of the loop to stop training

def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA, set this as well
    
    # Additional configuration to promote deterministic behavior:
    # Note: This might impact performance (trade-off determinism for speed).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(seed=418)
    
    args = parse_args()
    
    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configure logging to file and console
    logger = configure_logging(args.save_dir)

    # Setup device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize the model
    model = JointSegmentationModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        lr=args.learning_rate,
        prediction_layers=args.prediction_layers, 
        prediction_loss_weights=args.prediction_loss_weights, 
        label_smoothing=args.label_smoothing_alpha, 
        logger=logger, 
    )
    model = model.to(device)
    if os.path.exists(os.path.join(args.save_dir, f'best_loss_model.pth')):
        current_epoch, decision_threshold = model.load_model(os.path.join(args.save_dir, f'best_loss_model.pth'))
        model.set_decision_threshold(decision_threshold)
    else: 
        current_epoch = 0 
        decision_threshold = 0.5

    # Setup dataset configs
    train_dataset_config = DatasetConfig(
        token_file_path=args.train_token_file_path, 
        phone_label_file_path=args.train_phone_label_file_path, 
        syllable_label_file_path=args.train_syllable_label_file_path, 
        word_label_file_path=args.train_word_label_file_path, 
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        shuffle=True,
        num_workers=1
    )
    dev_dataset_config = DatasetConfig(
        token_file_path=args.dev_token_file_path, 
        phone_label_file_path=args.dev_phone_label_file_path, 
        syllable_label_file_path=args.dev_syllable_label_file_path, 
        word_label_file_path=args.dev_word_label_file_path, 
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        shuffle=False,
        num_workers=1
    )

    # Setup dataloaders 
    train_dataloader = get_train_loader(train_dataset_config)
    dev_dataloader = get_eval_loader(dev_dataset_config)
    
    # Run the training
    train(model, train_dataloader, dev_dataloader, current_epoch, args.epochs, logger, args.log_interval, args.learning_rate, args.gradient_acc_steps, decision_threshold, args.prediction_loss_weights, args.save_dir, device)

if __name__ == "__main__":  
    main()

