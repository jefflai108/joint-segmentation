import logging, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.modules import ( 
    PositionalEncoding, 
    DynamicWeightedBCEWithLogitsLoss, 
)
from src.utils import (
    calculate_accuracy, 
    calculate_boundary_metrics, 
)

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('JointSegmentationModel')

PAD_TOKEN = 1024

class JointSegmentationModel(nn.Module):
    """
    A neural network model designed for joint segmentation and prediction of linguistic units such as phones, 
    syllables, and words from input sequences. This model leverages the Transformer architecture, utilizing 
    a custom encoder setup for processing sequences. Predictions are made at different stages within the encoder 
    for phones and syllables, with a final prediction for words made from the last encoder layer's output.

    The model dynamically adjusts the loss computation based on the class distribution within each batch, which is 
    essential for dealing with imbalanced datasets often found in language tasks. It integrates multiple heads for 
    different prediction tasks and employs custom loss functions that handle potential class imbalances by dynamically 
    adjusting weights.

    Attributes:
        d_model (int): The number of expected features in the encoder inputs (also known as the embedding dimension).
        logger (logging.Logger): Logger for recording model-related messages.
        embedding (nn.Embedding): Embedding layer for input tokens.
        encoder (nn.TransformerEncoder): The Transformer encoder consisting of a stack of N encoder layers.
        pos_encoder (PositionalEncoding): Adds positional encodings to the input embeddings to preserve the notion of 
                                          sequence order.
        word_prediction_head (nn.Linear): Linear layer for predicting word-level outputs.
        syllable_prediction_head (nn.Linear): Linear layer for predicting syllable-level outputs.
        phone_prediction_head (nn.Linear): Linear layer for predicting phone-level outputs.
        phone_prediction_layer (int): The encoder layer at which phone predictions are made.
        syllable_prediction_layer (int): The encoder layer at which syllable predictions are made.
        phone_criterion (DynamicWeightedBCEWithLogitsLoss): Custom loss function for phone prediction.
        syllable_criterion (DynamicWeightedBCEWithLogitsLoss): Custom loss function for syllable prediction.
        word_criterion (DynamicWeightedBCEWithLogitsLoss): Custom loss function for word prediction.
        optimizer (torch.optim.Optimizer): Optimizer used for updating model parameters.

    Methods:
        forward: Process input through the model and returns logits for phone, syllable, and word predictions.
        train_step: Executes a training step including a forward pass, loss computation, and a backward pass.
        eval_step: Evaluates the model on a given input batch and computes metrics such as loss and accuracy.

    Example:
        >>> model = JointSegmentationModel(vocab_size=1000, d_model=512, nhead=8, num_encoder_layers=6,
                                           dim_feedforward=2048, dropout=0.1, max_seq_length=500, lr=0.001)
        >>> input_ids = torch.randint(0, 1000, (32, 100))
        >>> phone_targets = torch.randint(0, 2, (32, 100))
        >>> syllable_targets = torch.randint(0, 2, (32, 100))
        >>> word_targets = torch.randint(0, 2, (32, 100))
        >>> src_mask = torch.randint(0, 2, (32, 100)).bool()
        >>> loss, (phone_acc, syllable_acc, word_acc) = model.train_step(input_ids, phone_targets, syllable_targets,
                                                                         word_targets, src_mask=src_mask)
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, max_seq_length, lr, prediction_layers=(2, 4, 6), prediction_loss_weights=(0.25, 0.25, 0.50), label_smoothing=0.0, logger=None):
        super(JointSegmentationModel, self).__init__()
        self.d_model = d_model
        self.logger = logger or logging.getLogger('JointSegmentationModel')
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)

        self.word_prediction_head = nn.Linear(d_model, 1)
        self.syllable_prediction_head = nn.Linear(d_model, 1)
        self.phone_prediction_head = nn.Linear(d_model, 1)
        
        # Define the layers at which predictions are made (layer number start at 1) 
        self.phone_prediction_layer = prediction_layers[0]
        self.syllable_prediction_layer = prediction_layers[1]
        assert prediction_layers[2] == num_encoder_layers

        # set frame-wise F1 / binary classification decision threshold. Make adjustable later on. 
        # >decision_threshold is 1 and <= decision_threshold is 0. 
        self.decision_threshold = 0.5

        self.init_weights()

        # Set up three separate loss functions with specific positive weights
        # fix this by online BCEWithLogitsLoss 
        self.phone_criterion = DynamicWeightedBCEWithLogitsLoss(ignore_index=PAD_TOKEN, alpha=label_smoothing)
        self.syllable_criterion = DynamicWeightedBCEWithLogitsLoss(ignore_index=PAD_TOKEN, alpha=label_smoothing)
        self.word_criterion = DynamicWeightedBCEWithLogitsLoss(ignore_index=PAD_TOKEN, alpha=label_smoothing)
        self.phone_loss_weight = prediction_loss_weights[0]
        self.syllable_loss_weight = prediction_loss_weights[1]
        self.word_loss_weight = prediction_loss_weights[2]

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.print_parameters()

    def set_decision_threshold(self, new_threhsold): 
        self.decision_threshold = new_threhsold

    def print_parameters(self):
        total_params = 0
        for name, param in self.named_parameters():
            self.logger.info(f"{name}, shape: {param.size()}")
            total_params += param.numel()
        self.logger.info(f"Total parameters: {total_params}")

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

        # Initialize encoder layers
        for layer in self.encoder.layers:
            # Initialize all weights using the Glorot method adapted for uniform distribution
            for p in layer.parameters():
                if p.dim() > 1:  # This checks if the parameter is a weight matrix and not a bias vector
                    nn.init.xavier_uniform_(p)
        
        # Initialize prediction heads. 
        # Init the bias term as a small negative at the start of training to avoid excessive 1s predictions
        nn.init.xavier_uniform_(self.word_prediction_head.weight)
        nn.init.constant_(self.word_prediction_head.bias, -0.1)
        nn.init.xavier_uniform_(self.syllable_prediction_head.weight)
        nn.init.constant_(self.syllable_prediction_head.bias, -0.1)
        nn.init.xavier_uniform_(self.phone_prediction_head.weight)
        nn.init.constant_(self.phone_prediction_head.bias, -0.1)

    def forward(self, x, src_mask, src_key_padding_mask):
        """
        Processes input data through the embedding layer, positional encoding, and a series of transformer encoder layers. 
        Makes predictions at specified encoder layers for phone and syllable classes and a final prediction for word class 
        from the last encoder layer output. 

        Args:
            x (Tensor): The input data consisting of token indices. Shape [seq_length, batch_size].
            src_mask (Tensor, optional): The source mask tensor, which is used in the transformer to mask out certain 
                                         positions within the input sequence during attention calculations. This is typically 
                                         used for preventing attention to padding positions. Shape should match the 
                                         dimensions suitable for the transformer's attention mechanism.
            src_key_padding_mask (Tensor, optional): Mask indicating which elements of the input batch are padding. This 
                                                     mask is applied at multiple points in the network to ensure that padding 
                                                     elements do not affect the loss calculations and model's predictions.
                                                     Shape [batch_size, seq_length].

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the logits for phone predictions, syllable predictions, 
                                           and word predictions. These are computed at different layers as specified 
                                           by the model configuration.
        """
        # Embedding layer which expects x of shape [seq_length, batch_size]
        x = self.embedding(x)  # Output shape will be [seq_length, batch_size, d_model]
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Initialize prediction variables
        phone_prediction, syllable_prediction, word_prediction = None, None, None 
        for i, layer in enumerate(self.encoder.layers):
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

            # Make predictions at specified layers without applying sigmoid
            if i == self.phone_prediction_layer - 1:
                phone_prediction = self.phone_prediction_head(x)  # Outputs are logits
            if i == self.syllable_prediction_layer - 1:
                syllable_prediction = self.syllable_prediction_head(x)  # Outputs are logits

        # Prediction from the last layer
        word_prediction = self.word_prediction_head(x)  # Outputs are logits
        
        return phone_prediction.squeeze(-1), syllable_prediction.squeeze(-1), word_prediction.squeeze(-1)

    def train_step(self, src, phone_targets, syllable_targets, word_targets, src_mask, src_key_padding_mask, current_step, accumulation_steps=4):
        """
        Performs a single training step which includes a forward pass, loss computation, and a backward pass with 
        gradient accumulation. This method is specifically designed to handle multiple prediction tasks and apply 
        gradient accumulation for more stable updates when using large batches or small learning rates.

        Args:
            src (Tensor): Input tensor containing token indices for the batch.
            phone_targets (Tensor): Ground truth labels for phone predictions, must be binary encoded.
            syllable_targets (Tensor): Ground truth labels for syllable predictions, must be binary encoded.
            word_targets (Tensor): Ground truth labels for word predictions, must be binary encoded.
            src_mask (Tensor, optional): An optional tensor that masks out certain positions from the attention mechanism 
                                         within the transformer layers.
            src_key_padding_mask (Tensor, optional): An optional tensor that indicates which positions in the batch are 
                                                     padding and should not be considered in loss calculations or model updates.
            current_step (int): The current training step number, used to manage when to perform optimizer steps in 
                                the context of gradient accumulation.
            accumulation_steps (int): The number of steps over which gradients are accumulated before updating model 
                                      parameters. Useful for simulating larger batch sizes or for more stable training 
                                      with very small batches.

        Returns:
            Tuple[float, Tuple[float, float, float]]: A tuple containing the total loss for the batch, scaled by the number 
                                                      of accumulation steps, and a tuple of accuracy values for each prediction 
                                                      task (phone, syllable, word).
        """

        self.train()
        phone_prediction, syllable_prediction, word_prediction = self.compute_predictions(src, src_mask, src_key_padding_mask)
        total_loss = self.compute_total_weighted_loss(phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets) / accumulation_steps
        total_loss.backward()
        if current_step % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        accuracies = self.compute_accuracies(phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets, self.decision_threshold)
        f1s, rvals, oss, accs = self.compute_boundary_metrics(phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets, self.decision_threshold)
        return total_loss.item() * accumulation_steps, accs, f1s, rvals, oss

    def eval_step(self, src, phone_targets, syllable_targets, word_targets, src_mask=None, src_key_padding_mask=None):
        """Evaluates the model on the provided batch of data.

        Args:
            src (Tensor): Input tensor containing token indices.
            phone_targets (Tensor): Ground truth binary labels for phone predictions.
            syllable_targets (Tensor): Ground truth binary labels for syllable predictions.
            word_targets (Tensor): Ground truth binary labels for word predictions.
            src_mask (Tensor, optional): Optional mask for the source input tokens.
            src_key_padding_mask (Tensor, optional): Mask for padding elements in the input data.

        Returns:
            Tuple containing:
                - Total loss for the batch.
                - A tuple of accuracy metrics for each prediction level (phones, syllables, words).
        """
        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            phone_prediction, syllable_prediction, word_prediction = self.compute_predictions(src, src_mask, src_key_padding_mask)
            total_loss = self.compute_total_weighted_loss(phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets)
            accuracies = self.compute_accuracies(phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets, self.decision_threshold)
            f1s, rvals, oss, accs = self.compute_boundary_metrics(phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets, self.decision_threshold)
        return total_loss.item(), accs, f1s, rvals, oss

    def compute_predictions(self, src, src_mask, src_key_padding_mask):
        """Compute the model predictions for phone, syllable, and word levels."""
        return self.forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def compute_accuracies(self, phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets, decision_threshold):
        """Calculate accuracies for each prediction level."""
        phone_acc = calculate_accuracy(phone_prediction, phone_targets, threshold=decision_threshold)
        syllable_acc = calculate_accuracy(syllable_prediction, syllable_targets, threshold=decision_threshold)
        word_acc = calculate_accuracy(word_prediction, word_targets, threshold=decision_threshold)
        return phone_acc, syllable_acc, word_acc

    def compute_boundary_metrics(self, phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets, decision_threshold):
        """Calculate boundary_metrics for each prediction level."""
        phone_f1, phone_rval, phone_os, phone_acc = calculate_boundary_metrics(phone_prediction, phone_targets, threshold=decision_threshold)
        syllable_f1, syllable_rval, syllable_os, syllable_acc = calculate_boundary_metrics(syllable_prediction, syllable_targets, threshold=decision_threshold)
        word_f1, word_rval, word_os, word_acc = calculate_boundary_metrics(word_prediction, word_targets, threshold=decision_threshold)
        return (phone_f1, syllable_f1, word_f1), (phone_rval, syllable_rval, word_rval), (phone_os, syllable_os, word_os), (phone_acc, syllable_acc, word_acc)

    def compute_total_weighted_loss(self, phone_prediction, syllable_prediction, word_prediction, phone_targets, syllable_targets, word_targets):
        """
        Compute the individual losses for phone, syllable, and word predictions and return the total loss weighted 
        by predefined loss weights.

        Args:
            phone_prediction (Tensor): Logits for phone predictions.
            syllable_prediction (Tensor): Logits for syllable predictions.
            word_prediction (Tensor): Logits for word predictions.
            phone_targets (Tensor): Ground truth binary labels for phone predictions.
            syllable_targets (Tensor): Ground truth binary labels for syllable predictions.
            word_targets (Tensor): Ground truth binary labels for word predictions.

        Returns:
            Tensor: The total weighted loss.
        """
        # Compute losses for each prediction stage using their specific targets
        phone_loss = self.phone_criterion(phone_prediction, phone_targets)
        syllable_loss = self.syllable_criterion(syllable_prediction, syllable_targets)
        word_loss = self.word_criterion(word_prediction, word_targets)

        # Calculate weighted total loss
        total_loss = (self.phone_loss_weight * phone_loss +
                      self.syllable_loss_weight * syllable_loss +
                      self.word_loss_weight * word_loss)
        return total_loss

    def save_model(self, epoch, file_save_path):
        if not os.path.exists(os.path.dirname(file_save_path)):
            os.makedirs(os.path.dirname(file_save_path))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'decision_threshold': self.decision_threshold, 
        }, file_save_path)

    def load_model(self, file_load_path):
        checkpoint = torch.load(file_load_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        decision_threshold = checkpoint['decision_threshold']
        self.logger.info(f"Model and optimizer loaded. Resuming from epoch {epoch}.")
        return epoch, decision_threshold
