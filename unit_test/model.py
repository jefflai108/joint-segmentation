import unittest
import torch
import torch.nn as nn

from src.model import JointSegmentationModel

PAD_TOKEN = 1024

class TestJointSegmentationModel(unittest.TestCase):
    def setUp(self):
        # Initialize the model with some standard hyperparameters
        self.batch_size = 64 
        self.vocab_size = 1024 + 1
        self.d_model = 64
        self.nhead = 4
        self.num_encoder_layers = 6
        self.dim_feedforward = 256
        self.dropout = 0.1
        self.max_seq_length = 100
        self.lr = 0.01
        self.prediction_layers = (2, 4, 6)
        self.prediction_loss_weights = (0.25, 0.25, 0.50)
        self.label_smoothing_alpha = 0.2
        self.model = JointSegmentationModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length,
            lr=self.lr, 
            prediction_layers=self.prediction_layers, 
            prediction_loss_weights=self.prediction_loss_weights, 
            label_smoothing=self.label_smoothing_alpha, 
        )

    def test_initialization(self):
        # Test initializations of model components
        self.assertIsInstance(self.model.embedding, nn.Embedding, "Embedding layer not initialized correctly.")
        self.assertEqual(self.model.embedding.num_embeddings, self.vocab_size, "Incorrect vocabulary size.")

    def test_forward_pass(self):
        # Test the forward pass of the model
        src = torch.randint(0, self.vocab_size, (self.batch_size, self.max_seq_length))
        src_mask = None 
        src_key_padding_mask = torch.zeros(self.batch_size, self.max_seq_length).type(torch.bool)  # Assuming no padding for simplicity

        # make the input [sequence_length, batch_size] instead. Remember to also transpose the targets later 
        src = src.transpose(0, 1) 

        phone_pred, syllable_pred, word_pred = self.model.forward(src, src_mask, src_key_padding_mask)
        self.assertEqual(phone_pred.shape, torch.Size([self.max_seq_length, self.batch_size]), "Phone prediction output shape is incorrect.")
        self.assertEqual(syllable_pred.shape, torch.Size([self.max_seq_length, self.batch_size]), "Syllable prediction output shape is incorrect.")
        self.assertEqual(word_pred.shape, torch.Size([self.max_seq_length, self.batch_size]), "Word prediction output shape is incorrect.")

    def test_boundaries_metrics(self):
        predictions = (
            torch.randn(self.batch_size, self.max_seq_length), 
            torch.randn(self.batch_size, self.max_seq_length), 
            torch.randn(self.batch_size, self.max_seq_length)
        )
        targets = (
            torch.randint(0, 2, (self.batch_size, self.max_seq_length)), 
            torch.randint(0, 2, (self.batch_size, self.max_seq_length)), 
            torch.randint(0, 2, (self.batch_size, self.max_seq_length))
        )
        # masking 
        targets[0][:, -10:] = PAD_TOKEN 
        targets[1][:, -10:] = PAD_TOKEN 
        targets[2][:, -10:] = PAD_TOKEN 
        # make the input [sequence_length, batch_size] instead. Remember to also transpose the targets later 
        predictions = (predictions[0].transpose(0, 1),
                       predictions[1].transpose(0, 1),
                       predictions[2].transpose(0, 1))
        targets = (targets[0].transpose(0, 1), 
                   targets[1].transpose(0, 1),
                   targets[2].transpose(0, 1)) 

        # Assuming binary targets for a simple test
        phone_acc, syllable_acc, word_acc = self.model.compute_accuracies(*predictions, *targets)
        print(f'Phone acc is {phone_acc:.4f}, Syllable acc is {syllable_acc:.4f}, Word acc is {word_acc:.4f}')
        (phone_f1, syllable_f1, word_f1), (phone_rval, syllable_rval, word_rval), (phone_os, syllable_os, word_os) = self.model.compute_boundary_metrics(*predictions, *targets)
        print(f'Phone f1 is {phone_f1:.4f}, Syllable f1 is {syllable_f1:.4f}, Word f1 is {word_f1:.4f}')
        print(f'Phone rval is {phone_rval:.4f}, Syllable rval is {syllable_rval:.4f}, Word rval is {word_rval:.4f}')
        print(f'Phone os is {phone_os:.4f}, Syllable os is {syllable_os:.4f}, Word os is {word_os:.4f}')

    def test_loss_computation(self):
        predictions = (
            torch.randn(self.batch_size, self.max_seq_length), 
            torch.randn(self.batch_size, self.max_seq_length), 
            torch.randn(self.batch_size, self.max_seq_length)
        )
        targets = (
            torch.randint(0, 2, (self.batch_size, self.max_seq_length)), 
            torch.randint(0, 2, (self.batch_size, self.max_seq_length)), 
            torch.randint(0, 2, (self.batch_size, self.max_seq_length))
        )
        # masking 
        targets[0][:, -10:] = PAD_TOKEN 
        targets[1][:, -10:] = PAD_TOKEN 
        targets[2][:, -10:] = PAD_TOKEN 
        # make the input [sequence_length, batch_size] instead. Remember to also transpose the targets later 
        predictions = (predictions[0].transpose(0, 1),
                       predictions[1].transpose(0, 1),
                       predictions[2].transpose(0, 1))
        targets = (targets[0].transpose(0, 1), 
                   targets[1].transpose(0, 1),
                   targets[2].transpose(0, 1)) 

        # Assuming binary targets for a simple test
        loss = self.model.compute_total_weighted_loss(*predictions, *targets)
        print(loss)
        self.assertIsInstance(loss, torch.Tensor, "Loss is not computed as a tensor.")

    def test_train_step(self):
        # Test the train_step to ensure it does not crash and returns expected outputs
        src = torch.randint(0, self.vocab_size, (self.batch_size, self.max_seq_length))
        src_mask = None 
        src_key_padding_mask = torch.zeros(self.batch_size, self.max_seq_length, dtype=torch.bool)
        src_key_padding_mask[:, -10:] = True 

        phone_targets = torch.randint(0, 2, (self.batch_size, self.max_seq_length))
        syllable_targets = torch.randint(0, 2, (self.batch_size, self.max_seq_length))
        word_targets = torch.randint(0, 2, (self.batch_size, self.max_seq_length))
        # masking 
        phone_targets[:, -10:] = PAD_TOKEN 
        syllable_targets[:, -10:] = PAD_TOKEN 
        word_targets[:, -10:] = PAD_TOKEN 

        # make the input [sequence_length, batch_size] instead. Remember to also transpose the targets later 
        src = src.transpose(0, 1) 
        phone_targets = phone_targets.transpose(0, 1)
        syllable_targets = syllable_targets.transpose(0, 1)
        word_targets = word_targets.transpose(0, 1)

        for step in range(50): 
            loss, _, _, _, _ = self.model.train_step(src, phone_targets, syllable_targets, word_targets, src_mask, src_key_padding_mask, step, 4)
            print(f'Loss is {loss:.2f}')
            self.assertIsInstance(loss, float, "Train step should return a float loss.")
        self.model.eval_step(src, phone_targets, syllable_targets, word_targets, src_mask, src_key_padding_mask)

# Run the tests
if __name__ == '__main__':
    unittest.main()

