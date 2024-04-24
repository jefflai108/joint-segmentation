import unittest
import torch 
from torch.utils.data import DataLoader

from src.data import BoundaryDataset, collate_fn

class TestBoundaryDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = BoundaryDataset(
            '/data/sls/scratch/clai24/word-seg/joint-seg/data/spokencoco/speechtokens/rvq1/spokencoco_rvq1_tokens_dev.txt', 
            '/data/sls/scratch/clai24/word-seg/joint-seg/data/spokencoco/labels/spokencoco_dev_phone_labels.npz', 
            '/data/sls/scratch/clai24/word-seg/joint-seg/data/spokencoco/labels/spokencoco_dev_syllable_labels.npz', 
            '/data/sls/scratch/clai24/word-seg/joint-seg/data/spokencoco/labels/spokencoco_dev_word_labels.npz', 
        )
        self.dataloader = DataLoader(self.dataset, batch_size=32, collate_fn=collate_fn)

    def test_length(self):
        # Ensure the dataset reports the correct length
        self.assertEqual(len(self.dataset), len(self.dataset.tokens))

    def test_getitem(self):
        # Check the types of items returned by __getitem__
        data = self.dataset[0]
        print(f'checking {data[0]}')
        self.assertIsInstance(data[1], torch.Tensor)
        self.assertIsInstance(data[2], torch.Tensor)
        self.assertIsInstance(data[3], torch.Tensor)
        self.assertIsInstance(data[4], torch.Tensor)

        # Optionally, check the tensor dimensions or other properties
        self.assertEqual(data[1].shape[0], min(self.dataset.max_seq_len, len(self.dataset.tokens[0][1])))

    def test_collate(self):
        for uttids, tokens, phones, syllables, words, src_mask, src_key_padding_mask in self.dataloader:
            # Check if the padded sequences have the same length
            self.assertEqual(tokens.shape[1], phones.shape[1])
            self.assertEqual(tokens.shape[1], syllables.shape[1])
            self.assertEqual(tokens.shape[1], words.shape[1])

            # Check if the mask has the correct dimensions
            self.assertEqual(tokens.shape, src_key_padding_mask.shape)

            break  # Only test with the first batch

if __name__ == '__main__':
    unittest.main()

