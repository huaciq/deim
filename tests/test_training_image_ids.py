import contextlib
import io
import unittest
from unittest.mock import Mock

import torch

from engine.solver.det_engine import (
    _retry_forward_fp32,
    _snapshot_forward_state,
    _training_image_ids,
    train_one_epoch,
)


class ReplayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer('forward_count', torch.tensor(0))

    def forward(self, samples, targets=None):
        self.forward_count.add_(1)
        return {'pred_boxes': samples * self.weight + torch.rand_like(samples)}


class TrainingImageIDTests(unittest.TestCase):
    def test_single_and_mosaic_ids_remain_grouped(self):
        targets = [
            {'image_id': torch.tensor(7)},
            {'image_id': torch.tensor([8])},
            {'image_id': torch.tensor([10, 20, 30, 40])},
        ]
        self.assertEqual(_training_image_ids(targets), [[7], [8], [10, 20, 30, 40]])

    def test_invalid_prediction_reports_original_error_with_mosaic_ids(self):
        model = ReplayModel()
        model.forward = Mock(return_value={'pred_boxes': torch.full((1, 1, 4), float('nan'))})
        criterion = Mock()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        targets = [{'image_id': torch.tensor([10, 20, 30, 40])}]
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FloatingPointError) as error:
                train_one_epoch(False, None, model, criterion,
                                [(torch.ones(1, 1, 4), targets)], optimizer,
                                torch.device('cpu'), epoch=6)
        self.assertIn('Non-finite pred_boxes before matching', str(error.exception))
        self.assertIn('image_ids=[[10, 20, 30, 40]]', str(error.exception))
        criterion.assert_not_called()
        self.assertEqual(model.weight.item(), 2.0)

    def test_fp32_replay_logs_mosaic_ids_and_restores_forward_state(self):
        model = ReplayModel()
        device = torch.device('cpu')
        samples = torch.ones(1, 1, 4)
        targets = [{'image_id': torch.tensor([10, 20, 30, 40])}]
        state = _snapshot_forward_state(model, device)
        original = model(samples, targets)['pred_boxes'].detach().clone()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            replay = _retry_forward_fp32(model, samples, targets, state,
                                         device, 6, 200, 'nonfinite_output')
        self.assertIn('image_ids=[[10, 20, 30, 40]]', output.getvalue())
        self.assertIn('AMP_FP32_RETRY', output.getvalue())
        self.assertEqual(model.forward_count.item(), 1)
        torch.testing.assert_close(replay['pred_boxes'], original)
        replay['pred_boxes'].sum().backward()
        self.assertEqual(model.weight.grad.item(), 4.0)


if __name__ == '__main__':
    unittest.main()
