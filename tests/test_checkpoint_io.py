import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from engine.misc import dist_utils
from engine.solver._solver import BaseSolver


class CheckpointIOTests(unittest.TestCase):
    def test_readers_see_previous_checkpoint_until_save_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best_stg1.pth'
            torch.save({'epoch': 70}, path)
            original_save = torch.save

            def save_with_reader(obj, destination, **kwargs):
                self.assertEqual(torch.load(path)['epoch'], 70)
                original_save(obj, destination, **kwargs)
                self.assertEqual(torch.load(path)['epoch'], 70)

            with patch.object(dist_utils, 'is_main_process', return_value=True), \
                    patch.object(torch, 'save', side_effect=save_with_reader):
                dist_utils.save_on_master({'epoch': 71}, path)
            self.assertEqual(torch.load(path)['epoch'], 71)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best_stg1.pth'
            torch.save({'epoch': 70}, path)

            def failed_save(obj, destination, **kwargs):
                destination.write(b'incomplete checkpoint')
                raise OSError('interrupted write')

            with patch.object(dist_utils, 'is_main_process', return_value=True), \
                    patch.object(torch, 'save', side_effect=failed_save):
                with self.assertRaisesRegex(OSError, 'interrupted write'):
                    dist_utils.save_on_master({'epoch': 71}, path)
            self.assertEqual(torch.load(path)['epoch'], 70)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_non_master_does_not_save(self):
        with patch.object(dist_utils, 'is_main_process', return_value=False), \
                patch.object(torch, 'save') as save:
            dist_utils.save_on_master({}, 'unused.pth')
            save.assert_not_called()

    def test_resume_waits_before_reading(self):
        solver = BaseSolver.__new__(BaseSolver)
        solver.load_state_dict = Mock()
        calls = []
        for distributed in (True, False):
            calls.clear()
            with patch.object(dist_utils, 'is_dist_available_and_initialized',
                              return_value=distributed), \
                    patch.object(torch.distributed, 'barrier',
                                 side_effect=lambda: calls.append('barrier')), \
                    patch.object(torch, 'load',
                                 side_effect=lambda *a, **k: calls.append('load') or {}):
                solver.load_resume_state('best_stg1.pth')
            self.assertEqual(calls, ['barrier', 'load'] if distributed else ['load'])


if __name__ == '__main__':
    unittest.main()
