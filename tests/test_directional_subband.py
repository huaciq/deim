"""DSG transform, ablation isolation, initialization and gradient tests."""
import unittest
from unittest.mock import patch

import torch

from engine.deim.directional_subband import DirectionalSubbandGate, haar_dwt2, haar_iwt2


class DirectionalSubbandTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_haar_reconstruction_odd_extents_and_gradient(self):
        for shape in [(2, 3, 8, 10), (1, 2, 7, 9), (1, 2, 1, 3)]:
            original = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            bands, extent = haar_dwt2(original)
            reconstructed = haar_iwt2(bands, extent)
            torch.testing.assert_close(reconstructed, original, rtol=1e-14, atol=1e-14)
            reconstructed.sum().backward()
            torch.testing.assert_close(original.grad, torch.ones_like(original), rtol=0, atol=0)

    def test_haar_signed_orientation_bands(self):
        horizontal = torch.tensor([[[[1., 1.], [-1., -1.]]]])
        vertical = torch.tensor([[[[1., -1.], [1., -1.]]]])
        diagonal = torch.tensor([[[[1., -1.], [-1., 1.]]]])
        for image, expected_index in [(horizontal, 1), (vertical, 2), (diagonal, 3)]:
            bands, _ = haar_dwt2(image)
            expected = torch.zeros_like(bands)
            expected[:, expected_index] = 2
            torch.testing.assert_close(bands, expected, rtol=0, atol=0)
            negative, _ = haar_dwt2(-image)
            torch.testing.assert_close(negative, -expected, rtol=0, atol=0)

    def test_zero_alpha_identity_and_single_zero_initialization(self):
        for mode in DirectionalSubbandGate.MODES:
            module = DirectionalSubbandGate(16, 8, mode)
            features = torch.randn(2, 16, 7, 9, requires_grad=True)
            output = module(features)
            torch.testing.assert_close(output, features, rtol=0, atol=0)
            self.assertGreater(float(module.expand.weight.abs().sum()), 0)
            output.square().mean().backward()
            self.assertIsNotNone(module.alpha.grad)
            self.assertTrue(torch.isfinite(module.alpha.grad))
            self.assertGreater(float(module.alpha.grad.abs()), 0)
            self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))

    def test_nonzero_residual_all_modes_learn_and_preserve_extents(self):
        counts = []
        for mode in DirectionalSubbandGate.MODES:
            torch.manual_seed(4)
            module = DirectionalSubbandGate(16, 8, mode, alpha_init=.2)
            counts.append(sum(p.numel() for p in module.parameters()))
            features = torch.randn(2, 16, 7, 9, requires_grad=True)
            output = module(features)
            self.assertEqual(output.shape, features.shape)
            self.assertEqual(output.dtype, features.dtype)
            self.assertFalse(torch.equal(output, features))
            output.square().mean().backward()
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            self.assertTrue(torch.isfinite(features.grad).all())
        self.assertEqual(len(set(counts)), 1)

    def test_shared_gate_and_ll_only_ignore_high_frequency_input(self):
        module = DirectionalSubbandGate(8, 8, 'shared', alpha_init=.3)
        low = torch.randn(1, 8, 4, 5)
        details = torch.randn(1, 3, 8, 4, 5)
        gates = module._gates(low, details)
        torch.testing.assert_close(gates[:, 0], gates[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(gates[:, 0], gates[:, 2], rtol=0, atol=0)
        # At the transform boundary, equal LL plus arbitrary changed HF must
        # produce equal LL-only updates. The directional mode must respond.
        low_only = DirectionalSubbandGate(8, 8, 'll_only', alpha_init=1.)
        features = torch.randn(1, 8, 8, 10)
        coefficients = torch.randn(1, 4, 8, 4, 5)
        changed = coefficients.clone()
        changed[:, 1:] = -3 * coefficients[:, 1:]
        with patch('engine.deim.directional_subband.haar_dwt2', return_value=(coefficients, (8, 10))):
            first = low_only(features)
        with patch('engine.deim.directional_subband.haar_dwt2', return_value=(changed, (8, 10))):
            second = low_only(features)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        low_only.mode = 'directional'
        with patch('engine.deim.directional_subband.haar_dwt2', return_value=(coefficients, (8, 10))):
            first = low_only(features)
        with patch('engine.deim.directional_subband.haar_dwt2', return_value=(changed, (8, 10))):
            second = low_only(features)
        self.assertGreater(float((first - second).abs().sum()), 0)

    def test_amp_dtype_large_energy_and_nonfinite_visibility(self):
        module = DirectionalSubbandGate(8, 8, 'directional', alpha_init=.1)
        features = (torch.randn(2, 8, 7, 9) * 1000).to(torch.bfloat16).requires_grad_()
        gate_dtypes = []
        hook = module.gate[0].register_forward_pre_hook(lambda layer, args:gate_dtypes.append(args[0].dtype))
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = module(features)
        hook.remove()
        self.assertEqual(gate_dtypes, [torch.float32])
        torch.testing.assert_close(output, module(features), rtol=0, atol=0)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
        invalid = torch.randn(2, 8, 7, 9)
        invalid[0, 0, 0, 0] = float('nan')
        self.assertFalse(torch.isfinite(module(invalid)).all())


if __name__ == '__main__':
    unittest.main()
