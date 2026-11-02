import copy
import unittest
from pathlib import Path

import torch
from torch import nn
from engine.core.yaml_utils import load_config
from engine.deim.dfine_decoder import MagnitudeDecoupledCosineClassifier, DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher
from engine.optim.ema import ModelEMA


class MDCCTests(unittest.TestCase):
    def test_formula_bounds_zero_and_gradients(self):
        head = MagnitudeDecoupledCosineClassifier(nn.Linear(32, 7))
        x = torch.randn(2, 5, 32, requires_grad=True)
        actual = head(x)
        tau = 8 + 8 * torch.sigmoid(head.a * x.norm(dim=-1, keepdim=True).log1p() + head.d)
        expected = tau * torch.nn.functional.linear(torch.nn.functional.normalize(x, dim=-1),
                     torch.nn.functional.normalize(head.weight, dim=-1)) + head.bias
        torch.testing.assert_close(actual, expected)
        for scale in [0., 1e-8, 1., 1e8]:
            value = head(x.detach() * scale)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(((head.temperature(x.detach() * scale) >= 8) &
                             (head.temperature(x.detach() * scale) <= 16)).all())
        (actual.square().mean() + head.orthogonality_loss(.1)).backward()
        for parameter in [x, head.weight, head.bias, head.a, head.d]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        head.zero_grad()
        head(torch.zeros_like(x)).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_orthogonality_ordered_pairs_and_checkpoint(self):
        head = MagnitudeDecoupledCosineClassifier(nn.Linear(8, 3))
        with torch.no_grad():
            head.weight.fill_(1)
        self.assertAlmostEqual(head.orthogonality_loss(.1).item(), 6 * .9 ** 2, places=5)
        other = MagnitudeDecoupledCosineClassifier(nn.Linear(8, 3))
        other.load_state_dict(head.state_dict())
        x = torch.randn(2, 8)
        torch.testing.assert_close(head(x), other(x))

    def test_decoder_detection_loss_and_deploy(self):
        args = dict(num_classes=7, hidden_dim=32, feat_channels=[32]*3,
                    num_queries=5, num_layers=3, nhead=4, num_points=2,
                    dim_feedforward=64, reg_max=8, num_denoising=4, eval_spatial_size=None)
        torch.manual_seed(5)
        baseline = DFINETransformer(**copy.deepcopy(args))
        torch.manual_seed(5)
        model = DFINETransformer(**copy.deepcopy(args), use_mdcc=True)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        feats = [torch.randn(2, 32, h, w) for h, w in [(8,8), (4,4), (2,2)]]
        targets = [dict(labels=torch.tensor([i]), boxes=torch.tensor([[.5,.5,.3,.3]])) for i in [1,3]]
        output = model(feats, targets)
        heads = [model.enc_score_head, *model.dec_score_head]
        torch.testing.assert_close(output['mdcc_ortho'], sum(h.orthogonality_loss(.1) for h in heads))
        criterion = DEIMCriterion(matcher=HungarianMatcher(weight_dict={'cost_class':2,'cost_bbox':5,'cost_giou':2}),
            weight_dict={'loss_mal':1, 'loss_bbox':5, 'loss_giou':2, 'loss_fgl':.15,
                         'loss_ddf':1.5, 'loss_mdcc_ortho':.1},
            losses=['mal','boxes','local'], num_classes=7, reg_max=8)
        losses = criterion(output, targets)
        torch.testing.assert_close(losses['loss_mdcc_ortho'], output['mdcc_ortho'] * .1)
        self.assertEqual(sum(k.startswith('loss_mdcc_ortho') for k in losses), 1)
        self.assertIn('loss_mal_enc_0', losses)
        self.assertIn('loss_mal_aux_0', losses)
        self.assertIn('loss_mal_dn_0', losses)
        sum(losses.values()).backward()
        for head in heads:
            for p in head.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
        restored = DFINETransformer(**copy.deepcopy(args), use_mdcc=True)
        restored.load_state_dict(model.state_dict())
        ema = ModelEMA(model, decay=.5, warmups=0)
        with torch.no_grad():
            model.enc_score_head.a.add_(1)
        ema.update(model)
        torch.testing.assert_close(ema.module.enc_score_head.a, model.enc_score_head.a - .5)
        model.eval()
        with torch.no_grad():
            expected = model(feats)['pred_logits']
            self.assertNotIn('mdcc_ortho', model(feats))
            model.convert_to_deploy()
            torch.testing.assert_close(model(feats)['pred_logits'], expected)

    def test_configs_preserve_protocol(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / 'configs/deim_dfine'
        for server in ['new_server','old_server']:
            basefile = 'deim_hgnetv2_m_sar_aircraft_102e_b28' + ('_old_server' if server == 'old_server' else '') + '.yml'
            base = load_config(str(folder / basefile), {})
            for rem in [False, True]:
                method = 'rem_mdcc' if rem else 'mdcc'
                cfg = load_config(str(folder / f'deim_hgnetv2_m_sar_aircraft_{method}_102e_b28_{server}.yml'), {})
                for key in ['use_mdcc','mdcc_tau_min','mdcc_tau_max','mdcc_ortho_gamma']:
                    cfg['DFINETransformer'].pop(key)
                self.assertEqual(cfg['DEIMCriterion']['weight_dict'].pop('loss_mdcc_ortho'), .1)
                if rem:
                    self.assertTrue(cfg['HybridEncoder'].pop('use_rem'))
                self.assertTrue((root / cfg['experiment_protocol_document']).is_file())
                for item in [cfg, base]:
                    for key in ['__include__','output_dir','experiment_protocol_document']:
                        item.pop(key, None)
                self.assertEqual(cfg, base)


if __name__ == '__main__':
    unittest.main()
