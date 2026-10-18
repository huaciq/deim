"""
Quick test script for BCM module functionality
Run this locally to ensure BCM can be instantiated and runs without errors
"""

import torch
import sys
sys.path.insert(0, '.')

from engine.deim.sar_boundary import BoundaryConsistencyMap, generate_gt_boundary_map


def test_bcm_module():
    """Test BCM module can run forward pass"""
    print("Testing BCM module...")

    # Create dummy encoder feature
    B, C, H, W = 2, 256, 80, 80  # batch=2, stride-8 feature
    encoder_feat = torch.randn(B, C, H, W)

    # Create BCM module
    bcm = BoundaryConsistencyMap(in_channels=C, hidden_dim=64)

    # Forward pass
    boundary_map = bcm(encoder_feat)

    assert boundary_map.shape == (B, 1, H, W), f"Expected shape (2, 1, 80, 80), got {boundary_map.shape}"
    assert boundary_map.min() >= 0 and boundary_map.max() <= 1, "Boundary map should be in [0, 1]"

    print(f"✓ BCM forward pass OK: {boundary_map.shape}")
    print(f"  Boundary map range: [{boundary_map.min():.3f}, {boundary_map.max():.3f}]")
    return True


def test_gt_boundary_generation():
    """Test GT boundary map generation"""
    print("\nTesting GT boundary map generation...")

    # Create dummy targets
    targets = [
        {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.1, 0.15]]),  # cxcywh normalized
            'labels': torch.tensor([0, 1])
        },
        {
            'boxes': torch.tensor([[0.8, 0.4, 0.15, 0.2]]),
            'labels': torch.tensor([2])
        }
    ]

    H, W = 80, 80
    gt_boundary = generate_gt_boundary_map(targets, H, W, feat_stride=8, line_width=2)

    assert gt_boundary.shape == (2, 1, H, W), f"Expected shape (2, 1, 80, 80), got {gt_boundary.shape}"
    assert gt_boundary.dtype == torch.float32, f"Expected float32, got {gt_boundary.dtype}"
    assert gt_boundary.min() >= 0 and gt_boundary.max() <= 1, "GT boundary should be binary [0, 1]"

    num_pos_0 = (gt_boundary[0] > 0.5).sum().item()
    num_pos_1 = (gt_boundary[1] > 0.5).sum().item()

    print(f"✓ GT boundary generation OK: {gt_boundary.shape}")
    print(f"  Batch 0: {num_pos_0} boundary pixels (2 boxes)")
    print(f"  Batch 1: {num_pos_1} boundary pixels (1 box)")
    return True


def test_bcm_loss_computation():
    """Test BCM loss computation"""
    print("\nTesting BCM loss computation...")

    import torch.nn.functional as F

    B, H, W = 2, 80, 80
    pred_boundary = torch.randn(B, 1, H, W)  # Logits (before sigmoid)

    # Generate GT
    targets = [
        {'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3]]), 'labels': torch.tensor([0])},
        {'boxes': torch.tensor([[0.3, 0.7, 0.1, 0.15]]), 'labels': torch.tensor([1])}
    ]
    gt_boundary = generate_gt_boundary_map(targets, H, W, feat_stride=8)

    # Compute loss
    num_pos = (gt_boundary > 0.5).sum().clamp(min=1.0)
    num_neg = (gt_boundary <= 0.5).sum().clamp(min=1.0)
    pos_weight = (num_neg / num_pos).clamp(max=10.0)

    loss = F.binary_cross_entropy_with_logits(pred_boundary, gt_boundary, pos_weight=pos_weight)

    assert loss.numel() == 1, "Loss should be scalar"
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss should be finite"

    print(f"✓ BCM loss computation OK")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Pos/Neg ratio: {num_pos.item()}/{num_neg.item()} = {(num_pos/num_neg).item():.4f}")
    print(f"  Pos weight: {pos_weight.item():.2f}")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("BCM Module Functionality Test")
    print("=" * 60)

    try:
        test_bcm_module()
        test_gt_boundary_generation()
        test_bcm_loss_computation()

        print("\n" + "=" * 60)
        print("✓ All tests passed! BCM module is ready to use.")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
