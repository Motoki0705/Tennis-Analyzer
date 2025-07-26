"""Unit tests for Conv3D TSM FPN court detection model."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from src.court.models.conv3d_tsm_fpn import (
    Conv3DStem,
    temporal_shift,
    ResidualBlock2D,
    BiFPN,
    BallDetNet,
    SigmoidFocalLoss
)


@pytest.mark.unit
class TestConv3DStem:
    """Test Conv3D Stem component."""
    
    def test_conv3d_stem_init(self):
        """Test Conv3DStem initialization."""
        stem = Conv3DStem(in_ch=3, out_ch=64)
        
        # Check component existence
        assert hasattr(stem, 't_conv')
        assert hasattr(stem, 's_conv')
        assert hasattr(stem, 'bn')
        assert hasattr(stem, 'act')
        
        # Check Conv3D configurations
        assert stem.t_conv.in_channels == 3
        assert stem.t_conv.out_channels == 3
        assert stem.t_conv.kernel_size == (3, 1, 1)
        assert stem.t_conv.groups == 3  # Depth-wise
        
        assert stem.s_conv.in_channels == 3
        assert stem.s_conv.out_channels == 64
        assert stem.s_conv.kernel_size == (1, 3, 3)
        assert stem.s_conv.stride == (1, 2, 2)  # Spatial downsampling
    
    def test_conv3d_stem_forward_shape(self):
        """Test Conv3DStem forward pass output shape."""
        stem = Conv3DStem(in_ch=3, out_ch=32)
        
        # Input: [B, N, C, H, W] -> permuted to [B, C, N, H, W] for Conv3D
        x = torch.randn(2, 8, 3, 320, 640)
        x_permuted = x.permute(0, 2, 1, 3, 4)  # [B, C, N, H, W]
        
        with torch.no_grad():
            out = stem(x_permuted)
        
        # Output should be [B, out_ch, N, H/2, W/2]
        expected_shape = (2, 32, 8, 160, 320)
        assert out.shape == expected_shape
    
    def test_conv3d_stem_forward_values(self):
        """Test Conv3DStem forward pass produces valid values."""
        stem = Conv3DStem(in_ch=3, out_ch=32)
        x = torch.randn(1, 4, 3, 64, 64).permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            out = stem(x)
        
        # Check output is not all zeros
        assert not torch.allclose(out, torch.zeros_like(out))
        
        # Check output is finite
        assert torch.isfinite(out).all()


@pytest.mark.unit 
class TestTemporalShift:
    """Test Temporal Shift Module (TSM)."""
    
    def test_temporal_shift_basic(self):
        """Test basic temporal shift functionality."""
        x = torch.randn(2, 8, 64, 32, 32)  # [B, N, C, H, W]
        
        out = temporal_shift(x, fold_div=8)
        
        # Shape should remain the same
        assert out.shape == x.shape
        
        # Output should not be identical to input (unless N=1)
        assert not torch.allclose(out, x)
    
    def test_temporal_shift_single_frame(self):
        """Test temporal shift with single frame (should be no-op)."""
        x = torch.randn(2, 1, 64, 32, 32)  # N=1
        
        out = temporal_shift(x, fold_div=8)
        
        # Should return unchanged
        assert torch.allclose(out, x)
    
    def test_temporal_shift_channel_division(self):
        """Test temporal shift channel division logic."""
        B, N, C, H, W = 1, 4, 24, 16, 16
        x = torch.arange(B * N * C * H * W, dtype=torch.float32).view(B, N, C, H, W)
        
        out = temporal_shift(x, fold_div=8)
        fold = C // 8  # 3 channels
        
        # Check left shift: frame 1 should have frame 0's first 'fold' channels
        assert torch.allclose(out[0, 1, :fold], x[0, 0, :fold])
        
        # Check right shift: frame 0 should have frame 1's second 'fold' channels  
        assert torch.allclose(out[0, 0, fold:2*fold], x[0, 1, fold:2*fold])
        
        # Check unchanged channels
        if C > 2 * fold:
            assert torch.allclose(out[0, :, 2*fold:], x[0, :, 2*fold:])
    
    def test_temporal_shift_gradient_flow(self):
        """Test temporal shift preserves gradients."""
        x = torch.randn(1, 4, 8, 16, 16, requires_grad=True)
        
        out = temporal_shift(x, fold_div=8)
        loss = out.sum()
        loss.backward()
        
        # Gradients should flow back to input
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


@pytest.mark.unit
class TestResidualBlock2D:
    """Test ResidualBlock2D with TSM support."""
    
    def test_residual_block_init(self):
        """Test ResidualBlock2D initialization."""
        block = ResidualBlock2D(64, 128, stride=2, use_tsm=True)
        
        # Check components
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'bn1')
        assert hasattr(block, 'bn2')
        assert hasattr(block, 'downsample')
        assert block.use_tsm is True
        
        # Check downsample is created when needed
        assert block.downsample is not None  # c_in != c_out
    
    def test_residual_block_no_downsample(self):
        """Test ResidualBlock2D without downsampling."""
        block = ResidualBlock2D(64, 64, stride=1, use_tsm=False)
        
        assert block.downsample is None
        assert block.use_tsm is False
    
    def test_residual_block_forward_shape(self):
        """Test ResidualBlock2D forward pass shape."""
        block = ResidualBlock2D(32, 64, stride=2, use_tsm=True)
        x = torch.randn(2, 8, 32, 64, 64)  # [B, N, C, H, W]
        
        with torch.no_grad():
            out = block(x)
        
        # Should downsample spatial dimensions by stride
        expected_shape = (2, 8, 64, 32, 32)
        assert out.shape == expected_shape
    
    def test_residual_block_with_tsm(self):
        """Test ResidualBlock2D applies TSM when enabled."""
        block = ResidualBlock2D(32, 32, use_tsm=True)
        
        with patch('src.court.models.conv3d_tsm_fpn.temporal_shift') as mock_tsm:
            mock_tsm.return_value = torch.randn(2, 4, 32, 64, 64)
            
            x = torch.randn(2, 4, 32, 64, 64)
            block(x)
            
            # TSM should be called
            mock_tsm.assert_called_once_with(x)
    
    def test_residual_block_without_tsm(self):
        """Test ResidualBlock2D skips TSM when disabled."""
        block = ResidualBlock2D(32, 32, use_tsm=False)
        
        with patch('src.court.models.conv3d_tsm_fpn.temporal_shift') as mock_tsm:
            x = torch.randn(2, 4, 32, 64, 64)
            block(x)
            
            # TSM should not be called
            mock_tsm.assert_not_called()


@pytest.mark.unit
class TestBiFPN:
    """Test Bi-directional FPN."""
    
    def test_bifpn_init(self):
        """Test BiFPN initialization."""
        in_channels = [64, 128, 256, 512]
        fpn = BiFPN(in_channels, fpn_ch=128)
        
        # Check components
        assert len(fpn.lat_convs) == 4
        assert len(fpn.fuse_convs) == 4
        assert len(fpn.bns) == 4
        
        # Check learnable weights
        assert fpn.w.shape == (2, 3)  # (top-down, bottom-up) x (num_levels-1)
    
    def test_bifpn_forward_shape(self):
        """Test BiFPN forward pass shapes."""
        in_channels = [64, 128, 256, 256]
        fpn = BiFPN(in_channels, fpn_ch=128)
        
        # Create multi-scale features (B*N format)
        feats = [
            torch.randn(16, 64, 64, 64),   # High resolution
            torch.randn(16, 128, 32, 32),  # Medium resolution
            torch.randn(16, 256, 16, 16),  # Low resolution
            torch.randn(16, 256, 8, 8)     # Lowest resolution
        ]
        
        with torch.no_grad():
            out_feats = fpn(feats)
        
        # Should return same number of feature maps
        assert len(out_feats) == 4
        
        # Check output channels are all fpn_ch
        for feat in out_feats:
            assert feat.shape[1] == 128  # fpn_ch
        
        # Check spatial dimensions preserved
        assert out_feats[0].shape[2:] == (64, 64)
        assert out_feats[1].shape[2:] == (32, 32)
        assert out_feats[2].shape[2:] == (16, 16)
        assert out_feats[3].shape[2:] == (8, 8)
    
    def test_bifpn_forward_values(self):
        """Test BiFPN produces valid outputs."""
        in_channels = [32, 64]
        fpn = BiFPN(in_channels, fpn_ch=64)
        
        feats = [
            torch.randn(8, 32, 32, 32),
            torch.randn(8, 64, 16, 16)
        ]
        
        with torch.no_grad():
            out_feats = fpn(feats)
        
        # Outputs should be finite and non-zero
        for feat in out_feats:
            assert torch.isfinite(feat).all()
            assert not torch.allclose(feat, torch.zeros_like(feat))


@pytest.mark.unit
class TestBallDetNet:
    """Test main BallDetNet model."""
    
    def test_court_det_net_init(self):
        """Test BallDetNet initialization."""
        model = BallDetNet(num_frames=5, num_classes=1)
        
        # Check all components exist
        assert hasattr(model, 'stem')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3') 
        assert hasattr(model, 'layer4')
        assert hasattr(model, 'fpn')
        assert hasattr(model, 'head')
        
        # Check parameters
        assert model.num_frames == 5
        assert model.num_classes == 1
    
    def test_court_det_net_forward_shape(self):
        """Test BallDetNet forward pass output shape."""
        model = BallDetNet(num_frames=8, num_classes=1)
        
        # Input: [B, N, C, H, W]
        x = torch.randn(2, 8, 3, 320, 640)
        
        with torch.no_grad():
            out = model(x)
        
        # Output should be [B, N, H, W] for num_classes=1
        expected_shape = (2, 8, 320, 640)
        assert out.shape == expected_shape
    
    def test_court_det_net_multi_class(self):
        """Test BallDetNet with multiple classes."""
        model = BallDetNet(num_frames=4, num_classes=3)
        x = torch.randn(1, 4, 3, 64, 64)
        
        with torch.no_grad():
            out = model(x)
        
        # Output should be [B, N, num_classes, H, W]
        expected_shape = (1, 4, 3, 64, 64)
        assert out.shape == expected_shape
    
    def test_court_det_net_single_frame(self):
        """Test BallDetNet with single frame."""
        model = BallDetNet(num_frames=1, num_classes=1)
        x = torch.randn(3, 1, 3, 128, 128)
        
        with torch.no_grad():
            out = model(x)
        
        expected_shape = (3, 1, 128, 128)
        assert out.shape == expected_shape
    
    def test_court_det_net_gradient_flow(self):
        """Test BallDetNet gradient flow."""
        model = BallDetNet(num_frames=3, num_classes=1)
        x = torch.randn(1, 3, 3, 64, 64, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
        
        # Check input gradients
        assert x.grad is not None
    
    def test_court_det_net_different_input_sizes(self):
        """Test BallDetNet with different input sizes."""
        model = BallDetNet(num_frames=4, num_classes=1)
        
        test_sizes = [(160, 320), (240, 480), (360, 640)]
        
        for h, w in test_sizes:
            x = torch.randn(1, 4, 3, h, w)
            
            with torch.no_grad():
                out = model(x)
            
            # Output should match input spatial dimensions
            assert out.shape == (1, 4, h, w)
    
    @pytest.mark.slow
    def test_court_det_net_memory_efficiency(self):
        """Test BallDetNet memory usage is reasonable."""
        model = BallDetNet(num_frames=8, num_classes=1)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable for a lightweight model (<10M parameters)
        assert total_params < 10_000_000, f"Model too large: {total_params} parameters"
        
        # Test memory usage with reasonable input
        x = torch.randn(4, 8, 3, 320, 640)  # 4 batch, 8 frames
        
        with torch.no_grad():
            out = model(x)
        
        # Should complete without memory errors
        assert out.shape == (4, 8, 320, 640)


@pytest.mark.unit
class TestSigmoidFocalLoss:
    """Test Sigmoid Focal Loss."""
    
    def test_sigmoid_focal_loss_init(self):
        """Test SigmoidFocalLoss initialization."""
        loss_fn = SigmoidFocalLoss(alpha=2.0, beta=4.0)
        
        assert loss_fn.alpha == 2.0
        assert loss_fn.beta == 4.0
    
    def test_sigmoid_focal_loss_forward_shape(self):
        """Test SigmoidFocalLoss forward pass."""
        loss_fn = SigmoidFocalLoss()
        
        pred = torch.randn(8, 64, 64)
        target = torch.randint(0, 2, (8, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        
        # Loss should be scalar
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_sigmoid_focal_loss_perfect_prediction(self):
        """Test focal loss with perfect predictions."""
        loss_fn = SigmoidFocalLoss()
        
        # Perfect positive predictions
        pred = torch.full((2, 32, 32), 10.0)  # High logits -> ~1 after sigmoid
        target = torch.ones(2, 32, 32)
        
        loss = loss_fn(pred, target)
        
        # Loss should be very small for perfect predictions
        assert loss.item() < 0.1
    
    def test_sigmoid_focal_loss_worst_prediction(self):
        """Test focal loss with worst predictions."""
        loss_fn = SigmoidFocalLoss()
        
        # Wrong predictions (predicting 0 when target is 1)
        pred = torch.full((2, 16, 16), -10.0)  # Low logits -> ~0 after sigmoid
        target = torch.ones(2, 16, 16)
        
        loss = loss_fn(pred, target)
        
        # Loss should be high for wrong predictions
        assert loss.item() > 1.0
    
    def test_sigmoid_focal_loss_balanced(self):
        """Test focal loss with balanced positive/negative samples."""
        loss_fn = SigmoidFocalLoss()
        
        # Half positive, half negative
        pred = torch.randn(4, 32, 32)
        target = torch.cat([
            torch.ones(2, 32, 32),
            torch.zeros(2, 32, 32)
        ])
        
        loss = loss_fn(pred, target)
        
        # Should produce finite loss
        assert torch.isfinite(loss)
        assert loss.item() >= 0
    
    def test_sigmoid_focal_loss_no_positive(self):
        """Test focal loss with no positive samples."""
        loss_fn = SigmoidFocalLoss()
        
        pred = torch.randn(2, 16, 16)
        target = torch.zeros(2, 16, 16)  # No positive samples
        
        loss = loss_fn(pred, target)
        
        # Should handle no positive samples gracefully
        assert torch.isfinite(loss)
        assert loss.item() >= 0


@pytest.mark.unit
class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_model_components_compatibility(self):
        """Test that all model components work together."""
        # Create individual components
        stem = Conv3DStem(3, 32)
        block = ResidualBlock2D(32, 64, stride=1, use_tsm=True)
        fpn = BiFPN([64], fpn_ch=128)
        
        # Test data flow
        x = torch.randn(2, 4, 3, 64, 64)
        
        with torch.no_grad():
            # Stem processing
            x_perm = x.permute(0, 2, 1, 3, 4)
            x_stem = stem(x_perm)
            x_back = x_stem.permute(0, 2, 1, 3, 4)
            
            # Block processing
            x_block = block(x_back)
            
            # FPN processing (reshape for batch processing)
            B, N, C, H, W = x_block.shape
            x_fpn_in = x_block.view(B*N, C, H, W)
            x_fpn_out = fpn([x_fpn_in])
            
            # Should complete without errors
            assert len(x_fpn_out) == 1
            assert x_fpn_out[0].shape[1] == 128  # fpn_ch
    
    def test_end_to_end_training_step(self):
        """Test a complete forward + backward pass."""
        model = BallDetNet(num_frames=4, num_classes=1)
        loss_fn = SigmoidFocalLoss()
        
        # Create batch
        x = torch.randn(2, 4, 3, 128, 128, requires_grad=True)
        target = torch.randint(0, 2, (2, 4, 128, 128)).float()
        
        # Forward pass
        pred = model(x)
        loss = loss_fn(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Check all components have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                    f"Zero gradient for {name}"