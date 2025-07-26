"""Unit tests for LiteTrackNet ball detection model."""

import pytest
import torch
import torch.nn as nn

from src.ball.models.lite_tracknet import (
    SE,
    DSConv,
    make_dsconv_block,
    PixelShuffleBlock,
    LiteTrackNet
)


@pytest.mark.unit
class TestSE:
    """Test Squeeze-and-Excitation block."""
    
    def test_se_init(self):
        """Test SE block initialization."""
        se = SE(channels=64, se_ratio=0.25)
        
        # Check components
        assert hasattr(se, 'avg_pool')
        assert hasattr(se, 'fc')
        
        # Check hidden channel calculation
        expected_hidden = max(1, int(64 * 0.25))  # 16
        assert se.fc[0].out_channels == expected_hidden
        assert se.fc[2].out_channels == 64  # Back to original channels
    
    def test_se_forward_shape(self):
        """Test SE block forward pass shape."""
        se = SE(channels=32, se_ratio=0.5)
        x = torch.randn(2, 32, 64, 64)
        
        with torch.no_grad():
            out = se(x)
        
        # Output should have same shape as input
        assert out.shape == x.shape
    
    def test_se_forward_attention(self):
        """Test SE block applies attention properly."""
        se = SE(channels=16, se_ratio=0.25)
        x = torch.randn(1, 16, 32, 32)
        
        with torch.no_grad():
            out = se(x)
        
        # Output should be different from input (attention applied)
        assert not torch.allclose(out, x)
        
        # Output should be finite
        assert torch.isfinite(out).all()
    
    def test_se_zero_ratio(self):
        """Test SE with zero ratio uses minimum 1 hidden channel."""
        se = SE(channels=64, se_ratio=0.01)  # Very small ratio
        
        # Should still have at least 1 hidden channel
        assert se.fc[0].out_channels == 1


@pytest.mark.unit
class TestDSConv:
    """Test Depthwise Separable Convolution."""
    
    def test_dsconv_init(self):
        """Test DSConv initialization."""
        dsconv = DSConv(in_c=32, out_c=64, kernel_size=3, stride=1, se_ratio=0.25)
        
        # Check components
        assert hasattr(dsconv, 'dw')  # Depthwise conv
        assert hasattr(dsconv, 'pw')  # Pointwise conv
        assert hasattr(dsconv, 'bn1')
        assert hasattr(dsconv, 'bn2')
        assert hasattr(dsconv, 'act')
        assert hasattr(dsconv, 'se')
        
        # Check depthwise conv configuration
        assert dsconv.dw.groups == 32  # Groups should equal input channels
        assert dsconv.dw.in_channels == 32
        assert dsconv.dw.out_channels == 32
        
        # Check pointwise conv
        assert dsconv.pw.kernel_size == (1, 1)
        assert dsconv.pw.in_channels == 32
        assert dsconv.pw.out_channels == 64
    
    def test_dsconv_residual_connection(self):
        """Test DSConv residual connection logic."""
        # Should use residual when in_c == out_c and stride == 1
        dsconv_residual = DSConv(in_c=32, out_c=32, stride=1)
        assert dsconv_residual.use_residual is True
        
        # Should not use residual when channels differ
        dsconv_no_residual = DSConv(in_c=32, out_c=64, stride=1)
        assert dsconv_no_residual.use_residual is False
        
        # Should not use residual when stride != 1
        dsconv_stride = DSConv(in_c=32, out_c=32, stride=2)
        assert dsconv_stride.use_residual is False
    
    def test_dsconv_forward_shape(self):
        """Test DSConv forward pass shape."""
        dsconv = DSConv(in_c=16, out_c=32, stride=2)
        x = torch.randn(2, 16, 64, 64)
        
        with torch.no_grad():
            out = dsconv(x)
        
        # Should change channels and downsample by stride
        expected_shape = (2, 32, 32, 32)
        assert out.shape == expected_shape
    
    def test_dsconv_residual_forward(self):
        """Test DSConv with residual connection."""
        dsconv = DSConv(in_c=32, out_c=32, stride=1, se_ratio=0.0)  # No SE for simplicity
        x = torch.randn(1, 32, 64, 64)
        
        with torch.no_grad():
            out = dsconv(x)
        
        # Should have same shape as input
        assert out.shape == x.shape
        
        # With residual, output should be different from input
        assert not torch.allclose(out, x)
    
    def test_dsconv_no_se(self):
        """Test DSConv without SE block."""
        dsconv = DSConv(in_c=16, out_c=32, se_ratio=0.0)
        
        # SE should be Identity when ratio is 0
        assert isinstance(dsconv.se, nn.Identity)


@pytest.mark.unit
class TestMakeDSConvBlock:
    """Test DSConv block creation utility."""
    
    def test_make_dsconv_block_single(self):
        """Test creating single DSConv block."""
        block = make_dsconv_block(in_c=32, out_c=64, stride=2, repeat=1)
        
        assert isinstance(block, nn.Sequential)
        assert len(block) == 1
        assert isinstance(block[0], DSConv)
    
    def test_make_dsconv_block_multiple(self):
        """Test creating multiple DSConv blocks."""
        block = make_dsconv_block(in_c=32, out_c=64, stride=2, repeat=3)
        
        assert len(block) == 3
        
        # First block should have stride and channel change
        assert block[0].dw.stride == (2, 2)
        
        # Subsequent blocks should have stride 1
        for i in range(1, 3):
            assert block[i].dw.stride == (1, 1)
    
    def test_make_dsconv_block_forward(self):
        """Test DSConv block forward pass."""
        block = make_dsconv_block(in_c=16, out_c=32, stride=2, repeat=2)
        x = torch.randn(1, 16, 64, 64)
        
        with torch.no_grad():
            out = block(x)
        
        # Should change channels and downsample
        expected_shape = (1, 32, 32, 32)
        assert out.shape == expected_shape


@pytest.mark.unit
class TestPixelShuffleBlock:
    """Test Pixel Shuffle upsampling block."""
    
    def test_pixel_shuffle_block_init(self):
        """Test PixelShuffleBlock initialization."""
        ps_block = PixelShuffleBlock(in_c=64, out_c=32, upscale_factor=2)
        
        # Check components
        assert hasattr(ps_block, 'conv')
        assert hasattr(ps_block, 'ps')
        assert hasattr(ps_block, 'bn')
        assert hasattr(ps_block, 'act')
        
        # Check conv output channels: out_c * (upscale_factor^2)
        expected_conv_out = 32 * (2 ** 2)  # 128
        assert ps_block.conv.out_channels == expected_conv_out
    
    def test_pixel_shuffle_block_forward_shape(self):
        """Test PixelShuffleBlock forward pass shape."""
        ps_block = PixelShuffleBlock(in_c=128, out_c=64, upscale_factor=2)
        x = torch.randn(2, 128, 32, 32)
        
        with torch.no_grad():
            out = ps_block(x)
        
        # Should upscale spatial dimensions by upscale_factor
        expected_shape = (2, 64, 64, 64)
        assert out.shape == expected_shape
    
    def test_pixel_shuffle_block_different_factors(self):
        """Test PixelShuffleBlock with different upscale factors."""
        for factor in [2, 4]:
            ps_block = PixelShuffleBlock(in_c=64, out_c=16, upscale_factor=factor)
            x = torch.randn(1, 64, 8, 8)
            
            with torch.no_grad():
                out = ps_block(x)
            
            expected_shape = (1, 16, 8 * factor, 8 * factor)
            assert out.shape == expected_shape


@pytest.mark.unit
class TestLiteTrackNet:
    """Test main LiteTrackNet model."""
    
    def test_lite_tracknet_init(self):
        """Test LiteTrackNet initialization."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Check all components exist
        assert hasattr(model, 'enc1')
        assert hasattr(model, 'enc2')
        assert hasattr(model, 'enc3')
        assert hasattr(model, 'enc4')
        assert hasattr(model, 'bottleneck')
        assert hasattr(model, 'dec3_up')
        assert hasattr(model, 'dec3_conv')
        assert hasattr(model, 'dec2_up')
        assert hasattr(model, 'dec2_conv')
        assert hasattr(model, 'dec1_up')
        assert hasattr(model, 'dec1_conv')
        assert hasattr(model, 'head')
    
    def test_lite_tracknet_forward_shape(self):
        """Test LiteTrackNet forward pass output shape."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        x = torch.randn(2, 9, 360, 640)
        
        with torch.no_grad():
            out = model(x)
        
        # Output should maintain spatial dimensions and squeeze channel dim
        expected_shape = (2, 360, 640)  # Channel squeezed for single output
        assert out.shape == expected_shape
    
    def test_lite_tracknet_multi_channel_output(self):
        """Test LiteTrackNet with multiple output channels."""
        model = LiteTrackNet(in_channels=9, out_channels=3)
        x = torch.randn(1, 9, 128, 128)
        
        with torch.no_grad():
            out = model(x)
        
        # Should not squeeze when out_channels > 1
        expected_shape = (1, 3, 128, 128)
        assert out.shape == expected_shape
    
    def test_lite_tracknet_different_input_sizes(self):
        """Test LiteTrackNet with different input sizes."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        test_sizes = [(160, 320), (240, 480), (320, 640)]
        
        for h, w in test_sizes:
            x = torch.randn(1, 9, h, w)
            
            with torch.no_grad():
                out = model(x)
            
            # Output should match input spatial dimensions
            assert out.shape == (1, h, w)
    
    def test_lite_tracknet_different_input_channels(self):
        """Test LiteTrackNet with different input channel counts."""
        for in_ch in [3, 6, 12]:
            model = LiteTrackNet(in_channels=in_ch, out_channels=1)
            x = torch.randn(1, in_ch, 64, 64)
            
            with torch.no_grad():
                out = model(x)
            
            assert out.shape == (1, 64, 64)
    
    def test_lite_tracknet_gradient_flow(self):
        """Test LiteTrackNet gradient flow."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        x = torch.randn(1, 9, 128, 128, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
        
        # Check input gradients
        assert x.grad is not None
    
    def test_lite_tracknet_unet_structure(self):
        """Test LiteTrackNet U-Net structure with skip connections."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Test that encoder-decoder structure works
        x = torch.randn(1, 9, 128, 128)
        
        with torch.no_grad():
            # Forward pass should complete without errors
            out = model(x)
        
        assert out.shape == (1, 128, 128)
        assert torch.isfinite(out).all()
    
    @pytest.mark.slow
    def test_lite_tracknet_memory_efficiency(self):
        """Test LiteTrackNet memory efficiency."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be lightweight (<5M parameters)
        assert total_params < 5_000_000, f"Model too large: {total_params} parameters"
        
        # Test with reasonable batch size
        x = torch.randn(8, 9, 320, 640)
        
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (8, 320, 640)
    
    def test_lite_tracknet_batch_consistency(self):
        """Test LiteTrackNet consistency across batch sizes."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Single sample
        x_single = torch.randn(1, 9, 64, 64)
        with torch.no_grad():
            out_single = model(x_single)
        
        # Batch of same samples
        x_batch = x_single.repeat(4, 1, 1, 1)
        with torch.no_grad():
            out_batch = model(x_batch)
        
        # Each batch item should be same as single
        for i in range(4):
            assert torch.allclose(out_batch[i], out_single[0], atol=1e-6)


@pytest.mark.unit
class TestLiteTrackNetIntegration:
    """Integration tests for LiteTrackNet components."""
    
    def test_encoder_decoder_compatibility(self):
        """Test encoder-decoder feature map compatibility."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Test that skip connections work properly
        x = torch.randn(1, 9, 128, 128)
        
        with torch.no_grad():
            # Should complete forward pass without dimension mismatches
            out = model(x)
        
        assert out.shape == (1, 128, 128)
    
    def test_different_architectures(self):
        """Test LiteTrackNet with different configurations."""
        configs = [
            {'in_channels': 3, 'out_channels': 1},
            {'in_channels': 6, 'out_channels': 2},
            {'in_channels': 12, 'out_channels': 1},
        ]
        
        for config in configs:
            model = LiteTrackNet(**config)
            x = torch.randn(2, config['in_channels'], 64, 64)
            
            with torch.no_grad():
                out = model(x)
            
            if config['out_channels'] == 1:
                expected_shape = (2, 64, 64)  # Squeezed
            else:
                expected_shape = (2, config['out_channels'], 64, 64)
            
            assert out.shape == expected_shape
    
    def test_feature_extraction_levels(self):
        """Test that different encoder levels produce appropriate feature sizes."""
        # This test verifies the U-Net structure indirectly
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Test various input sizes to ensure proper downsampling/upsampling
        input_sizes = [(64, 64), (128, 128), (256, 256)]
        
        for h, w in input_sizes:
            x = torch.randn(1, 9, h, w)
            
            with torch.no_grad():
                out = model(x)
            
            # Output should match input size exactly
            assert out.shape == (1, h, w)
    
    def test_end_to_end_training_compatibility(self):
        """Test LiteTrackNet in training scenario."""
        model = LiteTrackNet(in_channels=9, out_channels=1)
        
        # Create training batch
        x = torch.randn(4, 9, 128, 128, requires_grad=True)
        target = torch.randn(4, 128, 128)
        
        # Forward pass
        pred = model(x)
        
        # Simple loss computation
        loss = nn.MSELoss()(pred, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                # Should have non-zero gradients
                assert param.grad.abs().sum() > 0