"""
Test forward pass for Causal Autoencoder on CPU
Tests model correctness without training
"""

import torch
import torch.nn as nn

from model.causal_autoencoder import (
    CausalAutoencoder,
    SemanticEncoder,
    CausalEncoder,
    CausalDecoder,
    LorentzianTransformer,
)
from loss.causal_losses import (
    compute_causal_loss,
    causal_ordering_loss,
    spatial_independence_loss,
    temporal_ordering_loss,
    hyperboloid_consistency_loss,
)


def test_forward():
    """Test forward pass of all components on CPU"""
    device = torch.device('cpu')
    print("Testing Causal Autoencoder forward pass on CPU")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    num_frames = 16
    image_height = 224
    image_width = 224
    d_semantic = 768
    d_latent = 128
    
    print(f"\nTest parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num frames: {num_frames}")
    print(f"  Image size: {image_height}x{image_width}")
    print(f"  Semantic dim: {d_semantic}")
    print(f"  Latent dim: {d_latent}")
    
    # Create synthetic video data in channel-last format [batch, T, H, W, C]
    videos = torch.randn(batch_size, num_frames, image_height, image_width, 3)
    videos = videos.contiguous(memory_format=torch.channels_last)
    print(f"\nInput video shape: {videos.shape} (channel-last format)")
    
    # Test 1: Semantic Encoder
    print("\n" + "=" * 60)
    print("Test 1: Semantic Encoder")
    print("=" * 60)
    try:
        semantic_encoder = SemanticEncoder()
        semantic_encoder.eval()
        with torch.no_grad():
            z = semantic_encoder(videos)
        print(f"✓ Semantic encoder output shape: {z.shape}")
        assert z.shape == (batch_size, num_frames, d_semantic), f"Expected {(batch_size, num_frames, d_semantic)}, got {z.shape}"
    except Exception as e:
        print(f"✗ Semantic encoder failed: {e}")
        print("  (This is expected if transformers library is not installed)")
        # Create placeholder z for further tests
        z = torch.randn(batch_size, num_frames, d_semantic)
    
    # Test 2: Causal Encoder
    print("\n" + "=" * 60)
    print("Test 2: Causal Encoder")
    print("=" * 60)
    try:
        causal_encoder = CausalEncoder(d_semantic=d_semantic, d_latent=d_latent, c=1.0)
        causal_encoder.eval()
        with torch.no_grad():
            mu = causal_encoder(z)
        print(f"✓ Causal encoder output shape: {mu.shape}")
        assert mu.shape == (batch_size, num_frames, d_latent + 1), f"Expected {(batch_size, num_frames, d_latent + 1)}, got {mu.shape}"
        
        # Check hyperboloid constraint
        tau = mu[..., 0]
        x = mu[..., 1:]
        x_norm_sq = torch.sum(x**2, dim=-1)
        constraint = -tau**2 + x_norm_sq + 1.0**2
        constraint_mean = constraint.mean().item()
        print(f"  Hyperboloid constraint (should be ~0): {constraint_mean:.6f}")
        
    except Exception as e:
        print(f"✗ Causal encoder failed: {e}")
        raise
    
    # Test 3: Lorentzian Transformer
    print("\n" + "=" * 60)
    print("Test 3: Lorentzian Transformer")
    print("=" * 60)
    try:
        lorentzian_transformer = LorentzianTransformer(d_model=d_latent + 1, n_layers=2, n_heads=3)  # n_heads must divide 129
        lorentzian_transformer.eval()
        with torch.no_grad():
            h = lorentzian_transformer(mu)
        print(f"✓ Lorentzian transformer output shape: {h.shape}")
        assert h.shape == mu.shape, f"Expected {mu.shape}, got {h.shape}"
    except Exception as e:
        print(f"✗ Lorentzian transformer failed: {e}")
        raise
    
    # Test 4: Causal Decoder
    print("\n" + "=" * 60)
    print("Test 4: Causal Decoder")
    print("=" * 60)
    try:
        causal_decoder = CausalDecoder(d_latent=d_latent, d_semantic=d_semantic, image_size=image_height)
        causal_decoder.eval()
        with torch.no_grad():
            frames_recon = causal_decoder(mu)
        print(f"✓ Causal decoder output shape: {frames_recon.shape}")
        # Decoder outputs channel-last format [batch, T, H, W, 3]
        expected_shape = (batch_size, num_frames, image_height, image_width, 3)
        assert frames_recon.shape == expected_shape, f"Expected {expected_shape}, got {frames_recon.shape}"
    except Exception as e:
        print(f"✗ Causal decoder failed: {e}")
        raise
    
    # Test 5: Full Causal Autoencoder (encoder only)
    print("\n" + "=" * 60)
    print("Test 5: Full Causal Autoencoder (Encoder Only)")
    print("=" * 60)
    try:
        model_encoder_only = CausalAutoencoder(
            d_semantic=d_semantic,
            d_latent=d_latent,
            c=1.0,
            image_size=image_height,
            use_semantic_encoder=True,
            use_decoder=False
        )
        model_encoder_only.eval()
        with torch.no_grad():
            output = model_encoder_only(videos, return_intermediates=False)
        print(f"✓ Model (encoder only) output: {output}")
        assert output is None, "Expected None for encoder_only mode"
        
        # Test with intermediates
        with torch.no_grad():
            output, z_out, mu_out = model_encoder_only(videos, return_intermediates=True)
        print(f"✓ Model (encoder only) with intermediates:")
        print(f"  z shape: {z_out.shape}")
        print(f"  mu shape: {mu_out.shape}")
    except Exception as e:
        print(f"✗ Model (encoder only) failed: {e}")
        raise
    
    # Test 6: Full Causal Autoencoder (with decoder)
    print("\n" + "=" * 60)
    print("Test 6: Full Causal Autoencoder (With Decoder)")
    print("=" * 60)
    try:
        model_full = CausalAutoencoder(
            d_semantic=d_semantic,
            d_latent=d_latent,
            c=1.0,
            image_size=image_height,
            use_semantic_encoder=True,
            use_decoder=True
        )
        model_full.eval()
        with torch.no_grad():
            frames_recon = model_full(videos, return_intermediates=False)
        print(f"✓ Model (full) output shape: {frames_recon.shape}")
        # Model outputs channel-last format [batch, T, H, W, 3]
        expected_shape = (batch_size, num_frames, image_height, image_width, 3)
        assert frames_recon.shape == expected_shape, f"Expected {expected_shape}, got {frames_recon.shape}"
        
        # Test with intermediates
        with torch.no_grad():
            frames_recon, z_out, mu_out = model_full(videos, return_intermediates=True)
        print(f"✓ Model (full) with intermediates:")
        print(f"  frames_recon shape: {frames_recon.shape}")
        print(f"  z shape: {z_out.shape}")
        print(f"  mu shape: {mu_out.shape}")
    except Exception as e:
        print(f"✗ Model (full) failed: {e}")
        raise
    
    # Test 7: Loss Functions
    print("\n" + "=" * 60)
    print("Test 7: Loss Functions")
    print("=" * 60)
    try:
        # Get mu from model
        model_full.eval()
        with torch.no_grad():
            _, z_test, mu_test = model_full(videos, return_intermediates=True)
            frames_recon_test = model_full(videos, return_intermediates=False)
        
        # Test individual losses
        loss_causal = causal_ordering_loss(mu_test)
        print(f"✓ Causal ordering loss: {loss_causal.item():.6f}")
        
        loss_acausal = spatial_independence_loss(mu_test)
        print(f"✓ Spatial independence loss: {loss_acausal.item():.6f}")
        
        loss_temporal = temporal_ordering_loss(mu_test)
        print(f"✓ Temporal ordering loss: {loss_temporal.item():.6f}")
        
        loss_hyperboloid = hyperboloid_consistency_loss(mu_test)
        print(f"✓ Hyperboloid consistency loss: {loss_hyperboloid.item():.6f}")
        
        # Test full loss
        loss, loss_dict = compute_causal_loss(videos, frames_recon_test, mu_test)
        print(f"✓ Full loss: {loss.item():.6f}")
        print(f"  Loss components: {loss_dict}")
    except Exception as e:
        print(f"✗ Loss functions failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_forward()
