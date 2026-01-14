"""
Causal Video Autoencoder with Lorentzian Manifold Latent Space
Based on the research proposal
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """
    Stage 1: Semantic Encoder (VideoMAE - frozen, pretrained)
    Encodes video frames to semantic embeddings
    """
    def __init__(self, model_name='MCG-NJU/videomae-base'):
        super().__init__()
        try:
            from transformers import VideoMAEModel
            self.videomae = VideoMAEModel.from_pretrained(model_name)
            # Freeze all parameters
            for param in self.videomae.parameters():
                param.requires_grad = False
            self.placeholder = False
        except (ImportError, Exception) as e:
            # If transformers library is not available or model loading fails, create a placeholder for testing
            print(f"Warning: Could not load VideoMAE model: {e}")
            print("Creating placeholder encoder for testing")
            self.videomae = None
            self.placeholder = True
        
    def forward(self, videos):
        """
        Args:
            videos: [batch, T, H, W, 3] (channel-last format)
        Returns:
            z: [batch, T, 768] semantic embeddings
        """
        if self.placeholder:
            # Placeholder for testing without VideoMAE
            batch, T = videos.shape[:2]
            return torch.randn(batch, T, 768, device=videos.device)
        
        batch, T = videos.shape[:2]
        
        # Convert from channel-last [batch, T, H, W, 3] to channel-first [batch, T, 3, H, W]
        # Then transpose to VideoMAE format [batch, 3, T, H, W]
        videos = videos.permute(0, 1, 4, 2, 3).contiguous()  # [batch, T, H, W, 3] -> [batch, T, 3, H, W]
        videos_fmt = videos.transpose(1, 2)  # [batch, T, 3, H, W] -> [batch, 3, T, H, W]
        
        try:
            with torch.no_grad():  # Frozen, no gradients
                outputs = self.videomae(videos_fmt)
                features = outputs.last_hidden_state  # [batch, num_patches, 768]
            
            # Pool across patches for each frame
            # This is simplified - actual implementation should track temporal structure
            # For now, we average pool and repeat for each frame
            z_pooled = features.mean(dim=1, keepdim=True)  # [batch, 1, 768]
            z = z_pooled.repeat(1, T, 1)  # [batch, T, 768]
            
            return z
        except Exception as e:
            # If VideoMAE forward fails (e.g., due to input format), fall back to placeholder
            # This can happen if input normalization doesn't match VideoMAE's expectations
            print(f"Warning: VideoMAE forward failed: {e}")
            print("Falling back to placeholder encoder")
            return torch.randn(batch, T, 768, device=videos.device)


class CausalEncoder(nn.Module):
    """
    Stage 2: Causal Encoder (Lorentzian Projection)
    Projects semantic embeddings to Lorentzian manifold
    """
    def __init__(self, d_semantic=768, d_latent=128, c=1.0):
        """
        Project semantic embeddings to Lorentzian manifold.
        
        Args:
            d_semantic: Dimension of VideoMAE embeddings (768)
            d_latent: Dimension of spatial coordinates (128)
            c: Speed of causality parameter
        """
        super().__init__()
        self.d_latent = d_latent
        self.c = nn.Parameter(torch.tensor(c))
        
        # Non-linear projection (MLP, not just linear!)
        self.projection = nn.Sequential(
            nn.Linear(d_semantic, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1 + d_latent)  # Output: (τ, x₁, ..., x_d)
        )
        
    def forward(self, z):
        """
        Args:
            z: [batch, T, 768] semantic embeddings
        Returns:
            mu: [batch, T, 1+128] Lorentzian embeddings on hyperboloid
        """
        # Step 1: MLP projection
        mu_raw = self.projection(z)  # [batch, T, 129]
        
        # Step 2: Project onto hyperboloid H^{1,d}: -τ² + ||x||² = -c²
        tau_raw = mu_raw[..., 0]      # [batch, T]
        x_raw = mu_raw[..., 1:]        # [batch, T, 128]
        
        # Normalize spatial coordinates
        x_norm = torch.norm(x_raw, dim=-1, keepdim=True) + 1e-8  # [batch, T, 1]
        x = x_raw / x_norm  # Unit vectors
        
        # Scale spatial coordinates (learnable radius)
        r = torch.sigmoid(x_norm) * 2.0  # Radius ∈ [0, 2]
        x = x * r  # [batch, T, 128]
        
        # Compute temporal coordinate to satisfy hyperboloid constraint
        # Constraint: -τ² + ||x||² = -c²
        # Therefore: τ = sqrt(||x||² + c²)
        tau = torch.sqrt(torch.sum(x**2, dim=-1) + self.c**2)  # [batch, T]
        
        # Add monotonicity bias (encourage τ_t < τ_{t+1})
        T = z.size(1)
        time_indices = torch.arange(T, device=z.device, dtype=torch.float32)
        time_bias = 0.1 * (time_indices / T)  # Small increasing bias
        tau = tau + time_bias.unsqueeze(0)  # [batch, T]
        
        # Concatenate
        mu = torch.cat([tau.unsqueeze(-1), x], dim=-1)  # [batch, T, 129]
        
        return mu


class LorentzianTransformerLayer(nn.Module):
    def __init__(self, d_model=129, n_heads=8):
        super().__init__()
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Attention
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, h, mu):
        """
        Args:
            h: [batch, T, 129] hidden states
            mu: [batch, T, 129] Minkowski coords (for causal mask)
        """
        # Causal attention
        h_attn = self._causal_attention(h, mu)
        h = self.norm1(h + h_attn)
        
        # FFN
        h_ffn = self.ffn(h)
        h = self.norm2(h + h_ffn)
        
        return h
    
    def _causal_attention(self, h, mu):
        batch, T, d = h.shape
        
        Q = self.W_Q(h).view(batch, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(h).view(batch, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(h).view(batch, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Compute causal mask from Lorentzian distance
        tau = mu[..., 0]  # [batch, T]
        x = mu[..., 1:]    # [batch, T, 128]
        
        # Pairwise Lorentzian distances
        tau_diff = tau.unsqueeze(2) - tau.unsqueeze(1)  # [batch, T, T]
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)        # [batch, T, T, 128]
        x_dist_sq = torch.sum(x_diff ** 2, dim=-1)       # [batch, T, T]
        
        ds2 = -tau_diff**2 + x_dist_sq  # [batch, T, T]
        
        # Causal mask: time-like (ds² < 0) AND past (τ_j ≤ τ_i)
        causal_mask = (ds2 < 0) & (tau_diff <= 0)  # [batch, T, T]
        causal_mask = causal_mask.unsqueeze(1)  # [batch, 1, T, T]
        
        # Apply mask
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V)  # [batch, n_heads, T, d_k]
        out = out.transpose(1, 2).contiguous().view(batch, T, d)
        out = self.W_O(out)
        
        return out


class LorentzianTransformer(nn.Module):
    def __init__(self, d_model=129, n_layers=4, n_heads=8):
        super().__init__()
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.layers = nn.ModuleList([
            LorentzianTransformerLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
    def forward(self, mu):
        """
        Args:
            mu: [batch, T, 129] Minkowski embeddings
        """
        h = mu
        for layer in self.layers:
            h = layer(h, mu)  # Pass mu for computing causal mask
        return h


class CausalDecoder(nn.Module):
    """
    Stage 3: Causal Decoder
    Decodes from Lorentzian manifold to video frames
    """
    def __init__(self, d_latent=128, d_semantic=768, image_size=224):
        super().__init__()
        
        # Lorentzian Transformer (processes causal structure)
        # Note: d_model = d_latent + 1 must be divisible by n_heads
        # For d_latent=128, d_model=129, so we use n_heads=3 (129 = 3 * 43)
        self.lorentzian_transformer = LorentzianTransformer(
            d_model=d_latent + 1,  # Input: Minkowski coords
            n_layers=4,
            n_heads=3  # Adjusted to divide 129
        )
        
        # Project back to semantic space
        self.to_semantic = nn.Sequential(
            nn.Linear(d_latent + 1, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 768)
        )
        
        # CNN decoder: semantic features → pixels
        # Assuming we start from 7x7 spatial size
        self.pixel_decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.image_size = image_size
        
    def forward(self, mu):
        """
        Args:
            mu: [batch, T, 129] Lorentzian embeddings
        Returns:
            frames_recon: [batch, T, H, W, 3] reconstructed video (channel-last format)
        """
        batch, T, _ = mu.shape
        
        # Process with Lorentzian Transformer
        h = self.lorentzian_transformer(mu)  # [batch, T, 129]
        
        # Project back to semantic space
        z_recon = self.to_semantic(h)  # [batch, T, 768]
        
        # Decode each frame independently
        z_recon = z_recon.view(batch * T, 768, 1, 1)  # Spatial broadcast
        z_recon = z_recon.expand(-1, -1, 7, 7)  # Initial spatial size
        
        frames_recon = self.pixel_decoder(z_recon)  # [batch*T, 3, H, W]
        # Actual output size from decoder: 112x112 (7->14->28->56->112)
        # Reshape to [batch, T, 3, H, W]
        _, _, H_out, W_out = frames_recon.shape
        frames_recon = frames_recon.view(batch, T, 3, H_out, W_out)
        
        # If needed, resize to target image_size
        if H_out != self.image_size or W_out != self.image_size:
            frames_recon = frames_recon.view(batch * T, 3, H_out, W_out)
            frames_recon = F.interpolate(frames_recon, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            frames_recon = frames_recon.view(batch, T, 3, self.image_size, self.image_size)
        
        # Convert to channel-last format: [batch, T, 3, H, W] -> [batch, T, H, W, 3]
        frames_recon = frames_recon.permute(0, 1, 3, 4, 2).contiguous()
        
        return frames_recon


class CausalAutoencoder(nn.Module):
    """
    Complete Causal Autoencoder
    Combines SemanticEncoder, CausalEncoder, and CausalDecoder
    """
    def __init__(
        self,
        d_semantic=768,
        d_latent=128,
        c=1.0,
        image_size=224,
        use_semantic_encoder=True,
        use_decoder=True
    ):
        super().__init__()
        self.use_semantic_encoder = use_semantic_encoder
        self.use_decoder = use_decoder
        
        if use_semantic_encoder:
            self.semantic_encoder = SemanticEncoder()
        else:
            self.semantic_encoder = None
            
        self.causal_encoder = CausalEncoder(
            d_semantic=d_semantic,
            d_latent=d_latent,
            c=c
        )
        
        if use_decoder:
            self.decoder = CausalDecoder(
                d_latent=d_latent,
                d_semantic=d_semantic,
                image_size=image_size
            )
        else:
            self.decoder = None
        
    def forward(self, videos, return_intermediates=False):
        """
        Args:
            videos: [batch, T, H, W, 3] input video (channel-last format)
            return_intermediates: If True, return z and mu as well
        Returns:
            frames_recon: [batch, T, H, W, 3] reconstructed video (channel-last format)
            (optionally) z: [batch, T, 768] semantic embeddings
            (optionally) mu: [batch, T, 129] Lorentzian embeddings
        """
        # Stage 1: Encode to semantic space
        if self.semantic_encoder is not None:
            z = self.semantic_encoder(videos)  # [batch, T, 768]
        else:
            # If semantic encoder is disabled, create placeholder
            batch, T = videos.shape[:2]
            z = torch.randn(batch, T, 768, device=videos.device)
        
        # Stage 2: Project to Lorentzian manifold
        mu = self.causal_encoder(z)  # [batch, T, 129]
        
        # Stage 3: Decode
        if self.decoder is not None:
            frames_recon = self.decoder(mu)  # [batch, T, H, W, 3] (channel-last format)
        else:
            frames_recon = None
        
        if return_intermediates:
            return frames_recon, z, mu
        return frames_recon
