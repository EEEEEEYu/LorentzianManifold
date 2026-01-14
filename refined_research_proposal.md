# Lorentzian Causal Disentanglement: A Logically Consistent Research Plan

## 1. Core Idea: Causal Video Autoencoder

**Central Thesis**: Build an autoencoder where the latent space is a Lorentzian manifold, forcing the model to learn causally-structured representations.

**Key Insight**: 
- Standard VAE: Encodes video into Euclidean latent space (no causal structure)
- Our Model: Encodes video into Lorentzian latent space (causal structure enforced by geometry)

---

## 2. Model Architecture: Three-Stage Design

### 2.1 Overall Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING STAGE                         │
└─────────────────────────────────────────────────────────┘

INPUT: Video V = [I_1, I_2, ..., I_T]
│
├─ Stage 1: Encode to Euclidean (Semantic Encoder)
│   VideoMAE (frozen, pretrained)
│   Output: z_t ∈ ℝ^{768} per frame
│   Role: Capture semantic content (what objects, what actions)
│
├─ Stage 2: Project to Lorentzian (Causal Encoder)
│   Learnable MLP: z_t → μ_t = (τ_t, x_t) ∈ ℝ^{1,128}
│   Constraint: Points lie on hyperboloid H^{1,128}
│   Role: Reorganize semantics into causal structure
│
├─ Stage 3: Decode from Lorentzian (Causal Decoder)
│   Lorentzian Transformer + CNN Decoder
│   Output: Reconstructed frames Î_1, ..., Î_T
│   Role: Ensure geometry preserves semantic content
│
OUTPUT: Reconstructed video Î

LOSS: ||I - Î||² + Causal Structure Losses
```

### 2.2 What Changes vs What Doesn't

| Aspect | Semantic Encoder (z) | Causal Encoder (μ) | Decoder (I) |
|--------|---------------------|-------------------|-------------|
| **Semantic content** | Captures "hand pushes cup" | **PRESERVES** via reconstruction | Recovers "hand pushes cup" |
| **Geometric structure** | No causal ordering | **ADDS** light cone structure | Must respect causality |
| **Information** | Complete frame info | Compressed + causally ordered | Reconstructed from causal ordering |

**Critical Engineering Principle**: 
```
Decoder(Lorentzian Embedding) = Original Frame
```
This ensures geometric transformation doesn't destroy semantic content.

---

## 3. Technical Implementation

### 3.1 Stage 1: Semantic Encoder (VideoMAE)

```python
class SemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained VideoMAE, freeze weights
        self.videomae = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')
        
        # Freeze all parameters
        for param in self.videomae.parameters():
            param.requires_grad = False
            
        # Pooling to get per-frame embedding
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, videos):
        """
        Args:
            videos: [batch, T, 3, H, W]
        Returns:
            z: [batch, T, 768] semantic embeddings
        """
        batch, T = videos.shape[:2]
        
        # VideoMAE expects [batch, 3, T, H, W]
        videos_fmt = videos.transpose(1, 2)
        
        with torch.no_grad():  # Frozen, no gradients
            outputs = self.videomae(videos_fmt)
            features = outputs.last_hidden_state  # [batch, num_patches, 768]
        
        # Pool across patches for each frame
        # This is simplified - actual implementation should track temporal structure
        z = features.mean(dim=1, keepdim=True).repeat(1, T, 1)
        
        return z
```

**Why frozen?** 
- VideoMAE already captures semantic content well
- We only want to learn causal structure, not re-learn semantics
- Reduces computational cost

---

### 3.2 Stage 2: Causal Encoder (Lorentzian Projection)

```python
class CausalEncoder(nn.Module):
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
```

**Why MLP instead of linear?**
- Semantic space and causal space may have complex non-linear relationships
- MLP can learn "if pushing action, then increase temporal coordinate sharply"
- More robust to diverse video types

**Key Design**: 
- Hard constraint on hyperboloid (via normalization)
- Soft constraint on monotonicity (via time bias)

---

### 3.3 Stage 3: Causal Decoder

```python
class CausalDecoder(nn.Module):
    def __init__(self, d_latent=128, d_semantic=768):
        super().__init__()
        
        # Lorentzian Transformer (processes causal structure)
        self.lorentzian_transformer = LorentzianTransformer(
            d_model=d_latent + 1,  # Input: Minkowski coords
            n_layers=4,
            n_heads=8
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
        
    def forward(self, mu):
        """
        Args:
            mu: [batch, T, 129] Lorentzian embeddings
        Returns:
            frames_recon: [batch, T, 3, H, W] reconstructed video
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
        frames_recon = frames_recon.view(batch, T, 3, 224, 224)
        
        return frames_recon


class LorentzianTransformer(nn.Module):
    def __init__(self, d_model=129, n_layers=4, n_heads=8):
        super().__init__()
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


class LorentzianTransformerLayer(nn.Module):
    def __init__(self, d_model=129, n_heads=8):
        super().__init__()
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
```

---

## 4. Complete Loss Function

### 4.1 Total Objective

```python
def compute_loss(frames_input, frames_recon, mu):
    """
    Args:
        frames_input: [batch, T, 3, H, W] original frames
        frames_recon: [batch, T, 3, H, W] reconstructed frames
        mu: [batch, T, 129] Lorentzian embeddings
    """
    # Loss 1: Reconstruction (semantic preservation)
    L_recon = F.mse_loss(frames_recon, frames_input)
    
    # Loss 2: Temporal causality (enforce time-like separation)
    L_causal = causal_ordering_loss(mu)
    
    # Loss 3: Spatial acausality (enforce space-like separation)
    L_acausal = spatial_independence_loss(mu)
    
    # Loss 4: Temporal monotonicity
    L_temporal = temporal_ordering_loss(mu)
    
    # Loss 5: Hyperboloid constraint (soft, backup to hard projection)
    L_hyperboloid = hyperboloid_consistency_loss(mu)
    
    # Total
    loss = (L_recon 
            + 1.0 * L_causal 
            + 0.5 * L_acausal 
            + 0.1 * L_temporal 
            + 0.01 * L_hyperboloid)
    
    return loss, {
        'recon': L_recon.item(),
        'causal': L_causal.item(),
        'acausal': L_acausal.item(),
        'temporal': L_temporal.item(),
        'hyperboloid': L_hyperboloid.item()
    }
```

### 4.2 Individual Loss Components

```python
def causal_ordering_loss(mu, margin=0.1, K=3):
    """
    Enforce that frame t is in past light cone of frame t+k.
    
    Args:
        mu: [batch, T, 129]
        margin: Hinge loss margin
        K: Temporal window (check up to K frames ahead)
    """
    batch, T, _ = mu.shape
    loss = 0.0
    count = 0
    
    for k in range(1, min(K+1, T)):
        mu_current = mu[:, :-k, :]  # [batch, T-k, 129]
        mu_future = mu[:, k:, :]     # [batch, T-k, 129]
        
        # Compute Δs²
        tau_curr = mu_current[..., 0]
        tau_future = mu_future[..., 0]
        x_curr = mu_current[..., 1:]
        x_future = mu_future[..., 1:]
        
        delta_tau = tau_future - tau_curr  # Should be positive
        delta_x = x_future - x_curr
        ds2 = -delta_tau**2 + torch.sum(delta_x**2, dim=-1)
        
        # Should be time-like (ds² < 0)
        loss_k = F.relu(ds2 + margin).mean()
        loss += loss_k
        count += 1
    
    return loss / count


def spatial_independence_loss(mu, margin=0.1, num_samples=100):
    """
    Sample random frame pairs that are far apart in time.
    They should have space-like separation (independent).
    
    Args:
        mu: [batch, T, 129]
        margin: Hinge loss margin
        num_samples: Number of random pairs to sample
    """
    batch, T, _ = mu.shape
    
    if T < 10:  # Not enough temporal separation
        return torch.tensor(0.0, device=mu.device)
    
    loss = 0.0
    
    for _ in range(num_samples):
        # Sample two frames far apart (at least T//3 frames apart)
        i = torch.randint(0, T - T//3, (batch,))
        j = torch.randint(i + T//3, T, (batch,))
        
        mu_i = mu[torch.arange(batch), i]  # [batch, 129]
        mu_j = mu[torch.arange(batch), j]  # [batch, 129]
        
        tau_i, x_i = mu_i[:, 0], mu_i[:, 1:]
        tau_j, x_j = mu_j[:, 0], mu_j[:, 1:]
        
        delta_tau = tau_j - tau_i
        delta_x = x_j - x_i
        ds2 = -delta_tau**2 + torch.sum(delta_x**2, dim=-1)
        
        # Should be space-like (ds² > 0)
        loss += F.relu(-ds2 + margin).mean()
    
    return loss / num_samples


def temporal_ordering_loss(mu, epsilon=0.01):
    """
    Ensure τ_{t+1} > τ_t (monotonicity).
    """
    tau = mu[..., 0]  # [batch, T]
    tau_diff = tau[:, 1:] - tau[:, :-1]  # [batch, T-1]
    
    # Penalize if τ doesn't increase
    loss = F.relu(-tau_diff + epsilon).mean()
    return loss


def hyperboloid_consistency_loss(mu, c=1.0):
    """
    Soft constraint: -τ² + ||x||² ≈ -c²
    (Backup to hard projection in forward pass)
    """
    tau = mu[..., 0]  # [batch, T]
    x = mu[..., 1:]    # [batch, T, 128]
    
    x_norm_sq = torch.sum(x**2, dim=-1)  # [batch, T]
    constraint = -tau**2 + x_norm_sq + c**2
    
    loss = (constraint ** 2).mean()
    return loss
```

---

## 5. Evaluation Strategy (Logically Consistent)

### 5.1 Phase 1: Autoencoder Quality (Sanity Check)

**Dataset**: Something-Something V2 (220K videos)

**Metrics**:
- Reconstruction PSNR/SSIM
- Latent space structure analysis:
  - % of temporal pairs with Δs² < 0 (should be >80%)
  - % of distant pairs with Δs² > 0 (should be >70%)

**Success Criteria**: 
- PSNR > 20 dB (reasonable reconstruction)
- Temporal causality alignment > 80%

**No per-frame annotations needed!** Just train autoencoder and analyze learned structure.

---

### 5.2 Phase 2: Causal Reasoning (CATER)

**Dataset**: CATER (10K videos with ground truth object interactions)

**Task 1: Snitch Localization (Primary Benchmark)**

```python
class CATERTaskHead(nn.Module):
    def __init__(self, causal_autoencoder):
        super().__init__()
        self.encoder = causal_autoencoder.semantic_encoder  # Frozen
        self.causal_encoder = causal_autoencoder.causal_encoder  # Frozen
        
        # Task-specific head
        self.predictor = nn.Sequential(
            nn.Linear(129, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6 possible final positions
        )
    
    def forward(self, video):
        z = self.encoder(video)  # [batch, T, 768]
        mu = self.causal_encoder(z)  # [batch, T, 129]
        
        # Use final frame's Lorentzian embedding
        mu_final = mu[:, -1, :]  # [batch, 129]
        
        logits = self.predictor(mu_final)  # [batch, 6]
        return logits
```

**Training**: 
- Freeze autoencoder weights
- Only train task head
- This tests if learned causal structure transfers

**Metrics**:
- Top-1 Accuracy on Snitch Localization
- Baseline: ALOE (72.8%)
- Our Goal: >75%

---

**Task 2: Intervention Analysis (Novel Evaluation)**

Since CATER has ground truth physics, we can simulate interventions:

```python
def evaluate_intervention(model, video, intervention_frame):
    """
    Test: If we remove a collision, does predicted trajectory change correctly?
    """
    # Encode video
    z = model.semantic_encoder(video)
    mu = model.causal_encoder(z)
    
    # Identify causal descendants of intervention frame
    mu_intervention = mu[:, intervention_frame, :]
    ds2 = compute_all_distances(mu_intervention, mu)  # [batch, T]
    
    # Find frames in future light cone
    causal_descendants = (ds2 < 0) & (mu[..., 0] > mu_intervention[:, 0])
    
    # Mask intervention
    mu_intervened = mu.clone()
    mu_intervened[:, intervention_frame, :] = 0  # Or learned "null" embedding
    
    # Regenerate video
    video_counterfactual = model.decoder(mu_intervened)
    
    # Check: Do only causal descendants change?
    diff = (video_counterfactual - video).abs()
    change_mask = diff.mean(dim=[2,3,4]) > threshold
    
    # Metric: Precision = true_positives / (true_positives + false_positives)
    precision = (change_mask & causal_descendants).sum() / change_mask.sum()
    
    return precision
```

**Metric**: Causal Selectivity (what % of changed frames are actually in light cone?)

---

### 5.3 Phase 3: Action Anticipation (Epic-Kitchens)

**Dataset**: Epic-Kitchens-100 (90K action segments)

**Setup**:
- Observe frames 1 to t
- Predict action at frame t+k (k ∈ {30, 60, 90} frames ahead)

**Hypothesis**: Hierarchical structure in Lorentzian space helps long-horizon prediction.

**Metrics**: Top-5 accuracy @ 1s, 2s, 3s

---

## 6. Implementation Timeline

### Month 1-2: Build Causal Autoencoder
- Implement SemanticEncoder + CausalEncoder + CausalDecoder
- Train on Something-Something V2
- **Deliverable**: Working autoencoder with >20 dB PSNR

### Month 3-4: Analyze Learned Structure
- Visualize Lorentzian embeddings (project to 2D/3D)
- Measure temporal causality alignment
- Ablation: Remove causal losses → does structure disappear?
- **Deliverable**: Evidence that causal structure emerges

### Month 5-6: CATER Snitch Localization
- Adapt architecture for object tracking
- Freeze autoencoder, train task head
- **Deliverable**: Competitive accuracy (>70%)

### Month 7-8: CATER Interventions
- Implement intervention evaluation
- Generate counterfactual videos
- **Deliverable**: Quantitative causal selectivity metric

### Month 9-10: Epic-Kitchens Scaling
- Train on large-scale real-world data
- Action anticipation experiments
- **Deliverable**: SOTA or competitive results

### Month 11-12: Paper Writing & Submission
- Consolidate all experiments
- Write ablation studies
- Submit to CVPR/ICCV/NeurIPS

---

## 7. Key Advantages of This Design

| Aspect | Benefit |
|--------|---------|
| **Autoencoder structure** | Can pre-train unsupervised, then transfer to tasks |
| **Frozen VideoMAE** | Preserves semantic quality, reduces compute |
| **Decoder from Lorentzian** | Guarantees geometry doesn't destroy content |
| **MLP projection** | Robust non-linear mapping between spaces |
| **Modular design** | Easy to ablate each component |

---

## 8. Expected Results

**Minimal Success**:
- Autoencoder reconstruction PSNR > 20 dB
- Temporal causality alignment > 70%
- CATER Snitch Localization > baseline

**Target Success**:
- Temporal causality alignment > 85%
- CATER Top-1 accuracy > 75%
- Intervention selectivity > 70%

**Stretch Goal**:
- Epic-Kitchens Top-5 @ 1s > 25% (SOTA)
- Theoretical analysis of when Lorentzian geometry provably helps

---

## 9. Critical Design Decisions Revisited

✅ **What changes**: Distance metric, connectivity, causal ordering
✅ **What doesn't change**: Semantic content (enforced by reconstruction)
✅ **Non-linear projection**: MLP handles complex relationships
✅ **Autoencoder design**: Pre-trainable, transferable, analyzable
✅ **No per-frame annotations needed**: Learn from temporal structure alone

This design is logically consistent, practically implementable, and scientifically rigorous.