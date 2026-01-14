"""
Model Interface for Causal Autoencoder
Based on the research proposal
"""

import importlib
import inspect
from dataclasses import asdict

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from loss.causal_losses import compute_causal_loss
from configs.sections import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    DataConfig,
)


class CausalModelInterface(pl.LightningModule):
    """
    Lightning Module for Causal Autoencoder
    
    Supports two training modes:
    1. Stage 1: Train only CausalEncoder (projector) on Kinetics400
    2. Stage 2: Train full autoencoder (with decoder)
    """
    def __init__(
        self,
        model_cfg: ModelConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
        training_cfg: TrainingConfig,
        data_cfg: DataConfig,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg

        self.save_hyperparameters(
            {
                "model": asdict(self.model_cfg),
                "optimizer": asdict(self.optimizer_cfg),
                "scheduler": asdict(self.scheduler_cfg),
                "training": asdict(self.training_cfg),
                "data": asdict(self.data_cfg),
            }
        )

        self.model = self.__load_model()
        self.loss_weights = self.model_cfg.model_init_args.get('loss_weights', {
            'recon': 1.0,
            'causal': 1.0,
            'acausal': 0.5,
            'temporal': 0.1,
            'hyperboloid': 0.01
        })
        
        # Training stage: 'encoder_only' or 'full'
        self.training_stage = self.model_cfg.model_init_args.get('training_stage', 'encoder_only')

    def forward(self, videos, return_intermediates=False):
        """Forward pass"""
        return self.model(videos, return_intermediates=return_intermediates)

    def training_step(self, batch, batch_idx):
        """Training step"""
        videos = batch  # [batch, T, H, W, 3] (channel-last format)
        # Ensure contiguous memory layout for efficient access
        if not videos.is_contiguous():
            videos = videos.contiguous()
        
        if self.training_stage == 'encoder_only':
            # Stage 1: Train only CausalEncoder
            # Input: videos -> VideoMAE -> z -> CausalEncoder -> mu
            # Loss: causal losses only (no reconstruction)
            if self.model.semantic_encoder is not None:
                with torch.no_grad():
                    z = self.model.semantic_encoder(videos)  # [batch, T, 768]
            else:
                batch, T = videos.shape[:2]
                z = torch.randn(batch, T, 768, device=videos.device)
            
            mu = self.model.causal_encoder(z)  # [batch, T, 129]
            
            # Compute causal losses (no reconstruction)
            # For encoder-only training, we use placeholder reconstruction
            # In practice, you might want to add a reconstruction head or use different losses
            dummy_recon = torch.zeros_like(videos)  # Placeholder
            
            loss, loss_dict = compute_causal_loss(
                frames_input=videos,
                frames_recon=dummy_recon,
                mu=mu,
                loss_weights={
                    'recon': 0.0,  # No reconstruction loss
                    'causal': self.loss_weights.get('causal', 1.0),
                    'acausal': self.loss_weights.get('acausal', 0.5),
                    'temporal': self.loss_weights.get('temporal', 0.1),
                    'hyperboloid': self.loss_weights.get('hyperboloid', 0.01)
                }
            )
            
        else:
            # Stage 2: Train full autoencoder
            frames_recon, z, mu = self.model(videos, return_intermediates=True)
            
            loss, loss_dict = compute_causal_loss(
                frames_input=videos,
                frames_recon=frames_recon,
                mu=mu,
                loss_weights=self.loss_weights
            )
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}_loss', value, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'loss_dict': loss_dict}

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        videos = batch  # [batch, T, H, W, 3] (channel-last format)
        # Ensure contiguous memory layout for efficient access
        if not videos.is_contiguous():
            videos = videos.contiguous()
        
        if self.training_stage == 'encoder_only':
            # Stage 1: Validate only CausalEncoder
            if self.model.semantic_encoder is not None:
                with torch.no_grad():
                    z = self.model.semantic_encoder(videos)
            else:
                batch, T = videos.shape[:2]
                z = torch.randn(batch, T, 768, device=videos.device)
            
            mu = self.model.causal_encoder(z)
            
            dummy_recon = torch.zeros_like(videos)
            
            loss, loss_dict = compute_causal_loss(
                frames_input=videos,
                frames_recon=dummy_recon,
                mu=mu,
                loss_weights={
                    'recon': 0.0,
                    'causal': self.loss_weights.get('causal', 1.0),
                    'acausal': self.loss_weights.get('acausal', 0.5),
                    'temporal': self.loss_weights.get('temporal', 0.1),
                    'hyperboloid': self.loss_weights.get('hyperboloid', 0.01)
                }
            )
        else:
            # Stage 2: Validate full autoencoder
            frames_recon, z, mu = self.model(videos, return_intermediates=True)
            
            loss, loss_dict = compute_causal_loss(
                frames_input=videos,
                frames_recon=frames_recon,
                mu=mu,
                loss_weights=self.loss_weights
            )
            
            # Compute PSNR for reconstruction quality
            # Both frames_recon and videos are in channel-last format [batch, T, H, W, 3]
            mse = F.mse_loss(frames_recon, videos)
            psnr = -10 * torch.log10(mse + 1e-10)
            self.log('val_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val_{key}_loss', value, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'loss_dict': loss_dict}

    def test_step(self, batch, batch_idx):
        """Test step"""
        videos = batch  # [batch, T, H, W, 3] (channel-last format)
        # Ensure contiguous memory layout for efficient access
        if not videos.is_contiguous():
            videos = videos.contiguous()
        
        if self.training_stage == 'encoder_only':
            if self.model.semantic_encoder is not None:
                with torch.no_grad():
                    z = self.model.semantic_encoder(videos)
            else:
                batch, T = videos.shape[:2]
                z = torch.randn(batch, T, 768, device=videos.device)
            
            mu = self.model.causal_encoder(z)
            dummy_recon = torch.zeros_like(videos)
            
            loss, loss_dict = compute_causal_loss(
                frames_input=videos,
                frames_recon=dummy_recon,
                mu=mu,
                loss_weights={
                    'recon': 0.0,
                    'causal': self.loss_weights.get('causal', 1.0),
                    'acausal': self.loss_weights.get('acausal', 0.5),
                    'temporal': self.loss_weights.get('temporal', 0.1),
                    'hyperboloid': self.loss_weights.get('hyperboloid', 0.01)
                }
            )
        else:
            frames_recon, z, mu = self.model(videos, return_intermediates=True)
            
            loss, loss_dict = compute_causal_loss(
                frames_input=videos,
                frames_recon=frames_recon,
                mu=mu,
                loss_weights=self.loss_weights
            )
            
            # Both frames_recon and videos are in channel-last format [batch, T, H, W, 3]
            mse = F.mse_loss(frames_recon, videos)
            psnr = -10 * torch.log10(mse + 1e-10)
            self.log('test_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)
        
        for key, value in loss_dict.items():
            self.log(f'test_{key}_loss', value, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'loss_dict': loss_dict}

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        try:
            optimizer_class = getattr(torch.optim, self.optimizer_cfg.name)
        except AttributeError as exc:
            raise ValueError(f"Invalid optimizer: OPTIMIZER.{self.optimizer_cfg.name}") from exc

        optimizer_arguments = dict(self.optimizer_cfg.arguments or {})
        
        # For encoder_only stage, only optimize CausalEncoder
        if self.training_stage == 'encoder_only':
            params = list(self.model.causal_encoder.parameters())
        else:
            params = list(self.model.parameters())
            # Freeze semantic encoder if it exists
            if self.model.semantic_encoder is not None:
                for param in self.model.semantic_encoder.parameters():
                    param.requires_grad = False
        
        optimizer_instance = optimizer_class(params=params, **optimizer_arguments)

        learning_rate_scheduler_cfg = self.scheduler_cfg.learning_rate
        if not learning_rate_scheduler_cfg.enabled:
            return [optimizer_instance]

        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, learning_rate_scheduler_cfg.name)
        except AttributeError as exc:
            raise ValueError(
                f"Invalid learning rate scheduler: SCHEDULER.learning_rate.{learning_rate_scheduler_cfg.name}."
            ) from exc

        scheduler_arguments = dict(learning_rate_scheduler_cfg.arguments or {})
        scheduler_instance = scheduler_class(optimizer=optimizer_instance, **scheduler_arguments)

        return [optimizer_instance], [scheduler_instance]
    
    @staticmethod
    def filter_init_args(cls, config_dict):
        """Filter initialization arguments"""
        init_args = dict()
        for name in inspect.signature(cls.__init__).parameters.keys():
            if name not in ('self'):
                if name in config_dict:
                    init_args[name] = config_dict[name]
        return init_args

    def __load_model(self):
        """Load model from config"""
        file_name = self.model_cfg.file_name
        class_name = self.model_cfg.class_name
        if class_name is None:
            raise ValueError("MODEL.class_name must be specified in the configuration.")
        if file_name is None:
            raise ValueError("MODEL.file_name must be specified in the configuration.")
        try:
            model_class = getattr(importlib.import_module('model.' + file_name, package=__package__), class_name)
        except Exception:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {file_name}.{class_name}!')

        model_init_kwargs = self.model_cfg.model_init_args
        filtered_model_init_kwargs = self.filter_init_args(cls=model_class, config_dict=model_init_kwargs)
        model = model_class(**filtered_model_init_kwargs)
        if self.training_cfg.use_compile:
            model = torch.compile(model)
        return model
