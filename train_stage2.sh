#!/bin/bash
# Stage 2: Full Autoencoder Training
# Based on the research proposal
# Note: This stage can optionally load the CausalEncoder checkpoint from Stage 1

# Uncomment the following line to load Stage 1 checkpoint:
# python main_causal.py --config_path configs/config_causal_stage2.yaml --load_manual_checkpoint 'lightning_logs/causal_encoder_kinetics400_stage1/version_0/checkpoints/best-epoch=XXX-val_loss_epoch=X.XXXXX.ckpt' --weights_only

# Or train from scratch:
python main_causal.py --config_path configs/config_causal_stage2.yaml
