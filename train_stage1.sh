#!/bin/bash
# Stage 1: Train CausalEncoder (Projector) on Kinetics400
# Based on the research proposal

python main_causal.py --config_path configs/config_causal_stage1.yaml
