"""
Config Schema for Causal Autoencoder (Stage 1 and Stage 2)
Separate schemas for stage 1 (encoder only) and stage 2 (full autoencoder)
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple, Union

from omegaconf import OmegaConf

from configs.config_tracker import (
    ConfigUsageTracker,
    TrackedConfigMixin,
    attach_tracker,
)
from configs.sections import (
    TrainingConfig,
    DistributedConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    LoggerConfig,
    CheckpointConfig,
)


@dataclass
class AppConfigStage1(TrackedConfigMixin):
    """Config schema for Stage 1: CausalEncoder training (encoder only)"""
    TRAINING: TrainingConfig = field(default_factory=TrainingConfig)
    DISTRIBUTED: DistributedConfig = field(default_factory=DistributedConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)
    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    LOGGER: LoggerConfig = field(default_factory=LoggerConfig)
    CHECKPOINT: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class AppConfigStage2(TrackedConfigMixin):
    """Config schema for Stage 2: Full Autoencoder training"""
    TRAINING: TrainingConfig = field(default_factory=TrainingConfig)
    DISTRIBUTED: DistributedConfig = field(default_factory=DistributedConfig)
    DATA: DataConfig = field(default_factory=DataConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    OPTIMIZER: OptimizerConfig = field(default_factory=OptimizerConfig)
    SCHEDULER: SchedulerConfig = field(default_factory=SchedulerConfig)
    LOGGER: LoggerConfig = field(default_factory=LoggerConfig)
    CHECKPOINT: CheckpointConfig = field(default_factory=CheckpointConfig)


# Union type for both stages
AppConfigCausal = Union[AppConfigStage1, AppConfigStage2]


def _validate_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def validate_app_config_stage1(cfg: AppConfigStage1):
    """Validate configuration for Stage 1 (encoder only training)"""
    errors: List[str] = []

    if not cfg.MODEL.file_name:
        errors.append("MODEL.file_name must be specified.")
    if not cfg.MODEL.class_name:
        errors.append("MODEL.class_name must be specified.")

    dataset = cfg.DATA.dataset
    if not dataset.file_name:
        errors.append("DATA.dataset.file_name must be specified.")
    if not dataset.class_name:
        errors.append("DATA.dataset.class_name must be specified.")

    dataloader = cfg.DATA.dataloader
    if not _validate_positive(dataloader.batch_size):
        errors.append("DATA.dataloader.batch_size must be positive.")

    training = cfg.TRAINING
    if not _validate_positive(training.max_epochs):
        errors.append("TRAINING.max_epochs must be positive.")

    optimizer = cfg.OPTIMIZER
    if not optimizer.name:
        errors.append("OPTIMIZER.name must be specified.")
    if not optimizer.arguments:
        errors.append("OPTIMIZER.arguments must provide keyword args.")

    lr_sched = cfg.SCHEDULER.learning_rate
    if lr_sched.enabled:
        if not lr_sched.name:
            errors.append("SCHEDULER.learning_rate.name must be specified when enabled.")
        if not isinstance(lr_sched.arguments, dict):
            errors.append("SCHEDULER.learning_rate.arguments must be a mapping.")

    # Stage 1 specific validation
    model_init_args = cfg.MODEL.model_init_args
    training_stage = model_init_args.get("training_stage", "")
    if training_stage != "encoder_only":
        errors.append(f"Stage 1 config must have MODEL.model_init_args.training_stage='encoder_only', got '{training_stage}'")
    
    use_decoder = model_init_args.get("use_decoder", True)
    if use_decoder:
        errors.append("Stage 1 config must have MODEL.model_init_args.use_decoder=False")

    if errors:
        error_text = "\n - ".join(errors)
        raise ValueError(f"Configuration validation failed for Stage 1:\n - {error_text}")


def validate_app_config_stage2(cfg: AppConfigStage2):
    """Validate configuration for Stage 2 (full autoencoder training)"""
    errors: List[str] = []

    if not cfg.MODEL.file_name:
        errors.append("MODEL.file_name must be specified.")
    if not cfg.MODEL.class_name:
        errors.append("MODEL.class_name must be specified.")

    dataset = cfg.DATA.dataset
    if not dataset.file_name:
        errors.append("DATA.dataset.file_name must be specified.")
    if not dataset.class_name:
        errors.append("DATA.dataset.class_name must be specified.")

    dataloader = cfg.DATA.dataloader
    if not _validate_positive(dataloader.batch_size):
        errors.append("DATA.dataloader.batch_size must be positive.")

    training = cfg.TRAINING
    if not _validate_positive(training.max_epochs):
        errors.append("TRAINING.max_epochs must be positive.")

    optimizer = cfg.OPTIMIZER
    if not optimizer.name:
        errors.append("OPTIMIZER.name must be specified.")
    if not optimizer.arguments:
        errors.append("OPTIMIZER.arguments must provide keyword args.")

    lr_sched = cfg.SCHEDULER.learning_rate
    if lr_sched.enabled:
        if not lr_sched.name:
            errors.append("SCHEDULER.learning_rate.name must be specified when enabled.")
        if not isinstance(lr_sched.arguments, dict):
            errors.append("SCHEDULER.learning_rate.arguments must be a mapping.")

    # Stage 2 specific validation
    model_init_args = cfg.MODEL.model_init_args
    training_stage = model_init_args.get("training_stage", "")
    if training_stage != "full":
        errors.append(f"Stage 2 config must have MODEL.model_init_args.training_stage='full', got '{training_stage}'")
    
    use_decoder = model_init_args.get("use_decoder", False)
    if not use_decoder:
        errors.append("Stage 2 config must have MODEL.model_init_args.use_decoder=True")

    if errors:
        error_text = "\n - ".join(errors)
        raise ValueError(f"Configuration validation failed for Stage 2:\n - {error_text}")


def detect_stage_from_config(user_cfg: Any) -> int:
    """
    Detect which stage the config is for based on MODEL.model_init_args.training_stage
    
    Args:
        user_cfg: OmegaConf DictConfig or plain dict
    
    Returns:
        1 for stage 1 (encoder_only), 2 for stage 2 (full), or raises ValueError
    """
    try:
        # Handle both OmegaConf and plain dict
        if hasattr(user_cfg, 'get'):  # OmegaConf DictConfig
            model_cfg = user_cfg.get("MODEL", {})
            model_init_args = model_cfg.get("model_init_args", {}) if hasattr(model_cfg, 'get') else {}
            training_stage = model_init_args.get("training_stage", "")
            use_decoder = model_init_args.get("use_decoder", None)
        else:  # Plain dict
            training_stage = user_cfg.get("MODEL", {}).get("model_init_args", {}).get("training_stage", "")
            use_decoder = user_cfg.get("MODEL", {}).get("model_init_args", {}).get("use_decoder", None)
        
        if training_stage == "encoder_only":
            return 1
        elif training_stage == "full":
            return 2
        else:
            # Try to infer from use_decoder if training_stage is not set
            if use_decoder is False:
                return 1
            elif use_decoder is True:
                return 2
            else:
                raise ValueError(f"Could not detect stage from config. training_stage='{training_stage}', use_decoder={use_decoder}")
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Could not detect stage from config: {e}")


def load_config_with_schema(path: str) -> Tuple[AppConfigCausal, ConfigUsageTracker]:
    """
    Load config with schema, automatically detecting stage 1 or stage 2
    
    Returns:
        Tuple of (AppConfigCausal, ConfigUsageTracker)
    """
    user_cfg = OmegaConf.load(path)
    
    # Detect stage (before merging, so we can use the original config)
    user_cfg_plain = OmegaConf.to_container(user_cfg, resolve=True)
    try:
        stage = detect_stage_from_config(user_cfg_plain)
    except ValueError as e:
        raise ValueError(f"Failed to detect stage from config file {path}: {e}")
    
    # Load appropriate schema
    if stage == 1:
        structured_default = OmegaConf.structured(AppConfigStage1)
        merged_cfg = OmegaConf.merge(structured_default, user_cfg)
        cfg_obj: AppConfigStage1 = OmegaConf.to_object(merged_cfg)
        validate_app_config_stage1(cfg_obj)
    elif stage == 2:
        structured_default = OmegaConf.structured(AppConfigStage2)
        merged_cfg = OmegaConf.merge(structured_default, user_cfg)
        cfg_obj: AppConfigStage2 = OmegaConf.to_object(merged_cfg)
        validate_app_config_stage2(cfg_obj)
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    tracker = ConfigUsageTracker()
    attach_tracker(cfg_obj, tracker)
    return cfg_obj, tracker


def app_config_causal_to_dict(cfg: AppConfigCausal) -> Dict[str, Any]:
    """Convert AppConfigCausal to dictionary"""
    return asdict(cfg)
