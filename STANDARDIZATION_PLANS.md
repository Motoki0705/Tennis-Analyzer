# Tennis Systems Refactoring Plans

## Overview

This document outlines the refactoring plans for Court, Player, and Event modules to align them with the standardized Ball module architecture. The focus is on restructuring existing code for consistency, maintainability, and flexibility without adding new functionality. Each plan is divided into phases for systematic implementation.

-----

## Court Module Refactoring Plan

### Current State Assessment

**Existing Assets**:

  - 4 model architectures: LiteTrackNet, SwinUNet, VitUNet, FPN
  - 4 model-specific LitModules with hardcoded parameters
  - Basic training script with manual callback setup
  - Heatmap visualization callback
  - Court dataset and datamodule

**Refactoring Needs**:

  - Replace 4 model-specific LitModules with 1 generic LitModule
  - Extract hardcoded parameters to configuration files
  - Standardize training script to match ball module patterns
  - Restructure configuration hierarchy for consistency

### Phase 1: LitModule Consolidation

#### 1.1 Create Generic LitModule

**File**: `src/court/lit_module/lit_generic_court_model.py`

**Refactoring Goal**: Replace 4 existing LitModules with 1 generic implementation

**Current Files to Replace**:

  - `lit_lite_tracknet_focal.py` (128 lines, hardcoded focal loss)
  - `lit_swin_unet.py` (similar structure)
  - `lit_vit_unet.py` (similar structure)
  - `lit_fpn.py` (similar structure)

**New Implementation**:

```python
class LitGenericCourtModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: Dict[str, Any],
        optimizer_params: Dict[str, Any], 
        scheduler_params: Dict[str, Any],
        accuracy_threshold: float = 5.0,
        num_log_images: int = 4,
    ):
        # Consolidate common logic from 4 existing LitModules
```

**Benefits**:

  - Reduce code duplication from \~500 lines to \~180 lines
  - Centralize court-specific logic (heatmap accuracy, visualization)
  - Enable flexible model injection like ball module

#### 1.2 Extract Configuration Parameters

**Current State**: Parameters hardcoded in LitModules
**Target**: Parameters in configuration files

**Parameter Migration**:

From `lit_lite_tracknet_focal.py`:

```python
# Current hardcoded parameters
focal_alpha: 1.0
focal_gamma: 2.0  
lr: 0.0001
accuracy_threshold: 5
```

To `configs/train/court/litmodule/generic_focal_loss.yaml`:

```yaml
_target_: src.court.lit_module.lit_generic_court_model.LitGenericCourtModel
model: ${model}
criterion:
  alpha: 1.0
  gamma: 2.0
  reduction: "mean"
optimizer_params:
  lr: 0.0001
  weight_decay: 0.0001
```

**Configuration Restructure**:

```
configs/train/court/
├── lite_tracknet_generic.yaml          # Main config (rename from config.yaml)
├── swin_unet_generic.yaml              # New main config  
├── vit_unet_generic.yaml               # New main config
├── fpn_generic.yaml                    # New main config
├── model/                              # New directory
│   ├── lite_tracknet.yaml              # Extract model params
│   ├── swin_unet.yaml                  # Extract model params
│   ├── vit_unet.yaml                   # Extract model params
│   └── fpn.yaml                        # Extract model params
├── litmodule/
│   └── generic_focal_loss.yaml         # Unified LitModule config
├── litdatamodule/ (existing)
├── trainer/ (existing)
└── callbacks/ (existing)
```

### Phase 2: Training Script Refactoring

#### 2.1 Refactor Training Script

**File**: `src/court/api/train.py`

**Current Issues** (102 lines):

  - Manual callback instantiation
  - No checkpoint resumption
  - Basic error handling
  - No performance tracking

**Refactoring Changes**:

  - Extract callback setup logic to functions (like ball module)
  - Add checkpoint resumption capability
  - Implement error handling and logging
  - Standardize configuration path handling

**Functions to Add**:

```python
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]
def setup_callbacks(cfg: DictConfig) -> list  
def get_best_model_info(callbacks: list) -> tuple
```

**Result**: Consistent training script pattern across all modules

#### 2.2 Create Batch Training Script

**New File**: `scripts/train/court/train_all_models.sh`

**Purpose**: Replace manual individual training with automated batch processing

**Models to Process**:

  - `lite_tracknet_generic` (existing LiteTrackNet)
  - `swin_unet_generic` (existing SwinUNet)
  - `vit_unet_generic` (existing VitUNet)
  - `fpn_generic` (existing FPN)

**Features** (copied from ball module):

  - Sequential model training
  - Checkpoint archiving to `checkpoints/court/`
  - Environment validation
  - Comprehensive logging

### Phase 3: Integration and Validation

#### 3.1 Callback Integration

**Existing**: `HeatmapVisualizerCallback` (working)
**Refactoring**: Ensure compatibility with generic LitModule

  - Update callback to work with injected model
  - Maintain existing visualization functionality
  - No new features, just compatibility

#### 3.2 Backward Compatibility Testing

  - Verify existing checkpoints can still be loaded
  - Ensure training metrics remain consistent
  - Test all 4 model architectures with new system
  - Validate configuration migration works correctly

### Expected Outcomes

  - **Code Reduction**: 4 LitModules → 1 generic LitModule (\~70% less code)
  - **Consistency**: Training patterns match ball module exactly
  - **Maintainability**: Centralized configuration and logic
  - **Flexibility**: Easy model parameter tuning via configs
  - **Zero Functionality Loss**: All existing capabilities preserved

-----

## Player Module Refactoring Plan

### Current State Assessment

**Existing Assets**:

  - 1 model architecture: RT-DETR (HuggingFace integration)
  - 1 model-specific LitModule with hardcoded parameters
  - Basic training script with manual callback setup
  - COCO dataset and datamodule
  - Self-training components in `self_training/` directory

**Refactoring Needs**:

  - Extract hardcoded parameters from LitModule to configuration
  - Standardize training script to match ball module patterns
  - Restructure configuration hierarchy
  - Organize self-training components

### Phase 1: LitModule Refactoring

#### 1.1 Create Generic LitModule

**File**: `src/player/lit_module/lit_generic_player_model.py`

**Refactoring Goal**: Replace hardcoded LitModule with configurable generic version

**Current File to Replace**:

  - `lit_rtdetr.py` (\~150 lines, hardcoded RT-DETR logic)

**Parameter Extraction**:

From `lit_rtdetr.py`:

```python
# Current hardcoded parameters
pretrained_model_name_or_path: "PekingU/rtdetr_v2_r18vd"
lr: 0.0001
lr_backbone: 0.00001
num_freeze_epoch: 3
```

To configuration:

```yaml
# litmodule/generic_detection.yaml
_target_: src.player.lit_module.lit_generic_player_model.LitGenericPlayerModel
model: ${model}
optimizer_params:
  lr: 0.0001
  lr_backbone: 0.00001
scheduler_params:
  num_freeze_epoch: 3
```

#### 1.2 Model Configuration Extraction

**Current**: RT-DETR parameters embedded in LitModule
**Target**: Separate model configuration

**New Structure**:

```
configs/train/player/
├── rtdetr_generic.yaml                  # Main config (rename from config.yaml)
├── model/                               # New directory
│   └── rtdetr.yaml                      # Extract RT-DETR specific params
├── litmodule/
│   └── generic_detection.yaml           # Generic LitModule config
├── litdatamodule/ (existing)
├── trainer/ (existing)
└── callbacks/ (existing)
```

### Phase 2: Training Script Refactoring

#### 2.1 Refactor Training Script

**File**: `src/player/api/train.py`

**Current Issues** (103 lines):

  - Manual callback instantiation
  - No checkpoint resumption
  - Basic error handling
  - Hard-coded test execution logic

**Refactoring Changes**:

  - Extract callback setup logic to functions (like ball module)
  - Add checkpoint resumption capability
  - Implement error handling and logging
  - Standardize configuration path handling

**Functions to Add**:

```python
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]
def setup_callbacks(cfg: DictConfig) -> list
def get_best_model_info(callbacks: list) -> tuple
```

#### 2.2 Create Batch Training Script

**New File**: `scripts/train/player/train_all_models.sh`

**Purpose**: Automated training for player models

**Models to Process**:

  - `rtdetr_generic` (existing RT-DETR)

**Features** (copied from ball module):

  - Checkpoint archiving to `checkpoints/player/`
  - Environment validation
  - Comprehensive logging

### Phase 3: Self-Training Organization

#### 3.1 Organize Self-Training Components

**Current Files in** `src/player/self_training/`:

  - `clip_visualizer.py` (visualization utilities)
  - `clipwise_player_tracking.py` (tracking logic)
  - `specipy_player_by_FT_detr.py` (fine-tuning utilities)
  - `track_clip_globally.py` (global tracking)

**Refactoring Goal**: Organize and document existing components

  - Add proper documentation to each component
  - Create unified interface for self-training pipeline
  - Add configuration support where missing
  - **No new functionality**, just organization

### Phase 4: Integration and Validation

#### 4.1 Backward Compatibility Testing

  - Verify existing RT-DETR checkpoints can be loaded
  - Ensure training metrics remain consistent
  - Test COCO dataset loading
  - Validate self-training components still work

### Expected Outcomes

  - **Code Reduction**: Hardcoded parameters → configurable system
  - **Consistency**: Training patterns match ball module exactly
  - **Organization**: Self-training components properly documented
  - **Maintainability**: Centralized configuration and logic
  - **Zero Functionality Loss**: All existing capabilities preserved

-----

## Event Module Refactoring Plan

### Current State Assessment

**Existing Assets**:

  - 1 model architecture: Transformer V2 (multi-modal time-series)
  - 1 model-specific LitModule with hardcoded parameters
  - Basic training script with manual callback setup
  - Event dataset and datamodule with balanced dataset support
  - Multi-modal input handling (ball, court, player, pose)

**Refactoring Needs**:

  - Extract hardcoded parameters from LitModule to configuration
  - Standardize training script to match ball module patterns
  - Restructure configuration hierarchy
  - Improve balanced dataset configuration handling

### Phase 1: LitModule Refactoring

#### 1.1 Create Generic LitModule

**File**: `src/event/lit_module/lit_generic_event_model.py`

**Refactoring Goal**: Replace hardcoded LitModule with configurable generic version

**Current File to Replace**:

  - `lit_transformer_v2.py` (\~200 lines, hardcoded Transformer logic)

**Parameter Extraction**:

From `lit_transformer_v2.py`:

```python
# Current hardcoded parameters
d_model: 128
nhead: 8
num_layers: 4
dropout: 0.1
lr: 0.0005
no_hit_weight: 0.01
hit_weight: 1.0
bounce_weight: 1.0
```

To configuration:

```yaml
# litmodule/generic_classification.yaml
_target_: src.event.lit_module.lit_generic_event_model.LitGenericEventModel
model: ${model}
criterion:
  no_hit_weight: 0.01
  hit_weight: 1.0
  bounce_weight: 1.0
  clarity_weight: 0.02
optimizer_params:
  lr: 0.0005
  weight_decay: 0.001
```

#### 1.2 Model Configuration Extraction

**Current**: Transformer parameters embedded in LitModule
**Target**: Separate model configuration

**New Structure**:

```
configs/train/event/
├── transformer_v2_generic.yaml          # Main config (rename from config.yaml)
├── model/                               # New directory
│   └── transformer_v2.yaml              # Extract Transformer specific params
├── litmodule/
│   └── generic_classification.yaml      # Generic LitModule config
├── litdatamodule/ (existing)
├── trainer/ (existing)
└── callbacks/ (existing)
```

### Phase 2: Training Script Refactoring

#### 2.1 Refactor Training Script

**File**: `src/event/api/train.py`

**Current Issues** (100 lines):

  - Manual callback instantiation
  - No checkpoint resumption
  - Basic error handling
  - Hardcoded balanced dataset path logic

**Refactoring Changes**:

  - Extract callback setup logic to functions (like ball module)
  - Add checkpoint resumption capability
  - Implement error handling and logging
  - Improve balanced dataset configuration handling

**Functions to Add**:

```python
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]
def setup_callbacks(cfg: DictConfig) -> list
def get_best_model_info(callbacks: list) -> tuple
```

#### 2.2 Create Batch Training Script

**New File**: `scripts/train/event/train_all_models.sh`

**Purpose**: Automated training for event models

**Models to Process**:

  - `transformer_v2_generic` (existing Transformer V2)

**Features** (copied from ball module):

  - Checkpoint archiving to `checkpoints/event/`
  - Environment validation
  - Comprehensive logging

### Phase 3: Dataset Configuration Improvement

#### 3.1 Balanced Dataset Handling

**Current Issue**: Hardcoded balanced dataset path in training script
**Target**: Configuration-driven balanced dataset support

**Refactoring**:

  - Move balanced dataset logic to datamodule configuration
  - Remove hardcoded path handling from training script
  - Make balanced dataset an optional configuration parameter

#### 3.2 Multi-Modal Configuration

**Current**: Multi-modal inputs handled in dataset
**Enhancement**: Make input modalities configurable

  - Ball trajectory features
  - Court geometry features
  - Player motion features
  - Pose estimation features

### Phase 4: Integration and Validation

#### 4.1 Backward Compatibility Testing

  - Verify existing Transformer V2 checkpoints can be loaded
  - Ensure training metrics remain consistent
  - Test balanced dataset functionality
  - Validate multi-modal input processing

### Expected Outcomes

  - **Code Reduction**: Hardcoded parameters → configurable system
  - **Consistency**: Training patterns match ball module exactly
  - **Flexibility**: Balanced dataset and modality configuration
  - **Maintainability**: Centralized configuration and logic
  - **Zero Functionality Loss**: All existing capabilities preserved

-----

## Implementation Principles

1.  **No New Features**: Focus purely on restructuring existing code
2.  **Preserve Functionality**: All existing capabilities must be maintained
3.  **Configuration Extraction**: Move hardcoded parameters to config files
4.  **Pattern Consistency**: Align all modules with ball module patterns
5.  **Backward Compatibility**: Ensure existing checkpoints still work

### Success Criteria

1.  **Code Reduction**: Eliminate duplicate LitModule implementations
2.  **Consistency**: All modules follow identical training script patterns
3.  **Maintainability**: Centralized configuration and reusable components
4.  **Performance**: No degradation in training speed or model accuracy
5.  **Compatibility**: All existing checkpoints and configurations work

### Risk Mitigation

1.  **Incremental Refactoring**: Complete one module before starting the next
2.  **Backup Strategy**: Keep original implementations until validation complete
3.  **Regression Testing**: Validate training results match previous versions
4.  **Rollback Plan**: Ability to revert to original implementation if needed

### Expected Code Reduction

  - **Court Module**: 4 LitModules → 1 generic (\~70% reduction)
  - **Player Module**: Hardcoded parameters → configurable (\~30% complexity reduction)
  - **Event Module**: Hardcoded parameters → configurable (\~30% complexity reduction)
  - **Training Scripts**: Standardized error handling and checkpoint management across all modules

This refactoring will create a unified, maintainable codebase with consistent patterns across all tennis analysis components while preserving all existing functionality.