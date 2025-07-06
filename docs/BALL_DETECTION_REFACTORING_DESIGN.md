# 🎾 Ball Detection System Refactoring Design Document

## 📋 Executive Summary

This document outlines the comprehensive refactoring plan for the tennis ball detection system to achieve better separation of concerns, improved reusability, and cleaner architecture.

---

## 🔍 Current State Analysis

### **Current Architecture Problems**

#### 1. **src/ball/ball_detection_module.py** (660 lines)
- ❌ **Mixed Responsibilities**: Detection logic + Visualization + API + Factory functions
- ❌ **Monolithic Structure**: All classes and functions in single file
- ❌ **Poor Separation**: Abstract classes mixed with concrete implementations
- ❌ **Hard to Test**: Tightly coupled components

#### 2. **third_party/WASB_SBDT API Inconsistency**
- ❌ **Fragmented API**: `ball_detection_api.py`, `video_demo.py`, `detectors/`
- ❌ **API Mismatch**: `process_frames()` vs `detect_frames()` inconsistency
- ❌ **Limited Reusability**: Batch processing and tracking not standardized
- ❌ **Scattered Features**: Pre/Infer/Post logic distributed across files

#### 3. **demo/ball_detection_module.py** (594 lines)
- ❌ **UI/Logic Mixed**: Gradio UI definitions with business logic
- ❌ **Monolithic Demo**: Single file for all demo functionality
- ❌ **No Frontend/Backend Separation**: Hard to reuse components

### **Current File Structure**
```
src/ball/
├── ball_detection_module.py        # 660 lines - EVERYTHING
├── lit_module/                      # Model definitions
├── models/                          # Model architectures
└── README_ball_detection_module.md

demo/
├── ball_detection_module.py        # 594 lines - UI + Logic
├── ball.py                          # Legacy demo
└── example_overlay_usage.py        # Usage examples

third_party/WASB_SBDT/src/
├── ball_detection_api.py           # High-level API
├── video_demo.py                    # Demo with SimpleDetector
└── detectors/                       # Low-level detectors
```

---

## 🎯 Target Architecture

### **Design Principles**
1. **Separation of Concerns**: Clear responsibility boundaries
2. **Single Responsibility**: Each module has one clear purpose
3. **Dependency Injection**: Flexible component composition
4. **API Consistency**: Unified interfaces across all components
5. **Testability**: Easy to unit test each component
6. **Reusability**: Components can be used independently

### **New Directory Structure**
```
src/
├── predictor/                       # 🆕 Unified prediction system
│   ├── __init__.py
│   ├── base/                        # Abstract base classes
│   │   ├── __init__.py
│   │   ├── detector.py              # BaseBallDetector
│   │   ├── preprocessor.py          # BasePreprocessor
│   │   ├── postprocessor.py         # BasePostprocessor
│   │   └── tracker.py               # BaseTracker
│   ├── ball/                        # Ball-specific implementations
│   │   ├── __init__.py
│   │   ├── lite_tracknet_detector.py
│   │   ├── wasb_sbdt_detector.py
│   │   └── factory.py               # Detector factory
│   ├── batch/                       # Batch processing utilities
│   │   ├── __init__.py
│   │   ├── processor.py             # Batch inference manager
│   │   └── utils.py                 # Batch processing utilities
│   └── visualization/               # Visualization components
│       ├── __init__.py
│       ├── overlay.py               # Video overlay functionality
│       ├── renderer.py              # Drawing utilities
│       └── config.py                # Visualization configurations

ball/                                # 📝 Refactored ball module
├── __init__.py                      # Clean API exports
├── api.py                           # Public API (simplified)
├── lit_module/                      # (unchanged)
├── models/                          # (unchanged)
└── dataset/                         # (unchanged)

demo/ball/                           # 🆕 Organized demo structure
├── __init__.py
├── backend/                         # Backend services
│   ├── __init__.py
│   ├── detection_service.py         # Business logic
│   ├── overlay_service.py           # Overlay processing
│   └── config.py                    # Configuration management
├── frontend/                        # Frontend components
│   ├── __init__.py
│   ├── gradio_app.py                # Gradio interface
│   ├── components/                  # Reusable UI components
│   │   ├── __init__.py
│   │   ├── detection_tab.py
│   │   ├── overlay_tab.py
│   │   └── common.py
│   └── templates/                   # UI templates
│       └── main.html
├── main.py                          # Demo entry point
└── README.md                        # Demo documentation

third_party/WASB_SBDT/src/           # 🔧 Enhanced API
├── __init__.py                      # Unified exports
├── api/                             # 🆕 Comprehensive API layer
│   ├── __init__.py
│   ├── detector.py                  # Unified detector API
│   ├── batch.py                     # Batch processing API
│   ├── tracking.py                  # Tracking API
│   └── utils.py                     # Utility functions
├── ball_detection_api.py           # (enhanced)
├── video_demo.py                   # (simplified to demo only)
└── detectors/                      # (unchanged)
```

---

## 🏗️ Detailed Component Design

### **1. src/predictor/ - Unified Prediction System**

#### **predictor/base/detector.py**
```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class BaseBallDetector(ABC):
    """Abstract base class for all ball detectors."""
    
    @abstractmethod
    def preprocess(self, frame_data: List[Tuple[np.ndarray, dict]]) -> List[Tuple[Any, dict]]:
        """Convert frames to model input format while preserving metadata."""
        pass
    
    @abstractmethod
    def infer(self, model_inputs: List[Tuple[Any, dict]]) -> List[Tuple[Any, dict]]:
        """Perform batch inference while maintaining metadata association."""
        pass
    
    @abstractmethod
    def postprocess(self, inference_results: List[Tuple[Any, dict]]) -> Dict[str, List[List[float]]]:
        """Convert to frame_id keyed dictionary with [x, y, conf] values."""
        pass
    
    @property
    @abstractmethod
    def frames_required(self) -> int:
        """Number of consecutive frames required by the model."""
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
```

#### **predictor/batch/processor.py**
```python
class BatchProcessor:
    """Efficient batch processing manager for ball detection."""
    
    def __init__(self, detector: BaseBallDetector, batch_size: int = 16):
        self.detector = detector
        self.batch_size = batch_size
    
    def process_video_batched(self, frame_data: List[Tuple[np.ndarray, dict]]) -> Dict[str, List[List[float]]]:
        """Process video frames in optimized batches."""
        pass
    
    def process_streaming(self, frame_iterator) -> Iterator[Dict[str, List[List[float]]]]:
        """Process frames in streaming fashion."""
        pass
```

#### **predictor/visualization/overlay.py**
```python
class VideoOverlay:
    """Handles video overlay functionality."""
    
    def __init__(self, config: OverlayConfig):
        self.config = config
    
    def create_overlay_video(
        self, 
        video_path: str, 
        detections: Dict[str, List[List[float]]], 
        output_path: str
    ) -> str:
        """Create video with detection overlays."""
        pass
    
    def render_frame(self, frame: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        """Render detections on a single frame."""
        pass
```

### **2. Enhanced Third-Party API**

#### **third_party/WASB_SBDT/src/api/detector.py**
```python
class UnifiedWASBDetector:
    """Unified WASB-SBDT detector API with complete pre/infer/post support."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.detector = TennisBallDetector(model_path, device)
    
    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """Compatibility wrapper for existing API."""
        result = self.detector.detect_frames(frames)
        if result:
            return [{'xy': [result.x, result.y], 'score': result.score}]
        return []
    
    def process_frames_batch(self, frame_sequences: List[List[np.ndarray]]) -> List[List[Dict]]:
        """Efficient batch processing of multiple frame sequences."""
        pass
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> Any:
        """Expose preprocessing step for custom pipelines."""
        pass
    
    def postprocess_predictions(self, predictions: Any) -> List[Dict]:
        """Expose postprocessing step for custom pipelines."""
        pass
```

#### **third_party/WASB_SBDT/src/api/tracking.py**
```python
class TrackingAPI:
    """Dedicated tracking API for downstream processing."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.tracker = build_tracker(config or self._default_config())
    
    def update_detections(self, detections: List[Dict]) -> Dict:
        """Update tracker with new detections."""
        return self.tracker.update(detections)
    
    def reset_tracker(self):
        """Reset tracker state."""
        self.tracker.refresh()
    
    def get_trajectory(self, window_size: int = 30) -> List[Tuple[float, float]]:
        """Get recent trajectory points."""
        pass
```

### **3. Demo Architecture Separation**

#### **demo/ball/backend/detection_service.py**
```python
class DetectionService:
    """Backend service for ball detection operations."""
    
    def __init__(self):
        self.detectors = {}  # Cache for loaded detectors
    
    def detect_balls(self, video_path: str, model_path: str, **kwargs) -> Dict:
        """Core ball detection business logic."""
        pass
    
    def get_available_models(self) -> List[str]:
        """Get list of available model files."""
        pass
    
    def validate_inputs(self, video_path: str, model_path: str) -> bool:
        """Validate input parameters."""
        pass
```

#### **demo/ball/frontend/components/detection_tab.py**
```python
def create_detection_tab(detection_service: DetectionService) -> gr.TabItem:
    """Create detection tab component."""
    with gr.TabItem("🔍 Ball Detection"):
        # UI components for detection
        video_input = gr.File(label="Upload Video")
        model_dropdown = gr.Dropdown(label="Model Path")
        detect_btn = gr.Button("Detect Balls")
        
        # Event handlers
        detect_btn.click(
            fn=detection_service.detect_balls,
            inputs=[video_input, model_dropdown],
            outputs=[status_output, results_output]
        )
        
        return video_input, model_dropdown, detect_btn
```

---

## 📝 Implementation Plan & TODO

### **Phase 1: Foundation & Base Classes** 🟡
**Priority: High | Estimated Time: 1-2 weeks**

- [ ] **1.1 Create predictor/base/ structure**
  - [ ] Create `src/predictor/base/detector.py` with `BaseBallDetector`
  - [ ] Create `src/predictor/base/preprocessor.py` with `BasePreprocessor`
  - [ ] Create `src/predictor/base/postprocessor.py` with `BasePostprocessor`
  - [ ] Create `src/predictor/base/tracker.py` with `BaseTracker`
  - [ ] Add comprehensive docstrings and type hints

- [ ] **1.2 Extract and refactor existing detector classes**
  - [ ] Move `LiteTrackNetDetector` to `src/predictor/ball/lite_tracknet_detector.py`
  - [ ] Move `WASBSBDTDetector` to `src/predictor/ball/wasb_sbdt_detector.py`
  - [ ] Implement proper inheritance from base classes
  - [ ] Add unit tests for each detector class

- [ ] **1.3 Create factory pattern**
  - [ ] Implement `src/predictor/ball/factory.py` with detector factory
  - [ ] Support auto-detection of model types
  - [ ] Add configuration-based detector creation

### **Phase 2: Enhanced Third-Party API** 🟡
**Priority: High | Estimated Time: 1 week**

- [ ] **2.1 Create unified WASB-SBDT API**
  - [ ] Implement `third_party/WASB_SBDT/src/api/detector.py`
  - [ ] Add `UnifiedWASBDetector` class with backward compatibility
  - [ ] Implement `process_frames()` wrapper for existing API
  - [ ] Add batch processing support with `process_frames_batch()`

- [ ] **2.2 Expose preprocessing/postprocessing steps**
  - [ ] Add `preprocess_frames()` method for custom pipelines
  - [ ] Add `postprocess_predictions()` method for custom pipelines
  - [ ] Document API usage patterns

- [ ] **2.3 Enhanced tracking API**
  - [ ] Create `third_party/WASB_SBDT/src/api/tracking.py`
  - [ ] Implement `TrackingAPI` class for downstream processing
  - [ ] Add trajectory management functions

- [ ] **2.4 Update main exports**
  - [ ] Update `third_party/WASB_SBDT/src/__init__.py` with new API
  - [ ] Ensure backward compatibility with existing imports
  - [ ] Add deprecation warnings for old API usage

### **Phase 3: Visualization System** 🟢
**Priority: Medium | Estimated Time: 1 week**

- [ ] **3.1 Extract visualization components**
  - [ ] Create `src/predictor/visualization/overlay.py`
  - [ ] Move `visualize_detections_on_video()` to `VideoOverlay` class
  - [ ] Move `create_overlay_video()` to overlay module

- [ ] **3.2 Create reusable visualization components**
  - [ ] Implement `src/predictor/visualization/renderer.py` for drawing utilities
  - [ ] Create `src/predictor/visualization/config.py` for visualization settings
  - [ ] Add trajectory rendering capabilities

- [ ] **3.3 Add batch processing support**
  - [ ] Implement `src/predictor/batch/processor.py`
  - [ ] Add efficient batch inference for multiple videos
  - [ ] Add streaming processing capabilities

### **Phase 4: Demo Refactoring** 🟢
**Priority: Medium | Estimated Time: 1-2 weeks**

- [ ] **4.1 Create backend services**
  - [ ] Implement `demo/ball/backend/detection_service.py`
  - [ ] Implement `demo/ball/backend/overlay_service.py`
  - [ ] Add configuration management in `demo/ball/backend/config.py`

- [ ] **4.2 Create frontend components**
  - [ ] Create `demo/ball/frontend/components/detection_tab.py`
  - [ ] Create `demo/ball/frontend/components/overlay_tab.py`
  - [ ] Create `demo/ball/frontend/components/common.py` for shared UI components

- [ ] **4.3 Implement main application**
  - [ ] Create `demo/ball/frontend/gradio_app.py` with clean UI logic
  - [ ] Create `demo/ball/main.py` as entry point
  - [ ] Add proper error handling and logging

### **Phase 5: Integration & Testing** 🔵
**Priority: Medium | Estimated Time: 1 week**

- [ ] **5.1 Update existing code**
  - [ ] Update `src/ball/api.py` to use new predictor system
  - [ ] Update imports throughout the codebase
  - [ ] Ensure backward compatibility where needed

- [ ] **5.2 Add comprehensive testing**
  - [ ] Add unit tests for all base classes
  - [ ] Add integration tests for detector implementations
  - [ ] Add end-to-end tests for demo functionality

- [ ] **5.3 Documentation updates**
  - [ ] Update README files for new structure
  - [ ] Add API documentation for new modules
  - [ ] Create migration guide for existing users

### **Phase 6: Optimization & Polish** 🔵
**Priority: Low | Estimated Time: 1 week**

- [ ] **6.1 Performance optimization**
  - [ ] Profile batch processing performance
  - [ ] Optimize memory usage for large videos
  - [ ] Add caching for frequently used models

- [ ] **6.2 Advanced features**
  - [ ] Add support for multiple ball tracking
  - [ ] Implement real-time processing capabilities
  - [ ] Add export capabilities (JSON, CSV, annotations)

- [ ] **6.3 Final cleanup**
  - [ ] Remove deprecated code
  - [ ] Clean up unused imports and files
  - [ ] Add final documentation review

---

## 🎯 Success Metrics

### **Code Quality**
- [ ] **Separation of Concerns**: Each module has single, clear responsibility
- [ ] **Test Coverage**: >90% unit test coverage for new modules
- [ ] **Documentation**: Complete API documentation for all public interfaces
- [ ] **Type Safety**: Full type hints throughout the codebase

### **Functionality**
- [ ] **Backward Compatibility**: Existing code continues to work without changes
- [ ] **API Consistency**: Unified interface across all detection methods
- [ ] **Performance**: No regression in detection speed or accuracy
- [ ] **Reusability**: Components can be easily used independently

### **User Experience**
- [ ] **Demo Usability**: Improved demo interface with clear separation
- [ ] **Developer Experience**: Easy to add new detectors and extend functionality
- [ ] **Error Handling**: Clear error messages and graceful failure handling

---

## 🔄 Migration Strategy

### **Backward Compatibility**
1. **Deprecation Warnings**: Add warnings for old API usage
2. **Legacy Wrapper**: Maintain old imports with proxy classes
3. **Documentation**: Clear migration guide for existing users
4. **Gradual Migration**: Allow incremental adoption of new architecture

### **Testing Strategy**
1. **Parallel Testing**: Run both old and new implementations during migration
2. **Integration Tests**: Ensure new architecture works with existing systems
3. **Performance Benchmarks**: Verify no performance regressions
4. **User Acceptance Testing**: Validate improved developer experience

---

## 📋 Risk Assessment

### **Technical Risks**
- **🟡 API Breaking Changes**: Mitigation through careful backward compatibility
- **🟡 Performance Regression**: Mitigation through comprehensive benchmarking
- **🟢 Integration Complexity**: Mitigation through incremental migration

### **Timeline Risks**
- **🟡 Scope Creep**: Mitigation through strict phase-based development
- **🟢 Dependency Changes**: Mitigation through careful dependency management

---

## ✅ Definition of Done

A phase is considered complete when:
1. **All tasks** in the phase are implemented and tested
2. **Documentation** is updated for new components
3. **Backward compatibility** is maintained (where applicable)
4. **Code review** is completed by team members
5. **Integration tests** pass for affected components
6. **Performance benchmarks** show no significant regression

---

*This document serves as the master plan for the ball detection system refactoring. Each phase should be treated as a separate milestone with clear deliverables and success criteria.* 