from typing import Any
from omegaconf import DictConfig

from .intra_frame_peak import IntraFramePeakTracker
from .online import OnlineTracker

__tracker_factory = {
    'intra_frame_peak': IntraFramePeakTracker,
    'online': OnlineTracker,
        }

def build_tracker(cfg: DictConfig) -> Any:
    """Build a tracker based on configuration.
    
    Args:
        cfg: Configuration object containing tracker parameters
        
    Returns:
        Tracker instance ready for ball tracking
        
    Raises:
        KeyError: If tracker name is not supported
    """
    tracker_name = cfg['tracker']['name']
    if tracker_name not in __tracker_factory.keys():
        raise KeyError('unknown tracker: {}'.format(tracker_name))
    return __tracker_factory[tracker_name](cfg)

