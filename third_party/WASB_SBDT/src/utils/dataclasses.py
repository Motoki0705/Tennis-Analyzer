from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import numpy as np

@dataclass(frozen=True)
class Center:
    is_visible: bool # visibility
    x: float # x coordinate
    y: float # y coordinate
    r: Optional[float] = field(default=-1) # radius
    @property
    def diameter(self) -> float:
        if self.r is None or self.r < 0:
            raise ValueError('r is not defined')
        return self.r * 2
    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass(frozen=True)
class Keypoint:
    x: float # x coordinates of center 
    y: float # y coordinates of center
    r: float # radius 
    @property
    def diameter(self) -> float:
        return self.r * 2

