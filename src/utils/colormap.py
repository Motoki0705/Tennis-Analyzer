import cv2


def get_colormap_code(name: str) -> int:
    if hasattr(cv2, f"COLORMAP_{name.upper()}"):
        return getattr(cv2, f"COLORMAP_{name.upper()}")
    raise ValueError(f"Invalid colormap name: {name}")