import numpy as np
import torch

def generate_heatmap(label, output_size=(512, 512), sigma=3):
    scale_x = output_size[1] / 1280.0
    scale_y = output_size[0] / 720.0
    new_x = label["x"] * scale_x
    new_y = label["y"] * scale_y

    xs = np.arange(output_size[1])
    ys = np.arange(output_size[0])
    xs, ys = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xs - new_x) ** 2 + (ys - new_y) ** 2) / (2 * sigma ** 2))
        
    return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

if __name__ == '__main__':
    label = {'x': 100, 'y': 100}
    sample_heatmap = generate_heatmap(label)
    print(sample_heatmap.max(), sample_heatmap.min())