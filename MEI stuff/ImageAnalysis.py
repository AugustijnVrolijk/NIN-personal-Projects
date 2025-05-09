import os
import cv2
import numpy as np
from skimage.util import view_as_windows
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_saliency_map(img_gray):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliency_map) = saliency.computeSaliency(img_gray)
    return saliency_map.astype(np.float32)

def compute_entropy_map(img_gray, patch_size=32):
    H, W = img_gray.shape
    entropy_map = np.zeros_like(img_gray, dtype=np.float32)

    windows = view_as_windows(img_gray, (patch_size, patch_size), step=patch_size)
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            patch = windows[i, j]
            entropy = shannon_entropy(patch)
            entropy_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = entropy
    return entropy_map

def compute_edge_grid(img_gray, grid_size=(18, 32)):
    edges = sobel(img_gray)
    h, w = edges.shape
    gh, gw = grid_size
    cell_h, cell_w = h // gh, w // gw
    grid = np.zeros((gh, gw))

    for i in range(gh):
        for j in range(gw):
            patch = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            grid[i, j] = patch.mean()
    return grid

def load_and_resize_image(image_path, **kwargs):
    # Open image using PIL for robustness (handles JPEG, PNG, BMP, etc.)
    max_size = kwargs.pop("max_size", 512)

    img = Image.open(image_path).convert("L")  # Convert to grayscale directly
    # Compute scale to maintain aspect ratio
    ratio = max_size / max(img.size)
    new_size = tuple([int(x * ratio) for x in img.size])
    img_resized = img.resize(new_size, Image.BILINEAR)

    # Optional: pad to make it square
    new_img = Image.new("L", (max_size, max_size), color=0)
    upper_left = ((max_size - new_size[0]) // 2, (max_size - new_size[1]) // 2)
    new_img.paste(img_resized, upper_left)

    return np.array(new_img)

def analyze_image(image_path, save=False, **kwargs):
    image_path = Path(image_path)

    if not image_path.suffix.lower() in ('.jpg', '.png', '.jpeg', '.bmp'):
        raise ValueError("Unsupported file format")

    image_id = image_path.stem  # e.g., '0003'
    img_gray = load_and_resize_image(image_path, **kwargs)

    saliency_map = compute_saliency_map(img_gray)
    entropy_map = compute_entropy_map(img_gray, **kwargs)
    edge_grid = compute_edge_grid(img_gray)

    # Save results
    if save:
        output_dir = Path("bias_results")
        output_dir.mkdir(exist_ok=True)
    
        plt.imsave(output_dir / f"saliency_{image_id}.png", saliency_map, cmap='hot')
        plt.imsave(output_dir / f"entropy_{image_id}.png", entropy_map, cmap='hot')

        plt.figure()
        plt.title(f"Edge Grid (3x3) - Image {image_id}")
        plt.imshow(edge_grid, cmap='hot')
        plt.colorbar()
        plt.savefig(output_dir / f"edge_grid_{image_id}.png")
        plt.close()

        print(f"Analysis complete for image {image_id}. Results saved in '{output_dir}/'.")
        
    return saliency_map, entropy_map, edge_grid


def analyze_image_folder(folder_path):
    saliency_sum = None
    entropy_sum = None
    edge_grid_sum = None
    count = 0

    for filename in tqdm(os.listdir(folder_path)):
        try:
            path = os.path.join(folder_path, filename)
            saliency_map, entropy_map, edge_grid = analyze_image(path)

            if saliency_sum is None:
                saliency_sum = saliency_map
                entropy_sum = entropy_map
                edge_grid_sum = edge_grid
            else:
                saliency_sum += saliency_map
                entropy_sum += entropy_map
                edge_grid_sum += edge_grid

            count += 1
        except:
            print(f"failed to process {filename}")
            continue

    # Averages
    saliency_avg = saliency_sum / count
    entropy_avg = entropy_sum / count
    edge_grid_avg = edge_grid_sum / count

    # Save results
    os.makedirs("bias_results", exist_ok=True)
    plt.imsave("bias_results/saliency_average.png", saliency_avg, cmap='hot')
    plt.imsave("bias_results/entropy_average.png", entropy_avg, cmap='hot')

    plt.figure()
    plt.title("Mean Edge Grid (3x3)")
    plt.imshow(edge_grid_avg, cmap='hot')
    plt.colorbar()
    plt.savefig("bias_results/edge_grid_average.png")

    print("Analysis complete. Results saved in 'bias_results/'.")

# Example usage
if __name__ == "__main__":
    folder = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images"
    analyze_image_folder(folder)
    analyze_image(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images\0001.bmp", save=True)