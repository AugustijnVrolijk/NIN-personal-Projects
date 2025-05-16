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
from imageComp import expand_folder_path, npImage

def compute_saliency_map(img_gray):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliency_map) = saliency.computeSaliency(img_gray)
    return saliency_map.astype(np.float32)

def smallest_common_divisor_above_threshold(a, b, t):
    # Find all common divisors
    common_divisors = [i for i in range(1, min(a, b) + 1) if a % i == 0 and b % i == 0]
    
    # Filter those greater than the threshold
    valid_divisors = [d for d in common_divisors if d > t]
    
    # Return the smallest one above the threshold, or None if none exists
    return min(valid_divisors) if valid_divisors else None

def compute_entropy_map(img_gray, patch_size=16):
    height, width = img_gray.shape
    patch_size = smallest_common_divisor_above_threshold(height, width, patch_size)
    if patch_size is None or patch_size > 50:
        raise ValueError("bad patch_size")
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

def load_and_resize_image(image_path, resize = True,**kwargs):
    # Open image using PIL for robustness (handles JPEG, PNG, BMP, etc.)
    max_size = kwargs.pop("max_size", 512)

    img = Image.open(image_path).convert("L")  # Convert to grayscale directly
    # Compute scale to maintain aspect ratio
    if resize:
        print(resize)
        ratio = max_size / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])
        img_resized = img.resize(new_size, Image.BILINEAR)

        # Optional: pad to make it square
        new_img = Image.new("L", (max_size, max_size), color=0)
        upper_left = ((max_size - new_size[0]) // 2, (max_size - new_size[1]) // 2)
        new_img.paste(img_resized, upper_left)
        img = new_img

    return np.array(img)

def analyze_image(image_path, save=False, **kwargs):
    image_path = Path(image_path)

    if not image_path.suffix.lower() in ('.jpg', '.png', '.jpeg', '.bmp'):
        raise ValueError("Unsupported file format")

    image_id = image_path.stem  # e.g., '0003'
    resize = kwargs.pop("resize", True)
    img_gray = load_and_resize_image(image_path, resize)

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

def checkName(filename:str|Path, name_skip:str|None):
    if not isinstance(name_skip, str):
        return False
    
    p = Path(filename)
    name = p.name.lower()
    return (name_skip.lower() in name)

@expand_folder_path
def analyze_image_folder(folder_path, save_dir:str|Path, label:str="",name_skip:str=None,**kwargs):
    saliency_sum = None
    entropy_sum = None
    edge_grid_sum = None
    count = 0

    for filename in tqdm(folder_path):
        if checkName(filename, name_skip):
            continue
        try:
            saliency_map, entropy_map, edge_grid = analyze_image(filename, **kwargs)

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
    os.makedirs(save_dir, exist_ok=True)
    plt.imsave(os.path.join(save_dir,f"{label}saliency_average.png"), saliency_avg, cmap='hot')
    plt.imsave(os.path.join(save_dir,f"{label}entropy_average.png"), entropy_avg, cmap='hot')
    """
    plt.imsave(os.path.join(save_dir,f"{label}edge_grid_average.png"), edge_grid_avg, cmap='hot')
    """
    plt.figure()
    plt.title("Mean Edge Grid (3x3)")
    plt.imshow(edge_grid_avg, cmap='hot')
    plt.savefig(os.path.join(save_dir,f"{label}edge_grid_average.png"))
    
    print(f"Analysis complete. Results saved in {save_dir}'.")

# Example usage
if __name__ == "__main__":
    folder = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images"
    both = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType\both"
    nOcc = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType\notOccluded"
    occ = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType\occluded"
    dest = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\RFbyResponseType\analysis"
    analyze_image_folder(both, dest, label="both_", name_skip="true", resize=False)
    analyze_image_folder(nOcc, dest, label="notOccluded_", name_skip="true", resize=False)
    analyze_image_folder(occ, dest, label="occluded_", name_skip="true", resize=False)

    #analyze_image(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images\0460.bmp", resize=False, save=True)