import pandas as pd
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from imageComp import expand_folder_path
from ImageAnalysis import smallest_common_divisor_above_threshold, compute_entropy_map
from skimage.util import view_as_windows
from skimage.measure import shannon_entropy
from tqdm import tqdm
from createRFmask import masked_mean_calc
from scipy.stats import entropy as scipy_entropy

def relative_entropy(target:np.ndarray, base:np.ndarray) -> float:
    target = target.flatten()
    base = base.flatten()
    bins = np.linspace(0, 255, num=256)  # 256 bins

    # histograms from datasets A and B
    hist_A, _ = np.histogram(target, bins=bins)
    hist_B, _ = np.histogram(base, bins=bins)

    epsilon = 1e-4
    pk = hist_A + epsilon
    qk = hist_B + epsilon

    # Normalize to get proper probability distributions
    pk = pk / pk.sum()
    qk = qk / qk.sum()

    # Compute KL divergence
    kl_div = scipy_entropy(pk, qk, base=2)
    return kl_div

def copy_filtered_neurons(source, target, filtered_neurons_csv):

    good_neurons = pd.read_csv(filtered_neurons_csv)
    extension = ".png"
    total = len(good_neurons)
    for i, row in good_neurons.iterrows():
        print(f"({i}/{total})")
        extension = ".png"
        saveName = f"{row['Mouse']}_{row['mouseNeuron']}{extension}"

        FamiliarNO, FamiliarO, NovelNO, NovelO = row[['respFamiliarNO', 'respFamiliarO', 'respNovelNO', 'respNovelO']]
        keys = {
            "FamiliarNotOccluded":FamiliarNO,
            "FamiliarOccluded":FamiliarO,
            "NovelNotOccluded":NovelNO,
            "NovelOccluded":NovelO,
        }

        for key, val in keys.items():
            if val:
                sourcePath = Path(os.path.join(source, key, saveName))
                assert sourcePath.is_file(), f"source path does not exist for: {sourcePath}, row: {i}"
                destPath = Path(os.path.join(target, key, saveName))
                destPath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(sourcePath, destPath)

def testing_entropy(test_path):
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    patch_size = smallest_common_divisor_above_threshold(h, w, 100)
    print(patch_size)
    if patch_size is None:
        raise ValueError("bad patch size")
    assert img.dtype == "uint8", f"not uint8, is: {img.dtype}"

    windows = view_as_windows(img, (patch_size, patch_size), step=patch_size)
    patch = windows[1,1]
    entropy1 = shannon_entropy(patch)
    patch_as_arr = patch.flatten()
    patch_as_3d = patch.reshape(120,60,2)
    entropy2 = shannon_entropy(patch_as_arr)
    entropy3 = shannon_entropy(patch_as_3d)
    print(f"patch shape: {patch.shape}\nflattened shape: {patch_as_arr.shape}\nas 3d shape: {patch_as_3d.shape}")
    print(f"patch:     {entropy1}\n"
          f"flattened: {entropy2}\n"
          f"as 3d:     {entropy3}")
    assert entropy1 == entropy2 == entropy3, "entropy is not equal"
    plt.imshow(patch)
    plt.show()
    return

def test_image_entropy(img1, img2):
    img1_arr = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2_arr = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    t1 = np.stack((img1_arr, img2_arr), axis=0)
    avg_img = t1.mean(axis=0)
    print(avg_img.shape)

    plt.title("1")
    plt.imshow(img1_arr)
    plt.show()

    plt.title("2")
    plt.imshow(img2_arr)
    plt.show()

    plt.title("avg")
    plt.imshow(avg_img)
    plt.show()

    #test2:
    images = [img1, img2]
    patch_size = 60
    
    all_imgs = []
    for img in images:
        all_imgs.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))

    img_tensor = np.stack(all_imgs, axis=0)
    n, h, w = img_tensor.shape
    assert n == len(images) and h == 1080 and w == 1920

    patch_size = smallest_common_divisor_above_threshold(h, w, patch_size)
    if patch_size is None:
        raise ValueError("bad patch_size")
    
    i_x = 1
    j_x = 1
    assert avg_img.shape == (1080,1920)
    avg_img_patch = avg_img[i_x*patch_size:(i_x+1)*patch_size, j_x*patch_size:(j_x+1)*patch_size]
    plt.title("avg patch")
    plt.imshow(avg_img_patch)
    plt.show()
    
    n_h = h/patch_size
    n_w = w/patch_size
    assert n_h.is_integer() and n_w.is_integer()
    n_h, n_w = int(n_h), int(n_w)

    for i in range(n_h):
        if i != i_x:
            continue
        for j in range(n_w):
            if j != j_x:
                continue
            patch = img_tensor[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            assert np.allclose(patch.mean(axis=0), avg_img_patch), "arrays were not the same"
            print("success")
    return

@expand_folder_path
def folder_entropy_img(images, patch_size):
    all_imgs = []
    for img in images:
        all_imgs.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))

    img_tensor = np.stack(all_imgs, axis=0)
    n, h, w = img_tensor.shape
    assert n == len(images) and h == 1080 and w == 1920

    patch_size = smallest_common_divisor_above_threshold(h, w, patch_size)
    if patch_size is None:
        raise ValueError("bad patch_size")
    

    entropy_map = np.zeros((h, w), dtype=np.float32)
    n_h = h/patch_size
    n_w = w/patch_size
    assert n_h.is_integer() and n_w.is_integer()
    n_h, n_w = int(n_h), int(n_w)

    for i in range(n_h):
        for j in range(n_w):
            patch = img_tensor[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            entropy = shannon_entropy(patch)
            entropy_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = entropy
    
    return entropy_map

@expand_folder_path
def folder_entropy_mask(images, mask:np.ndarray):
    
    if np.array_equal(np.unique(mask), [0, 1]):
        mask = mask.astype(bool)
    elif not np.array_equal(np.unique(mask), [False, True]):
        raise ValueError("Mask must be binary only")
    
    first = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    assert first.shape == mask.shape, f"img {images[0]} is not the same shape as the provided mask"
    on_mask_area = mask.sum()

    mask = mask.flatten()
    first = first.flatten()
    vals = first[mask]
    assert len(vals) == on_mask_area, "error getting masked values"
    
    for i in tqdm(range(1, len(images))):
        img = images[i]
        new_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE).flatten()
        assert new_img.shape == mask.shape, f"img {img} is not the same shape as the provided mask"
        t_vals = new_img[mask]
        vals = np.concatenate((vals, t_vals))

    assert len(vals) == on_mask_area*len(images), "error getting masked values, not all images were processed"
    
    entropy = shannon_entropy(vals)
    return entropy

@expand_folder_path
def compare_old_and_new(images, patch_size, save_dir):
    new = folder_entropy_img(images, patch_size)
    
    plt.title("new")
    plt.imshow(new)
    plt.show()
    old = None

    for img in images:
        img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        entropy_map = compute_entropy_map(img_gray, patch_size)
        if old is None:
            old = entropy_map
        else:
            old += entropy_map
    old /= len(images)
    plt.title("old")
    plt.imshow(old)
    plt.show()

    os.makedirs(save_dir, exist_ok=True)
    plt.imsave(os.path.join(save_dir,f"entropy_new.png"), new, cmap='hot')
    plt.imsave(os.path.join(save_dir,f"entropy_old.png"), old, cmap='hot')

def run_new_entropy(orig_dir, conditions, patch_size, save_dir, saveAsRaw=False):

    entropy_dict = {}
    for cond in conditions:
        entropy = folder_entropy_img(os.path.join(orig_dir, cond), patch_size)
        entropy_dict[cond] = entropy
        
    gmax = max(np.max(img) for img in entropy_dict.values())
    gmin = min(np.min(img) for img in entropy_dict.values())

    r_paths = []
    for cond, entropy in entropy_dict.items():
        im = plt.imshow(entropy, cmap='viridis', vmin=gmin, vmax=gmax)
        plt.colorbar(im, label="Entropy")  # Adds the legend
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,f"entropy_{cond}.png"), dpi=300)
        plt.close()
        if saveAsRaw:
            save_path = os.path.join(save_dir,f"entropy_{cond}_raw.png")
            r_paths.append(save_path)
            np.save(save_path, entropy)
    return r_paths   
    
def mean_cond_entropy(r_paths, cond, mask):
    
    if np.array_equal(np.unique(mask), [0, 1]):
        mask = mask.astype(bool)
    elif not np.array_equal(np.unique(mask), [False, True]):
        raise ValueError("Mask must be binary only")
    
    cor_path = None
    for p in r_paths:
        if cond in p:
            cor_path = p
    
    first:np.ndarray = np.load(cor_path)
    assert first.shape == mask.shape, f"img {cor_path} is not the same shape as the provided mask"
    on_mask_area = mask.sum()

    mask = mask.flatten()
    first = first.flatten()
    vals = first[mask]
    assert len(vals) == on_mask_area, "error getting masked values"
    
    return vals

def run_mask_entropy(orig_dir, conditions, mask_path, save_dir, r_paths):
    mask = np.load(mask_path).astype(bool)
    rhs = np.zeros_like(mask)
    rhs[:, 1080:] = 1
    msg = ""

    for cond in conditions:
        folder_path = os.path.join(orig_dir, cond)
        in_mask = folder_entropy_mask(folder_path, mask)
        out_mask = folder_entropy_mask(folder_path, ~mask)
        rhs_mask = folder_entropy_mask(folder_path, rhs)
        new_txt = (f"condition {cond}:\n"
                  f"  Pooled pixel entropy:\n\n"
                  f"    population RF:   {in_mask:.3f}\n"     
                  f"    everything else: {out_mask:.3f}\n"      
                  f"    right-hand side: {rhs_mask:.3f}\n\n")
        
        msg += new_txt      

        in_mask = mean_cond_entropy(r_paths, cond, mask)
        out_mask = mean_cond_entropy(r_paths, cond, ~mask)
        rhs_mask = mean_cond_entropy(r_paths, cond, rhs)
        new_txt = (f"  Mean spatial-chunk entropy:\n\n"
                  f"    population RF:   {in_mask.mean():.3f} +/- {in_mask.std():.3f}\n"     
                  f"    everything else: {out_mask.mean():.3f} +/- {out_mask.std():.3f}\n"      
                  f"    right-hand side: {rhs_mask.mean():.3f} +/- {rhs_mask.std():.3f}\n\n")
        
        msg += new_txt

        in_mask, _ = masked_mean_calc(folder_path, mask)
        out_mask, _ = masked_mean_calc(folder_path, ~mask)
        rhs_mask, _ = masked_mean_calc(folder_path, rhs)
        new_txt = (f"  Mean pixel intensity:\n\n"
                  f"    population RF:   {in_mask.mean():.3f} +/- {in_mask.std():.3f}\n"     
                  f"    everything else: {out_mask.mean():.3f} +/- {out_mask.std():.3f}\n"      
                  f"    right-hand side: {rhs_mask.mean():.3f} +/- {rhs_mask.std():.3f}\n\n\n")
        
        msg += new_txt    

    savePath = os.path.join(save_dir, f"results.txt")
    with open(savePath, "w") as file:
        file.write(msg)

    return

def images_entropy():
    t_p = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\FamiliarOccluded\Anton_582.png"
    t_p_2 = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\FamiliarOccluded\Feyenoord_462.png"
    #test_image_entropy(t_p, t_p_2)
    orig_dir = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant"
    conditions = ["FamiliarNotOccluded","FamiliarOccluded","NovelNotOccluded","NovelOccluded", "randomRFs"]
    patch_size = 30
    save_dir = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\entropy"
    #r_paths = run_new_entropy(orig_dir, conditions, patch_size, save_dir, saveAsRaw=True)
    r_paths = [r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\entropy\entropy_FamiliarNotOccluded_raw.png.npy",
               r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\entropy\entropy_FamiliarOccluded_raw.png.npy",
               r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\entropy\entropy_NovelNotOccluded_raw.png.npy",
               r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\entropy\entropy_NovelOccluded_raw.png.npy"]
    mask_path = r"C:\Users\augus\NIN_Stuff\data\koenData\newRFQuant\mask.npy"
    #run_mask_entropy(orig_dir, conditions, mask_path, save_dir, r_paths)
    run_new_entropy(orig_dir, ["randomRFs"], patch_size, save_dir)

if __name__ == "__main__":
    images_entropy()