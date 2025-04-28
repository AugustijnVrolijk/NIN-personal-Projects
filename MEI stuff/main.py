from PIL import Image
import numpy as np
import os
from typing import Callable, Any
from pathlib import Path

def is_grayscale(img_path):
    """
    Checks whether an image is grayscale.
    Accepts: Path to the image
    Returns: True if grayscale, False otherwise
    """
    img = Image.open(img_path)
    
    if img.mode == 'L':
        return True  # Already grayscale
    
    elif img.mode == 'RGB':
        arr = np.array(img)
        # Check if R == G == B for all pixels
        if np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 1] == arr[..., 2]):
            return True  # All color channels are equal => grayscale
        else:
            return False
    
    return False  # Other modes (e.g. RGBA, P) not supported here
def bmp_to_png(bmp_path:str, png_path:str=None, enforce=True):
    """
    Converts a .bmp file to a losslessly compressed .png file.
    """
    if png_path is None:
        png_path = bmp_path

    dest_path = checkPath(png_path, ".png")

    if enforce:
        if not is_grayscale(bmp_path):
            ValueError(f"{bmp_path} is not grayscale")

    img = Image.open(bmp_path).convert('L')  # Convert to 8-bit grayscale
    img.save(dest_path, format='PNG', optimize=True)
    return

def bmp_to_npy(bmp_path:str, npy_path:str=None, enforce=True):
    """
    Converts a .bmp file to a losslessly compressed .png file.
    """
    if npy_path is None:
        npy_path = bmp_path

    dest_path = checkPath(npy_path, ".npy")

    if enforce:
        if not is_grayscale(bmp_path):
            ValueError(f"{bmp_path} is not grayscale")

    img = Image.open(bmp_path).convert('L')
    arr = np.array(img)
    np.save(dest_path, arr)
    return

def png_to_npy(png_path:str, npy_path:str=None):
    """
    Converts a .png file to a .npy (NumPy array) file.
    """
    if npy_path is None:
        npy_path = png_path

    dest_path = checkPath(npy_path, ".npy")

    img = Image.open(png_path).convert('L')
    arr = np.array(img)
    np.save(dest_path, arr)
    return

def npy_to_png(npy_path:str=None, matrix:np.ndarray=None, png_path:str=None):
    """
    Converts a .npy (NumPy array) file back to a .png image file.
    """
    if npy_path == None and matrix is None:
        print("no path or matrix was given to npy_to_png")
        return
    elif npy_path != None and isinstance(matrix, np.ndarray):
        print("Cannot give both a path and matrix to npy_to_png")
        return
    
    if npy_path:
        if png_path is None:
            png_path = npy_path
        arr = np.load(npy_path)
    else:
        if png_path is None:
            raise ValueError("if matrix is given png_path must be provided")
        arr = matrix

    dest_path = checkPath(png_path, ".png")

    # Ensure array is 8-bit unsigned int
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(arr, mode='L')
    img.save(dest_path, format='PNG', optimize=True)
    return

def checkPath(path, corSuffix):
    return os.path.splitext(path)[0] + corSuffix

def weighted_average_images(paths: list[str], weights: list[float]) -> np.ndarray:
    """
    Loads image matrices from .npy files, applies weights, sums them,
    and normalizes by the number of images.
    """
    if len(paths) != len(weights):
        raise ValueError("Lists of file paths and weights must be the same length.")
    elif len(paths) <= 0:
        return None

    total_matrix = np.load(paths[0]).astype(np.float32) # ensure float for math
    total_matrix *= weights[0]
    
    for i in range(1, len(paths)):
        print(f"iteration: {i}")
        matrix = np.load(paths[i]).astype(np.float32)  # ensure float for math
        weighted_matrix = matrix * weights[i]
        total_matrix += weighted_matrix

    # Normalize by number of images
    total_matrix /= len(paths)
    return total_matrix

def boost_contrast(matrix: np.ndarray, format_max:int=255, clip = True) -> np.ndarray:
    """
    Stretches matrix values so that the maximum becomes 255.
    Output is returned as uint8 for PNG saving.
    """
    max_val = np.max(matrix)
    if max_val == 0:
        return matrix # avoid divide-by-zero
    
    contrast_stretched = matrix * (format_max / max_val)
    if clip:
        contrast_stretched = np.clip(contrast_stretched, 0, format_max).astype(np.uint8)
    return contrast_stretched

def iterFolderConvert(func:Callable[..., Any], origin:str, destination:str, verbose:bool= False, **kwargs):

    for filename in os.listdir(origin):
        if verbose:
            print(f"Processing: {filename}")
        file_path = os.path.join(origin, filename)
        dest_path = os.path.join(destination, filename)
        if os.path.isfile(file_path):
            func(file_path, dest_path, **kwargs)
    return

def main():
    bmp = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images")
    png = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgPNG")
    npy = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgNPY")
    res = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgResults")

    bmp2npy = [bmp_to_npy, bmp, npy, True]
    
    #iterFolderConvert(*bmp2npy)

    toMerge =  [file for file in npy.iterdir() if file.is_file()]
    weights = np.ones(shape=len(toMerge))
    testVal = weighted_average_images(toMerge, weights)
    boostedTest = boost_contrast(testVal)
    npy_to_png(matrix=testVal, png_path=os.path.join(res, "test"))
    npy_to_png(matrix=boostedTest, png_path=os.path.join(res, "boostedtest"))
    
    return

if __name__ == "__main__":
    main()