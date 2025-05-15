import PIL.Image
import numpy as np
import os
import PIL

from typing import Callable, Any
from pathlib import Path
from PIL import Image
from functools import wraps
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

class npImage():
    def __init__(self, image:Any, **kwargs):
        self.arr = None
        self.img = None
        self.name = None

        if isinstance(image, np.ndarray) :
            self._init_from_Arr(image)
            
        elif isinstance(image, PIL.Image.Image):
            self._init_from_Image(image, **kwargs)

        elif isinstance(image, (str, Path)):   
            self._init_from_Path(image, **kwargs)
        
        else:
            raise NotImplementedError(f"unable to instantiate image of type: {type(image)}")

    def _init_from_Arr(self, arr:np.ndarray):
        self.arr = arr.astype(np.float64)
        corArr =  np.clip(arr, 0, 255).astype(np.uint8)
        self.img = Image.fromarray(corArr, mode='L')

    def _init_from_Image(self, image:PIL.Image.Image, enforce=True):

        if enforce and not npImage._is_grayscale(image):
            raise TypeError("Given image is not black and white")
        
        self.img = image
        self.arr = np.array(image).astype(np.float64)

    def _init_from_Path(self, path:str, **kwargs):
        corPath = Path(path)
        if not corPath.is_file():
            raise ValueError("given path is not a correct path")

        img = [".png", ".bmp"]
        npy = [".npy"]

        if corPath.suffix in img:
            img = Image.open(corPath).convert('L')  # Convert to 8-bit grayscale
            self._init_from_Image(img, **kwargs)

        elif corPath.suffix in npy:
            arr = np.load(corPath)
            self._init_from_Arr(arr)

        self.name = corPath.stem

    def save(self, destPath:str, extension:str|list=[], defaultExt:str=".png", **kwargs):
        """
        either the full file name, or a folder to which to 
        save to with the same name as it was loaded in with
        """
        allowedExtension = [".npy", ".png"]
        if not defaultExt in allowedExtension:
            raise ValueError(f"default extension {defaultExt} is not supported")
        
        corPath = Path(destPath)

        #get the correct extensions
        if corPath.suffix:
            extension.append(corPath.suffix)
        if not extension:
            extension = [defaultExt]

        if isinstance(extension, str):
            extension = [extension]
        #get the path name if a folder is given
        if corPath.is_dir():
            if self.name:
                corPath = corPath / self.name
            else:
                raise FileNotFoundError(f"no name is recorded for {self}, so cannot make a filepath to save with a folder path input {corPath}")

        #save for each extension
        for ext in extension:
            cor_ext = ext.lower().strip()
            if not cor_ext in allowedExtension:
                raise ValueError(f"unrecognised extension {cor_ext}")
            final_path = self._checkPath(corPath, cor_ext)
            self._raw_save(final_path)
        
    def _raw_save(self, path:Path):
        #saving, only .npy and .png supported at the moment

        if path.suffix == ".npy":
            np.save(path, self.arr)
        else:
            self.img.save(path, format='PNG', optimize=True)

    @staticmethod
    def _is_grayscale(img:PIL.Image.Image):
        """
        Checks whether an image is grayscale.
        Accepts: Path to the image
        Returns: True if grayscale, False otherwise
        """
        
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

    def linearBoostContrast(self, matrix:np.ndarray = None, format_max:int=255, save=True, **kwargs) -> np.ndarray:
        """
        linearly stretches matrix values so that the maximum becomes 255.
        Output is returned as uint8 for PNG saving.
        """
        if not isinstance(matrix, np.ndarray):
            matrix = self.arr

        max_val = np.max(matrix)
        if max_val == 0:
            return matrix # avoid divide-by-zero
        
        contrast_stretched = matrix * (format_max / max_val)

        if save:
            self._init_from_Arr(contrast_stretched)
            
        return contrast_stretched
    
    def gamma_correction(self, gamma:float, linearBoost=True, **kwargs) -> np.ndarray:
        curMax = np.max(self.arr)
        gamma_raw = np.power(self.arr, gamma)

        if linearBoost:
            format_max = 255
        else:
            format_max = curMax

        return self.linearBoostContrast(matrix=gamma_raw, format_max=format_max, **kwargs)

    def blur(self, sigma:float, save=True, **kwargs):
        blurred = gaussian_filter(self.arr, sigma=sigma)
        
        if save:
            self._init_from_Arr(blurred)
            
        return blurred
    
    def resize(self, size:float|tuple, save=True, **kwargs):
        
        if isinstance(size, tuple):
            assert len(size) == 2
        else:
            w, h = self.img.size
            size = (int(w*size), int(h*size))

        resample = kwargs.pop("resample", None)
        resized_img = self.img.resize(size, resample=resample)
        resized_arr = np.array(resized_img).astype(np.float64)
        if save:
            self._init_from_Image(resized_img)

        return resized_arr

    def applyWeight(self, weight, save=True):
        t_r = self.arr * weight
        if save:
            self._init_from_Arr(t_r)
        return t_r

    @staticmethod
    def _normalise(arr:np.ndarray, newMax:int=1) -> np.ndarray:
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_array = (arr - min_val) / (max_val - min_val)
        return normalized_array*newMax
    
    @staticmethod
    def _checkPath(path:str|Path, corSuffix:str):
        return Path(os.path.splitext(path)[0] + corSuffix)

def saveImg(img:Any, path:str, **kwargs):
    tempImg = npImage(img, **kwargs)
    tempImg.save(path, **kwargs)
    return

def expand_folder_path(func):
    @wraps(func)
    def wrapper(paths, *args, **kwargs):
        if isinstance(paths, (str, Path)) and Path(paths).is_dir():
            folder = Path(paths)
            # Extract all file paths (ignoring subdirectories)
            paths = [f for f in folder.iterdir() if f.is_file() and f.name != "Thumbs.db"]
            #windows adds hidden "Thumbs.db" to folders full of images
        return func(paths, *args, **kwargs)
    return wrapper

@expand_folder_path
def iterFolderFun(paths:list[str|Path|np.ndarray|PIL.Image.Image], 
                  funcs: list[Callable[..., Any]] = None, 
                  args_list: list[dict] = None, 
                  reduction: Callable[..., Any] = None,
                  collect: bool = False,
                  init_val: Any = None,
                  verbose: bool = False,
                  **kwargs):

    if funcs is not None:
        if not (len(funcs) == len(args_list)):
            raise ValueError("Each function must have a corresponding kwargs dictionary.")

    if reduction is not None and collect == True:
        raise ValueError("Reduction and Collect cannot both be had. If a reduction function is specified the final reduction output is returned")

    results = init_val
    if collect == True:
        results = [0] * len(paths)

    for i, path in enumerate(tqdm(paths)):
        
        img_obj = npImage(path)
        res = None

        for func, kwargs in zip(funcs, args_list):
            kwargs["iter"] = i
            res = func(img_obj, **kwargs)
    
        if collect:
            results[i] = res
        elif reduction is not None:
            results = reduction(results, res)

    return results

@expand_folder_path
def weighted_average_images(paths: list[str], weights: list[float], blur_contrast=False, **kwargs) -> np.ndarray:
    """
    Loads image matrices from .npy files, applies weights, sums them,
    and normalizes by the number of images.
    """
    verbose = kwargs.get("verbose", True)

    if len(paths) != len(weights):
        raise ValueError(f"Lists of file paths and weights must be the same length\nlength of paths: {len(paths)}\n len of weights: {len(weights)}")
    elif len(paths) <= 0:
        return None

    def applyWeight(image:npImage, r_weights:list[float], iter:int, **kwargs):
        return image.applyWeight(r_weights[iter], **kwargs)

    if blur_contrast:
        sigma = kwargs.get("sigma", 8)
        gamma = kwargs.get("gamma", 1.2)
        funcs = [npImage.blur, npImage.gamma_correction, applyWeight]
        args = [{"sigma":sigma}, {"gamma":gamma, "save":False}, {"r_weights":weights, "save":False}]
    else:
        funcs = [applyWeight]
        args = [{"r_weights":weights, "save":False}]

    init_val = np.zeros(1)
    def sumMatrix(n1:np.ndarray, n2:np.ndarray):
        return n1 + n2

    total_matrix = iterFolderFun(paths, funcs, args, reduction=sumMatrix,init_val=init_val,verbose=verbose)
    total_matrix /= sum(weights)
    return total_matrix

@expand_folder_path
def resize_and_save(paths:list[str], resize_factor:float, destFolder:str, extensions:list[str]):
    funcs = [npImage.resize, npImage.save]
    args = [{"size":resize_factor},{"destPath":destFolder, "extension":extensions}]
    iterFolderFun(paths, funcs, args, verbose=True)
    return

def main():
    bmp = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\Muckli4000Images")
    res = Path(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgResults")
    

    dest = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgSmall"
    destNPY = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\ImagesSmallNPY"

    #resize_and_save(bmp, 0.2, dest, ".png")
    resize_and_save(bmp, 0.2, destNPY, ".npy")

    return

if __name__ == "__main__":
    main()