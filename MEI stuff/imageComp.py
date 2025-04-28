import numpy as np
import os
import PIL

from typing import Callable, Any
from pathlib import Path
from PIL import Image
from matLoader import loadMat
from scipy.ndimage import gaussian_filter

class npImage():
    def __init__(self, image:Any, **kwargs):
        
        if isinstance(image, np.ndarray) :
            self.__init_from_Arr(image)
            
        elif isinstance(image, PIL.Image.Image):
            self._init_from_Image(image, **kwargs)

        elif isinstance(image, str):   
            self._init_from_Path(image, **kwargs)
    
    def __init_from_Arr(self, arr:np.ndarray):
        self.arr = arr
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
            self.__init_from_Arr(arr)

    def save(self, destPath:str, extension:str|list=None):
        allowedExtension = [".npy", ".png", None]

        corPath = Path(destPath)
        if extension:
            if isinstance(extension, list):
                for ext in extension:
                    newPath = f"{destPath}_{ext[1:]}"
                    self.save(newPath, ext)
                return
            extension = extension.lower().strip()
            if not extension in allowedExtension:
                raise ValueError("unrecognised extension")
            corPath = self._checkPath(corPath, extension)
        else:
            if not corPath.suffix in allowedExtension:
                raise ValueError("unrecognised extension")
        
        if corPath.suffix == ".npy":
            np.save(corPath, self.arr)
        else:
            self.img.save(corPath, format='PNG', optimize=True)

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
        if matrix == None:
            matrix = self.arr

        max_val = np.max(matrix)
        if max_val == 0:
            return matrix # avoid divide-by-zero
        
        contrast_stretched = matrix * (format_max / max_val)

        if save:
            self.__init_from_Arr(contrast_stretched)
            
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
        blurred = gaussian_filter(self.arr, sigma=sigma, **kwargs)
        
        if save:
            self.__init_from_Arr(blurred)
            
        return blurred

    @staticmethod
    def _normalise(arr:np.ndarray) -> np.ndarray:
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_array = (arr - min_val) / (max_val - min_val)
        return normalized_array
    
    @staticmethod
    def _checkPath(path:str|Path, corSuffix:str):
        return Path(os.path.splitext(path)[0] + corSuffix)

def saveImg(img:Any, path:str, **kwargs):
    tempImg = npImage(img, **kwargs)
    tempImg.save(path, **kwargs)
    return

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
        matrix = np.load(paths[i]).astype(np.float64)  # ensure float for math
        weighted_matrix = matrix * weights[i]
        total_matrix += weighted_matrix

    # Normalize by number of images
    total_matrix /= len(paths)
    return total_matrix

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


    #bmp2npy = [bmp_to_npy, bmp, npy, True]
    
    #iterFolderConvert(*bmp2npy)

    """
    toMerge =  [file for file in npy.iterdir() if file.is_file()]
    weights = np.ones(shape=len(toMerge))
    testVal = weighted_average_images(toMerge, weights)
    boostedTest = boost_contrast(testVal)
    npy_to_png(matrix=testVal, png_path=os.path.join(res, "test"))
    npy_to_png(matrix=boostedTest, png_path=os.path.join(res, "boostedtest"))
    """
    testPath = r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgNPY\0004.npy"
    data = npImage(testPath)
    data.blur(3,True)
    data.save(r"C:\Users\augus\NIN_Stuff\data\koenData\Koen_to_Augustijn\4000imgResults\tester.png",[".png", ".npy"])
    return

if __name__ == "__main__":
    main()