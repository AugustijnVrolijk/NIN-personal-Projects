import numpy as np
import os

from typing import Callable, Any
from pathlib import Path
from PIL import Image
from matLoader import loadMat

class npImage():
    def __init__(self, image:Any, **kwargs):
        
        if isinstance(image, np.ndarray) :
            self.__init_from_Arr(image)
            
        elif isinstance(image, Image, **kwargs):
            self._init_from_Image(image)

        elif isinstance(image, str, **kwargs):   
            self._init_from_Path(image)
    
    def __init_from_Arr(self, arr:np.ndarray):
        self.arr = arr
        self.img = Image.fromarray(arr, mode='L')

    def _init_from_Image(self, image:Image, enforce=True):

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

    def save(self, destPath:str, extension:str=None):
        allowedExtension = [".npy", ".png", None]

        corPath = Path(destPath)
        if extension:
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
    def _is_grayscale(img:Image):
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
    
    def _update_img(self, arr):
        self.img = Image.fromarray(arr, mode='L')
    
    def linearBoostContrast(self, format_max:int=255, clip = False) -> np.ndarray:
        """
        linearly stretches matrix values so that the maximum becomes 255.
        Output is returned as uint8 for PNG saving.
        """
        max_val = np.max(self.arr)
        if max_val == 0:
            return self.arr # avoid divide-by-zero
        
        contrast_stretched = self.arr * (format_max / max_val)
        clipped_arr = np.clip(contrast_stretched, 0, format_max).astype(np.uint8)

        self._update_img(clipped_arr)

        if clip:
            self.arr = clipped_arr
        else:
            self.arr = contrast_stretched

        return self.arr

    @staticmethod
    def _checkPath(path:str|Path, corSuffix:str):
        return Path(os.path.splitext(path)[0] + corSuffix)

def saveImg(img:Any, path, **kwargs):
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
    largerMatrix = total_matrix.astype(np.float128)
    largerMatrix /= len(paths)
    return largerMatrix

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
    testPath = r"C:\Users\augus\NIN_Stuff\data\koenData\old\Ajax_20241012_001_normcorr_SPSIG_Res.mat"
    data = loadMat(testPath)
    print("Done!")
    return

if __name__ == "__main__":
    main()