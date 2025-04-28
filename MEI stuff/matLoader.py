import scipy
import numpy as np

from collections import deque, UserDict
from pathlib import Path

from collections import UserDict

class matData(UserDict):
    def __init__(self, *args, **kwargs):
        init_data = dict(*args, **kwargs)
        # Recursively wrap nested dicts
        for key, value in init_data.items():
            init_data[key] = self._wrap(value)
        super().__init__(init_data)

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError:
            msg = f"{self} has no attribute '{key}'"
            raise AttributeError(msg)

    def __setattr__(self, key, value):
        if key == 'data':  # required for UserDict internals
            super().__setattr__(key, value)
        else:
            self.data[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, dict):
            return matData(value)
        return value

class matLoader():

    @staticmethod
    def load(rawpath:str) -> matData:
        path = Path(rawpath)
        if matLoader._checkFile(path):
            data, _ = matLoader._loadScipy(path) #the _ is the file version
            datadict = matLoader._unwrap(data)
            return matData(datadict)
        
        FileExistsError("incorrect file")
    
    @staticmethod
    def _checkFile(path:Path) -> bool:
        validTypes = [".mat"]
        check = path.is_file()
        if path.suffix not in validTypes:
            check = False
        return check
    
    @staticmethod
    def _loadScipy(path:Path):
        with open(path, 'rb') as f:
            data = scipy.io.loadmat(f)
        if not isinstance(data, dict):
            TypeError(f"Scipy did not load {path} as the required dict")

        fileVers = data["__header__"].split()[1] #not doing anything with this but could be useful
        return data, fileVers

    @staticmethod
    def _getFromWrap(wrapped:np.ndarray|dict):
        iswrapped = True
        if isinstance(wrapped, dict):
            iswrapped = False

        while iswrapped:
            iswrapped = False
            wrapped = wrapped.squeeze()
            if isinstance(wrapped.dtype, np.dtypes.ObjectDType):
                wrapped = wrapped.item()
                iswrapped = True
            elif isinstance(wrapped.dtype, np.dtypes.VoidDType):
                wrapped = matLoader._voidToDict(wrapped)

        return wrapped
    
    @staticmethod
    def _voidToDict(voidArr:np.ndarray) -> dict:
        """
        iterate through the dtype as the idx and then add their values as keys, dont need to unwrap their values as _unwrap will do this

        look at exception cases where I may need to _collapseStruct: 
        
        data['info']['calibration'].squeeze().item().squeeze()['uv'].shape
(13,)
data['info']['calibration'].squeeze().item().squeeze()['uv'].dtype
dtype('O')
data['info']['calibration'].squeeze().item().squeeze()['uv'].item()

Error: (considered as 13 different items so cant shrink down to one with item())
        """
        return 

    def _collapseStruct():
        #collapse structure arrays into a single structure if the fields are the same
        #otherwise we get an object with an attribute for every single seperate struct arr, 
        #with duplicate fields between them
        """
        i.e. arr with struct: 
            x    y   z
        1
        2
        3

        would be stored as: arr.keys() = (1{(x,y,z)},2{(x,y,z)},3{(x,y,z)})
        arr.1.keys() = (x,y,z)
        and we want to collapse to:
        arr.keys() = (x,y,z)
        where each x y and z attribute has 3 values
        """
        return

    @staticmethod
    def _unwrap(data:dict):
        finalData = dict()

        toIgnore = ['__header__', '__version__', '__globals__']
        keyStack = [key for key in data.keys() if key not in toIgnore]

        for curKey in keyStack:
            print(curKey)

            unwrappedData = matLoader._getFromWrap(data[curKey])
            if isinstance(data[curKey], dict):
                finalData[curKey] = matLoader._unwrap(data[curKey])
            else:
                finalData[curKey] = unwrappedData

        return finalData
    


def loadMat(path):
    return matLoader.load(path)

if __name__ == "__main__":
    testPath = r"C:\Users\augus\NIN_Stuff\data\koenData\old\Ajax_20241012_001_normcorr_SPSIG_Res.mat"
    testLoad = matLoader()
    data = testLoad.load(testPath)
    """
    testStruct = matData()
    testStruct.val1 = 12412
    testStruct.path = testPath
    print(testStruct.keys())
    print(testStruct["val1"])
    testStruct.val4 = matData()
    print(testStruct.keys())


    testStruct.val4.testerino1 = 999
    testStruct.val4.testerino5 = "fdsfsdsjfusda"
    print(testStruct.val4["testerino5"])
    print(testStruct.keys())
    """