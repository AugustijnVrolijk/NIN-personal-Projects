import scipy
import numpy as np
import mat73

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
            try:
                data = matLoader._loadRaw(path)
            except (NotImplementedError):
                return matData(mat73.loadmat(path))
            
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
    def _loadRaw(path:Path):
        with open(path, 'rb') as f:
            data = scipy.io.loadmat(f)

        if not isinstance(data, dict):
            TypeError(f"Scipy did not load {path} as the required dict")

        #fileVers = data["__header__"].split()[1] #not doing anything with this but could be useful
        return data

    @staticmethod
    def _getFromWrap(wrapped:np.ndarray|dict) -> np.ndarray|dict|None:
        iswrapped = True
        if isinstance(wrapped, dict):
            iswrapped = False

        while iswrapped:
            iswrapped = False
            wrapped = wrapped.squeeze()
            if isinstance(wrapped.dtype, np.dtypes.ObjectDType):
                if wrapped.ndim == 0:
                    if wrapped.size == 0:
                        raise NotImplementedError("there should be a bug if ndim and size == 0")

                    wrapped = wrapped.item()
                    iswrapped = True

                elif wrapped.size == 0:
                    wrapped = None
                else:
                    wrapped = matLoader._collapseStruct(wrapped)

            elif isinstance(wrapped.dtype, np.dtypes.VoidDType):
                wrapped = matLoader._voidToDict(wrapped)

        return wrapped
    
    @staticmethod
    def _voidToDict(voidArr:np.ndarray) -> dict:
        if not isinstance(voidArr.dtype, np.dtypes.VoidDType):
            raise TypeError("_voidToDict requries a numpy array of void type")

        finalDict = {}

        for name in voidArr.dtype.names:
            finalDict[name] = voidArr[name]
        
        return finalDict

    @staticmethod
    def _collapseStruct(array: np.ndarray) -> np.ndarray:
        #collapse structure arrays into a single structure if the fields are the same
        #otherwise we get an object with an attribute for every single seperate struct arr, 
        #with duplicate fields between them
        if not isinstance(array.dtype, np.dtypes.ObjectDType):
            raise TypeError(f"_collapseStruct requires numpy array of type object, cur dtype: {array.dtype}")
        elif array.ndim != 1:
            raise NotImplementedError(f"array must have dimensionality one, please squeeze beforehand, cur ndim: {array.ndim}")
            # it is possible to collapse for more than one dim, need another for loop to iterate based on array.ndim        
            #check each sub array has the same properties

        baseShape = array[0].shape
        baseType = array[0].dtype

        for arr in array:
            if arr.shape != baseShape:
                Warning("object does not have collapsible structure")
                return array
            
            baseType = np.promote_types(baseType, arr.dtype)

        finalArr = np.stack(array, axis=0, dtype=baseType).squeeze()
        return finalArr

    @staticmethod
    def _unwrap(data:dict) -> dict:
        finalData = dict()

        toIgnore = ['__header__', '__version__', '__globals__']
        keyStack = [key for key in data.keys() if key not in toIgnore]

        for curKey in keyStack:
            unwrappedData = matLoader._getFromWrap(data[curKey])

            if isinstance(unwrappedData, dict):
                temp = matLoader._unwrap(unwrappedData)
                finalData[curKey] = temp
            else:
                finalData[curKey] = unwrappedData

        return finalData
    
def loadMat(path):
    return matLoader.load(path)

if __name__ == "__main__":
    pass