import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import pandas as pd
import os


from tqdm import tqdm
from skimage.feature import blob_dog, blob_log, blob_doh
from matplotlib.patches import Ellipse
from imageComp import npImage
from scipy.optimize import curve_fit
#from scipy.integrate import simpson
from pathlib import Path
from imageComp import expand_folder_path
from concurrent.futures import ThreadPoolExecutor




def fit_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta):
    """
    amplitude = peak height of the Gaussian.

    (xo, yo) = center of the Gaussian.

    sigma_x, sigma_y = spreads (standard deviations) in x and y.

    theta = rotation angle (0 means aligned with axes).

    a, b, and c are computed from sigma_x, sigma_y, and theta for anisotropic + rotated Gaussians.
    """
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    gauss = amplitude * np.exp(
        - (a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2))
    )

    # Affine skew term added linearly
    return gauss

def fit_gaussian_skew(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y):
    x,y = coords
    skew_term = skew_x * (x - xo) + skew_y * (y - yo)
    gaussian = fit_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta)
    return (gaussian + skew_term)

def ravel_fit_gaussian_skew(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y):
    return fit_gaussian_skew(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y).ravel()

def getPatchCoords(image:np.ndarray, blob:np.ndarray, buffer_ratio:int, is_doh:bool=False):
    assert len(blob) == 3, "Blob must contain at least (y, x, sigma)"

    y0, x0, sigma = blob[:3]  # blob_doh returns (y, x, sigma)
    x0, y0 = int(x0), int(y0)

    if not is_doh:
        sigma = np.sqrt(2)*sigma

    patch_radius = int(sigma*buffer_ratio)

    x_min = max(x0 - patch_radius, 0)
    x_max = min(x0 + patch_radius, image.shape[1])
    y_min = max(y0 - patch_radius, 0)
    y_max = min(y0 + patch_radius, image.shape[0])
    patch = image[y_min:y_max, x_min:x_max]

    x,y = patch.shape
    x_buffer = int(x/4)
    y_buffer = int(y/4)

    inner_patch = patch[y_buffer:-y_buffer, x_buffer:-x_buffer]

    is_neg = False
    if np.mean(inner_patch) < np.mean(patch):
        is_neg = True
        patch = 255 - patch  # Invert the patch if it's negative
    
    patch = patch - patch.min()

    x_corr = 0
    if x_min == 0:
        t_x = x0
    else:
        x_corr = x_min
        t_x = x0 - x_min

    y_corr = 0
    if y_min == 0:
        t_y = y0
    else:
        y_corr = y_min
        t_y = y0 - y_min

    ret_blob = (t_y, t_x, sigma)
    ret_flags= (x_corr, y_corr, is_neg)

    return patch, ret_blob, ret_flags

def fit_blob_to_patch(image, blob, buffer_ratio=1.5, is_doh=False):
    
    # Extract a patch around the blob
    patch, ret_blob, ret_flags = getPatchCoords(image, blob, buffer_ratio, is_doh)
    t_y, t_x, sigma = ret_blob
    x_corr, y_corr, _ = ret_flags
    # Create coordinate grid
    x_coords, y_coords = patch.shape
    r_y = np.arange(x_coords)
    r_x = np.arange(y_coords)
    m_x, m_y = np.meshgrid(r_x, r_y)

    #showBlobs(patch, [(t_y, t_x, blob[2])])  # Show the patch with the blob center
    # Initial guess for Gaussian parameters
    initial_guess = (
        patch.mean() + patch.std(), # amplitude
        t_x, t_y,                   # center
        sigma, sigma,               # sigma_x, sigma_y
        0,                          # theta (no rotation to start)
        0, 0                        # skew_x, skew_y (no skew to start)
    )

    # Fit
    try:
        popt, _ = curve_fit(ravel_fit_gaussian_skew, (m_x, m_y), patch.ravel(), p0=initial_guess)
    except RuntimeError as e:
        print("Fit failed.")
        raise e
    
    """ show images for debugging
    plt.imshow(patch)
    gauss_skew = fit_gaussian_skew((m_x, m_y), *popt)
    gauss = fit_gaussian((m_x, m_y), *popt[:-2])  # Exclude skew parameters for volume calculation
    plt.imshow(gauss)
    plt.imshow(gauss_skew)
    """
    
    popt[1] += x_corr # Adjust x center based on patch coordinates
    popt[2] += y_corr # Adjust y center based on patch coordinates

    return popt

def confidence_bounding_box(sigma_x, sigma_y, theta, Z=1):
    """
    Returns the minimal axis-aligned box around a rotated ellipse
    defined by the Gaussian confidence region. ~ around 40% confidence for z=1
    """
    # Compute ellipse axes in x and y directions
    x_radius = Z * np.sqrt((sigma_x * np.cos(theta))**2 + (sigma_y * np.sin(theta))**2)
    y_radius = Z * np.sqrt((sigma_x * np.sin(theta))**2 + (sigma_y * np.cos(theta))**2)

    r_x = np.arange(2*x_radius)
    r_y = np.arange(2*y_radius)
    m_x, m_y = np.meshgrid(r_x, r_y)

    return m_x, m_y, x_radius, y_radius

def calcVolume(amplitude, sigma_x, sigma_y, theta):
    Z = 0.5  # Confidence level, e.g., Z=1 for ~40% confidence

    m_x, m_y, x0, y0 = confidence_bounding_box(sigma_x, sigma_y, theta, Z)
    gauss = fit_gaussian((m_x, m_y), amplitude, x0, y0, sigma_x, sigma_y, theta)
    #gauss_and_skew = fit_gaussian_skew((m_x, m_y), *popt)
    threshold = amplitude * np.exp((Z**2)/-2)# Threshold for volume calculation
    
    filtered_gauss = gauss[gauss > threshold]
    
    #mask = (gauss >= threshold)
    #masked_gauss = np.where(mask, gauss, 0)
    #plt.imshow(masked_gauss)

    volume = filtered_gauss.sum()
    return volume

def showBlobs(image, blobs):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for blob in blobs:
        y, x, r = blob
        r = np.sqrt(2)*r
        circle = plt.Circle((x, y), r, color='red', fill=False)
        ax.add_patch(circle)
    return fig

def show_gaussian_fits(image, gaussian_params_list):
    """
    Overlays multiple confidence ellipses for each fitted Gaussian on the image.

    Each Gaussian is visualized with several ellipses indicating different 
    confidence levels based on amplitude decay.

    Args:
        image: 2D numpy array (grayscale image).
        gaussian_params_list: list of tuples (amplitude, xo, yo, sigma_x, sigma_y, theta)
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Confidence levels as fractions of the peak amplitude
    levels = [0.7, 0.9]  # from broader to tighter
    colors = ['orange', 'red']  # matching order
    alphas = [0.4, 0.8]

    for params in gaussian_params_list:
        _, xo, yo, sigma_x, sigma_y, theta, _, _ = params
        angle_deg = -np.degrees(theta)

        for level, color, alpha in zip(levels, colors, alphas):
            # Calculate radius for exp(-Z^2) = level ⇨ Z = sqrt(-ln(level))
            z = np.sqrt(-2 * np.log(level))

            # Width/height of ellipse for that level
            width = 2 * z * sigma_x
            height = 2 * z * sigma_y

            ellipse = Ellipse(
                (xo, yo),
                width,
                height,
                angle=angle_deg,
                edgecolor=color,
                facecolor='none',
                lw=2,
                alpha=alpha
            )
            ax.add_patch(ellipse)

    return fig

def filter_blobs(blobs, min_x=10, max_x=1000, min_y=10, max_y=500):
    """
    Filter blobs based on their coordinates.
    
    x = 1920
    y = 1080
    minimum shaves off the blobs found on the outer edge

    top left is (0,0)
    bottom right is (1920, 1080)
    """

    filtered = []
    for blob in blobs:
        y, x, r = blob
        if min_x <= x <= max_x and min_y <= y <= max_y:
            filtered.append(blob)
    return np.array(filtered)

def apply_blob_dog(path):
    raw_image = npImage(path)
    #raw_image.blur(15)

    inv_image = npImage(raw_image.inverse())

    #inv = inv_image.gamma_correction(1.5, linearBoost=False)
    #raw = raw_image.gamma_correction(1.5, linearBoost=False)

    inv = inv_image.arr
    raw = raw_image.arr

    min_sigma = 70
    max_sigma = 150
    thresh = 0.9

    res1 = blob_dog(raw, min_sigma=min_sigma, max_sigma=max_sigma,threshold=thresh, threshold_rel=thresh)
    res2 = blob_dog(inv, min_sigma=min_sigma, max_sigma=max_sigma,threshold=thresh, threshold_rel=thresh)

    res = np.vstack((res1, res2))
    return res

def apply_blob_doh(path):
    raw_image = npImage(path)
    raw_image.blur(15)
    raw_image.gamma_correction(1.5)

    # Apply Gaussian blur
    #blurred = cv2.GaussianBlur(image, (35, 35), sigmaX=3)
    image = raw_image.arr
    min_sigma = 70
    max_sigma = 300
    n=25
    thresh = 0.5

    res = blob_doh(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=n,threshold=thresh, threshold_rel=thresh)
    return res

def quantifyRF(originImg, destImg):
    try:
        raw = npImage(originImg).arr

        raw_res = apply_blob_dog(originImg)
        res = filter_blobs(raw_res)

        finalGauss = []
        final_volume = 0
        for blob in res:
            params = fit_blob_to_patch(raw, blob, buffer_ratio=1.5, is_doh=False)
            amplitute, _, _, sigma_x, sigma_y, theta, _, _ = params
            #amplitute, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y
            volume = calcVolume(amplitute, sigma_x, sigma_y, theta)  # Calculate volume based on fitted parameters
            
            finalGauss.append(params)
            final_volume += volume

            """ debugging output
            amplitude, x1, y1, sigma_x, sigma_y, theta, skew_x, skew_y = params
            y0, x0, sigma = blob[:3]
            sigma = int(np.sqrt(2)*sigma)
            print(f"Fitted parameters:\n"
                    "           orig | fitted\n"
                    f"x:         {int(x0)}  | {int(x1)}\n"
                    f"y:         {int(y0)}  | {int(y1)})\n"
                    f"sigma:     {int(sigma)}  | {int(sigma_x)}, {int(sigma_y)}\n"
                    f"skew:      n\\a  | {skew_x}, {skew_y}\n"
                    f"amplitude: n\\a  | {int(amplitude)}\n"
                    f"theta:     n\\a  | {theta}\n")
            """
        fig = show_gaussian_fits(raw, finalGauss)
        fig.savefig(destImg)
        plt.close(fig)
        return final_volume
    except Exception as e:
        print(f"failed to process {originImg}")
        #raise e
        return 0
    
def apply_blob_fitting_dog(saveFolder, imgs):
    extension = ".png"

    for path in imgs:
        path = Path(path)
        name = path.stem
        print(f"Processing {path}")
        imageObj = npImage(path) #blur(15, save=False)
        imageObj.save(os.path.join(saveFolder, name))

        raw = imageObj.arr
        raw_res = apply_blob_dog(path)

        fig = showBlobs(raw, raw_res)
        fig.savefig(os.path.join(saveFolder, f"{name}_all_blobs{extension}"))
        plt.close(fig)

        res = filter_blobs(raw_res)

        fig = showBlobs(raw, res)
        fig.savefig(os.path.join(saveFolder, f"{name}_filtered_blobs{extension}"))
        plt.close(fig)

        finalGauss = []
        for blob in res:
            params = fit_blob_to_patch(raw, blob, buffer_ratio=1.5, is_doh=False)
            amplitute, _, _, sigma_x, sigma_y, theta, _, _ = params
            #params = amplitute, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y
            volume = calcVolume(amplitute, sigma_x, sigma_y, theta)  # Calculate volume based on fitted parameters

            amplitude, x1, y1, sigma_x, sigma_y, theta, skew_x, skew_y = params
            y0, x0, sigma = blob[:3]
            sigma = int(np.sqrt(2)*sigma)
            finalGauss.append(params)
            print(f"Fitted parameters:\n"
                    "           orig | fitted\n"
                    f"x:         {int(x0)}  | {int(x1)}\n"
                    f"y:         {int(y0)}  | {int(y1)})\n"
                    f"sigma:     {int(sigma)}  | {int(sigma_x)}, {int(sigma_y)}\n"
                    f"skew:      n\\a  | {skew_x}, {skew_y}\n"
                    f"amplitude: n\\a  | {int(amplitude)}\n"
                    f"theta:     n\\a  | {theta}\n")

        fig = show_gaussian_fits(raw, finalGauss)
        fig.savefig(os.path.join(saveFolder, f"{name}_gauss{extension}"))

        plt.close(fig)
    return

@expand_folder_path
def comp_extra_intra(images):
    total = len(images)
    intra = np.ndarray(total)
    extra = np.ndarray(total)
    for i, path in enumerate(images):
        #print(f"Processing {i+1}/{total}: {path}")
        im_arr = npImage(path).arr
        y,x = im_arr.shape
        left_T = im_arr[:int(y/2), :int(x/2)]
        right_T = im_arr[:int(y/2), int(x/2):]
        left_B = im_arr[int(y/2):, :int(x/2)]
        right_B = im_arr[int(y/2):, int(x/2):]

        intra[i] = left_T.mean()
        extra[i] = (left_B.mean()+right_T.mean()+right_B.mean())/3
    
    return intra, extra

def compare_folder(destFolder, savePath):
    
    conditionNames = ["FamiliarNotOccluded",
                      "FamiliarOccluded", 
                      "NovelNotOccluded", 
                      "NovelOccluded"]

    for name in conditionNames:
        cond_path = os.path.join(destFolder, name)
        intra, extra = comp_extra_intra(cond_path)
        diff = np.abs(intra - extra)
        print(f"{name}:\n")
        print(f"Intra:      {intra.mean():.2f} ± {intra.std():.2f}")
        print(f"Extra:      {extra.mean():.2f} ± {extra.std():.2f}")
        print(f"Difference: {diff.mean():.2f} ± {diff.std():.2f}\n")
    return

def paraQuantRFs(rootPath, folderNames, saveFolder):

    allPaths = []
    for dirName in folderNames:
        t_path = Path(os.path.join(rootPath, dirName))
        dest_path = Path(os.path.join(saveFolder, dirName))
        dest_path.mkdir(exist_ok=True)

        for f in t_path.iterdir():
            if f.is_file() and f.name != "Thumbs.db":
                savePath = Path(os.path.join(saveFolder,dirName,f.name))
                allPaths.append([f, savePath, dirName, f.stem])

    final_res = {name:[] for name in folderNames}
    results = []
    with ThreadPoolExecutor() as executor:
        for path in tqdm(allPaths):
            results.append((executor.submit(quantifyRF, path[0], path[1]), path[2], path[3]))

        
        for pair in tqdm(results):
            res, condition, name = pair
            t_res = res.result()/1000000 #area under is always in the millions
            final_res[condition].append((name, t_res))
    
    analysisPath = Path(os.path.join(saveFolder, "analysis"))
    analysisPath.mkdir(exist_ok=True)
    for key in final_res.keys():  
        df = pd.DataFrame(final_res[key], columns=["Name", "AOC / 1000000"])
        vals = df["AOC / 1000000"].to_numpy()
        print(f"{key}:\n",
              f"mean: {vals.mean()}\n",
              f"std:  {vals.std()}\n")

        df.to_csv(os.path.join(analysisPath, f"{key}_results.csv"), index=False)
    return

def random():
    saveFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\RFquantification"

    imgs = [r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Anton_49.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Anton_54.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Ajax_305.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Ajax_387.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Fctwente_13.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Lana_347.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\NovelNotOccluded\Ajax_96.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\NovelNotOccluded\Anton_408.png",
            ]

    img1 = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded"
    cond_folder = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal"
    #compare_folder(cond_folder, None)
    apply_blob_fitting_dog(saveFolder, imgs)

def quantRFs():
    originFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\RFSigNormal"
    destFolder = r"C:\Users\augus\NIN_Stuff\data\koenData\RFquantification"

    subFolders = [r"NovelOccluded",
          r"NovelNotOccluded",
          r"FamiliarOccluded",
          r"FamiliarNotOccluded"]
    
    paraQuantRFs(originFolder, subFolders, destFolder)
    

if __name__ == "__main__":
    quantRFs()
    