import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.feature import blob_dog, blob_log, blob_doh
from matplotlib.patches import Ellipse
from imageComp import npImage
from scipy.optimize import curve_fit
from pathlib import Path

def fit_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y):
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
    skew_term = skew_x * (x - xo) + skew_y * (y - yo)

    return (gauss + skew_term).ravel()

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
    x_corr, y_corr, is_neg = ret_flags
    # Create coordinate grid
    x_coords, y_coords = patch.shape
    y = np.arange(x_coords)
    x = np.arange(y_coords)
    x, y = np.meshgrid(x, y)

    #showBlobs(patch, [(t_y, t_x, blob[2])])  # Show the patch with the blob center
    # Initial guess for Gaussian parameters
    initial_guess = (
        patch.mean() + patch.std(),  # amplitude
        t_x, t_y,                     # center
        sigma, sigma,              # sigma_x, sigma_y
        0,                         # theta (no rotation to start)
        0, 0                        # skew_x, skew_y (no skew to start)
    )

    # Fit
    try:
        popt, _ = curve_fit(fit_gaussian, (x, y), patch.ravel(), p0=initial_guess)
    except RuntimeError as e:
        print("Fit failed.")
        raise e
    #popt = amplitute, xo, yo, sigma_x, sigma_y, theta

    #show_gaussian_fits(patch, [popt])

    popt[1] += x_corr
    popt[2] += y_corr

    return popt

def showBlobsDoh(image, blobs):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for blob in blobs:
        y, x, r = blob
        circle = plt.Circle((x, y), r, color='red', fill=False)
        ax.add_patch(circle)
    plt.show()

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
        amplitude, xo, yo, sigma_x, sigma_y, theta, skew_x, skew_y = params
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

    plt.title("Fitted anisotropic Gaussians, skew is not shown")
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

def apply_blob_log(path):
    raw_image = npImage(path)
    #raw_image.blur(15)

    inv = npImage(raw_image.inverse())

    inv = inv.gamma_correction(1.5, linearBoost=False)
    raw = raw_image.gamma_correction(1.5, linearBoost=False)

    min_sigma = 70
    max_sigma = 150
    thresh = 0.9

    res1 = blob_log(raw, min_sigma=min_sigma, max_sigma=max_sigma,threshold=thresh, threshold_rel=thresh)
    res2 = blob_log(inv, min_sigma=min_sigma, max_sigma=max_sigma,threshold=thresh, threshold_rel=thresh)

    res = np.vstack((res1, res2))


def apply_blob_doh():
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
    
if __name__ == "__main__":
    # Example usage

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