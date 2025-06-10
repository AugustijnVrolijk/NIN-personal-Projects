from skimage.feature import blob_dog, blob_log, blob_doh
import numpy as np
import matplotlib.pyplot as plt

from imageComp import npImage
from scipy.optimize import curve_fit

def fit_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    (x, y) are meshgrid coordinates.

    amplitude = peak height of the Gaussian.

    (xo, yo) = center of the Gaussian.

    sigma_x, sigma_y = spreads (standard deviations) in x and y.

    theta = rotation angle (0 means aligned with axes).

    offset = baseline value (i.e., background intensity).

    a, b, and c are computed from sigma_x, sigma_y, and theta for anisotropic + rotated Gaussians.
    """
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp(
        - (a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2))
    )
    return g.ravel()

def fit_blob_to_patch(image, blob, patch_radius=50):
    y0, x0, sigma = blob[:3]  # blob_doh returns (y, x, sigma)
    x0, y0 = int(x0), int(y0)

    # Extract a patch around the blob
    x_min = max(x0 - patch_radius, 0)
    x_max = min(x0 + patch_radius, image.shape[1])
    y_min = max(y0 - patch_radius, 0)
    y_max = min(y0 + patch_radius, image.shape[0])
    patch = image[y_min:y_max, x_min:x_max]

    # Create coordinate grid
    y = np.arange(y_min, y_max)
    x = np.arange(x_min, x_max)
    x, y = np.meshgrid(x, y)

    # Initial guess for Gaussian parameters
    initial_guess = (
        patch.max() - patch.min(),  # amplitude
        x0, y0,                     # center
        sigma, sigma,              # sigma_x, sigma_y
        0,                         # theta (no rotation to start)
        patch.min()                # offset
    )

    # Fit
    try:
        popt, _ = curve_fit(fit_gaussian, (x, y), patch.ravel(), p0=initial_guess)
        return popt  # fitted parameters
    except RuntimeError:
        print("Fit failed.")
        return None

def showBlobsDog(image, blobs):
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
    plt.show()

def filter_blobs(blobs, min_x=30, max_x=1000, min_y=30, max_y=500):
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

    inv = inv_image.gamma_correction(1.5, linearBoost=False)
    raw = raw_image.gamma_correction(1.5, linearBoost=False)

    inv = inv_image.arr
    raw = raw_image.arr

    min_sigma = 70
    max_sigma = 150
    thresh = 0.9

    res1 = blob_dog(raw, min_sigma=min_sigma, max_sigma=max_sigma,threshold=thresh, threshold_rel=thresh)
    res2 = blob_dog(inv, min_sigma=min_sigma, max_sigma=max_sigma,threshold=thresh, threshold_rel=thresh)

    res = np.vstack((res1, res2))
    print(f"Found {len(res)} blobs in {path}")
    print(res)
    showBlobs(raw, res)
    res = filter_blobs(res)
    showBlobs(raw, res)

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
    showBlobs(inv, res2)

    res = np.vstack((res1, res2))
    print(f"Found {len(res)} blobs in {path}")
    showBlobs(raw, res)

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
    print(f"Found {len(res)} blobs in {path}")
    showBlobs(image, res)
    
if __name__ == "__main__":
    # Example usage
    imgs = [r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Anton_49.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Anton_54.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Ajax_305.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Ajax_387.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Fctwente_13.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Lana_347.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\NovelNotOccluded\Ajax_96.png",
            r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\NovelNotOccluded\Anton_408.png",
            ]

    for path in imgs:
        print(f"Processing {path}")
        apply_blob_dog(path)
