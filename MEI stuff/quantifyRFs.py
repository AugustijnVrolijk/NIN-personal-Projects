from scipy.ndimage import gaussian_filter
import cv2
import numpy as np

def laplace(imagePath):
    # Load receptive field image as grayscale array
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1)

    # Apply Laplacian (second derivative)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Optional: measure variance or max value to decide if RF has strong structure
    contrast_metric = np.var(laplacian)

    print(contrast_metric)
    print("hello")

def difference_of_gaussians(image):
    # Gaussian blurs with different sigmas
    g1 = gaussian_filter(image, sigma=1)
    g2 = gaussian_filter(image, sigma=3)

    dog = g1 - g2
    contrast_metric = np.var(dog)


def fit_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
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

if __name__ == "__main__":
    # Example usage
    img1 = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Anton_54.png"
    laplace(img1) #very good

    img2 = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Ajax_305.png"
    laplace(img2) #bad ish

    img2 = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Ajax_387.png"
    laplace(img2) #very bad

    img2 = r"C:\Users\augus\NIN_Stuff\data\koenData\RFanalysisNormal\FamiliarNotOccluded\Fctwente_13.png"
    laplace(img2) #bad