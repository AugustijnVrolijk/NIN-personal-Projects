from scipy.ndimage import gaussian_filter
import cv2
import numpy as np

def laplace():
    # Load receptive field image as grayscale array
    image = cv2.imread('receptive_field.png', cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1)

    # Apply Laplacian (second derivative)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Optional: measure variance or max value to decide if RF has strong structure
    contrast_metric = np.var(laplacian)

def difference_of_gaussians():
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
