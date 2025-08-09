import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    k = 5
    pad = k // 2
    padded = np.pad(image, pad, mode='edge')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+k, j:j+k]

            output[i, j] = np.median(region)

    return output

if __name__ == "__main__":
    noisy_image = cv2.imread("noisy_image.png", cv2.IMREAD_GRAYSCALE)
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    #denoised_image = remove_salt_and_pepper_noise(denoised_image)
    #denoised_image = remove_salt_and_pepper_noise(denoised_image)
    cv2.imwrite("denoised_image.png", denoised_image)
