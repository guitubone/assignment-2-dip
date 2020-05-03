import numpy as np
import imageio
import matplotlib.pyplot as plt

def bilateral_conv_point(image, x, y, filter_sz, sigma_s, sigma_r):
    c = (filter_sz-1)//2
    sub_image = image[x-c : x+c+1, y-c : y+c+1]
    
    # Spatial Gaussian
    kernel_s = lambda x : (1/(2*np.pi*sigma_s*sigma_s)) * np.exp(-(x*x)/(2*sigma_s*sigma_s))
    gaussian_s = [[np.linalg.norm(np.array((i, j)) - np.array((filter_sz//2, filter_sz//2))) for j in range(filter_sz)] for i in range(filter_sz)]
    gaussian_s = np.array([[kernel_s(x) for x in y] for y in gaussian_s])
    
    # Range Gaussian
    kernel_r = lambda x : (1/(2*np.pi*sigma_r*sigma_r)) * np.exp(-(x*x)/(2*sigma_r*sigma_r))
    gaussian_r = [[sub_image[filter_sz//2, filter_sz//2] - sub_image[i, j] for j in range(filter_sz)] for i in range(filter_sz)]
    gaussian_r = np.array([[kernel_r(x) for x in y] for y in gaussian_r])
    
    filter = gaussian_s * gaussian_r
    
    result = np.sum(sub_image * filter)
    result /= np.sum(filter)
    return result

def bilateral_conv_2d(image, filter_sz, sigma_s, sigma_r):
    c = (filter_sz-1)//2

    res_image = np.empty(image.shape)
    image = np.pad(image, (c, c), 'constant', constant_values=(0))

    for i in range(image.shape[0]-c-1):
        for j in range(image.shape[1]-c-1):
            res_image[i, j] = bilateral_conv_point(image, i+c, j+c, filter_sz, sigma_s, sigma_r)

    image = image[c : image.shape[0] - c , c : image.shape[1] - c]
            
    return res_image
