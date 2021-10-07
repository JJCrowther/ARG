import sys
import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import interpn
import astropy.convolution as conv
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import redshifting

def crop_and_resize(image, scale_factor):
    #Set boundries for cropping
    left = (424 - (424*scale_factor))/2
    right = 424 - left
    top = left
    bottom = right

    cropped_image = image.crop((left, top, right, bottom))
    newsize = (424, 424)
    cropped_image = cropped_image.resize(newsize)
    return cropped_image

if __name__ == '__main__':
    
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    scale = 0.2

    Im = Image.open("J000000.80+004200.0.png")
    galaxy_Im = asarray(Im)

    #Zooming
    zoom_output = redshifting.zoom_contents(galaxy_Im, scale, image_axes=[0, 1], conserve_flux=False, method='nearest')
    zoomed_pic = Image.fromarray(((1 - zoom_output)* 255).astype(np.uint8))

    zoomed_pic.save(f'zoomed_pre_crop.png')

    #Convolution
    pre_convolved = np.expand_dims(zoomed_pic, axis=0)

    convolve_output = redshifting.convolve_psf(pre_convolved, seeing=3.5) #Contorl seeing here
    convolve_output = convolve_output.squeeze()
    convolved_pic = Image.fromarray(((1 - convolve_output)* 255).astype(np.uint8))

    convolved_pic.save(f'convolved_pre_crop.png')

    #Crop and resize
    cropped_image = crop_and_resize(convolved_pic, scale)

    #Noise
    #TBD

    cropped_image.save(f'Resized_image_convoluted.png')