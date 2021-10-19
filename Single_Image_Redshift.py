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

if __name__ == '__main__':
    
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

    Im = Image.open("J000000.80+004200.0.png")
    galaxy_Im = asarray(Im)

    #Zooming
    zoom_output = redshifting.zoom_contents(galaxy_Im, scale=0.75, image_axes=[0, 1], conserve_flux=False, method='nearest')
    zoomed_pic = Image.fromarray(((1 - zoom_output)* 255).astype(np.uint8))

    #Convolution
    pre_convolved = np.expand_dims(zoomed_pic, axis=0)

    convolve_output = redshifting.convolve_psf(pre_convolved, seeing=3.5) #Contorl seeing here
    convolve_output = convolve_output.squeeze()
    convolved_pic = Image.fromarray(((1 - convolve_output)* 255).astype(np.uint8))

    #Noise
    #TBD

    convolved_pic.save(f'First_attempt_zoomed_and_convolved.png')


