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
    
    print('\n')

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    input_redshifts = 0.1
    output_redshifts = 0.2

    Im = Image.open("J000000.80+004200.0.png")
    galaxy_Im = asarray(Im)

    d_i = (cosmo.luminosity_distance(input_redshifts))
    d_o = (cosmo.luminosity_distance(output_redshifts))
    scale_factors = (d_i / (1 + input_redshifts)**2) / (d_o / (1 + output_redshifts)**2)

 
    print('Initial values:', galaxy_Im[0][0])

    for x in [0.01, 0.1]:

        galaxy_Im = np.multiply(galaxy_Im, scale_factors*x) #Some may reach values > 255, we need to account for this.
        galaxy_Im = np.add(galaxy_Im, np.sqrt(galaxy_Im)*x)

        #print('Corrected values and shape:', galaxy_Im.shape, galaxy_Im[0][0])

        noisy_pic = Image.fromarray(((galaxy_Im)* 255).astype(np.uint8))

        noisy_pic.save(f'factor_{x}_noisy_pic.png')

    
    #print('\n')
  