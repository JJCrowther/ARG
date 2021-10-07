import numpy as np
from numpy import asarray
import astropy.convolution as conv
import redshifting
from PIL import Image

print('\n\n')

if __name__ == '__main__':

    galaxy = Image.open("J000000.80+004200.0.png")
    galaxy_Im = asarray(galaxy)
    galaxy_Im_2 = np.expand_dims(galaxy_Im, axis=0)

    convolve_output = redshifting.convolve_psf(galaxy_Im_2, seeing=3.5)
    #print(convolve_output)
    print(convolve_output.shape)
    convolve_output = convolve_output.squeeze()
    convolved_pic = Image.fromarray(((1 - convolve_output)* 255).astype(np.uint8))
    convolved_pic.save(f'First_Convolved_Image.png')



