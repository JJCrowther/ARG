import numpy as np
from numpy import asarray
from PIL import Image
import os

print('\n\n')

if __name__ == '__main__':

    galaxy_1 = Image.open(os.getcwd() + "/J000000.80+004200.0.png")
    galaxy_Im_1 = asarray(galaxy_1)

    galaxy_2 = Image.open(os.getcwd() + "/J000fitstest/J000000.80+004200.0.fits.png")
    galaxy_Im_2 = asarray(galaxy_2)

    diff = galaxy_Im_1 - galaxy_Im_2

    diff_im = Image.fromarray(((1 - diff)* 255).astype(np.uint8))
    diff_im.save(f'Sshould_be_white_or_balck.png')
