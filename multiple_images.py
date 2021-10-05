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
import glob
import redshifting
import cv2
import os

print(os.getcwd())

"""
def folder_images(folder):
    images = []
    for name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, name))
        if img is not None:
            img = asarray(img)
            images.append(img)
    return images
"""

if __name__ == '__main__':
    print('\n Begin \n')
    imgs = []
    for filename in glob.iglob(os.getcwd() + '**/*.png', recursive=True):
        galaxy_image=Image.open(filename)
        galaxy_array=asarray(galaxy_image)
        imgs.append(galaxy_array)
    print('Images:', len(imgs))
    print('\n End')

"""
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

    Images = []

    for x in len(J000test):
        Im = Image.open(x)
        galaxy_Im = asarray(Im)
        Images.append(galaxy_Im)
        """
