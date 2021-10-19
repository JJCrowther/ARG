import glob
import os
import warnings
from astropy.io import fits
#import FITs_to_PNG_MW
#from PIL import Image
#from numpy import asarray
import numpy as np


if __name__ == '__main__':
    
    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n')
    
    
    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all png's within the desired directory
        try:
            img= fits.open(filename)
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))

    hdr = img[0].header

    print(hdr['bscale'])
