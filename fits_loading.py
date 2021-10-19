from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from PIL import Image

if __name__ == '__main__':

    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n')
    imgs = {}
    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all png's within the desired directory

        image_data = fits.getdata(filename)

        print((image_data).shape)
        #print((np.transpose(image_data)).shape)


        #plt.imshow(image_data[0])
        #plt.savefig(f'{filename}_from_fits_data.png')
        #plt.close()