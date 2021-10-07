import glob
import os
import warnings
from astropy.io import fits
import FITs_to_PNG_MW

if __name__ == '__main__':
    
    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n')
    imgs = {}
    i=0
    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all png's within the desired directory
        try:
            img, hdr = fits.getdata(filename, 0, header=True)
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        i+=1
        imgs[i]=img

    imgs_photon = {}
    for j in range(len(imgs)):
        j+=1
        imgs_photon[j]=FITs_to_PNG_MW.lupton_rgb(imgs[j], desaturate=True)

    print('maggies:', imgs[1])
    print('\n')
    print('photon counts:', imgs_photon[1])
    print('\n End \n')
