import glob
import os
import warnings
from astropy.io import fits
import FITs_to_PNG_MW
from PIL import Image
from numpy import asarray
import numpy as np

if __name__ == '__main__':
    
    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n')
    #imgs = {} 
    imgs = []
    filenames=[]

    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all png's within the desired directory
        try:
            img, hdr = fits.getdata(filename, 0, header=True)
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        #imgs[i]=img #Unsure if didctionary would be better here if required for a large quantity of data
        imgs.append(img)

    print('\n finished appending np arrays')

    j=0
    imgs_photon = {}
    rescaled_photon = {}

    for j in range(len(imgs)):
        imgs_photon[j]=FITs_to_PNG_MW.lupton_rgb(np.split(imgs[j], 3), desaturate=False)[0] #lupton requires input of a list of numpy arrays where each array is for one of the colour bands
        rescaled_photon[j]=FITs_to_PNG_MW.lupton_rgb(np.split(imgs[j], 3), desaturate=False)[1]
        j+=1

    print('\nCompleted converting to photon counts:', rescaled_photon[0], rescaled_photon[0].shape, np.transpose(rescaled_photon[0]).shape) #might need flipping

    returned_arrays = {}
    for k in range(len(rescaled_photon)):

        returned_array = FITs_to_PNG_MW.ndarray_from_photon_counts(rescaled_photon[k], img[k])
        returned_arrays[k] = returned_array

    print('\nFinished converting back into a numpy array ready for png-ing\n')

    print('maggies:', imgs)
    print('\n')
    print('rescaled photon counts:', imgs_photon[0])
    print('\n')
    print('returned to maggies:', returned_arrays[0])
    
    print('\n End \n')




    image = Image.fromarray(((imgs_photon[0])* 255).astype(np.uint8))
    image.save(f'photon_count_image.png')
