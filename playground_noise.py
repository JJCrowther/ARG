import numpy as np
from numpy import asarray

import redshifting
from PIL import Image

print('\n\n')

if __name__ == '__main__':

    galaxy = Image.open("J000000.80+004200.0.png")
    galaxy_Im = asarray(galaxy)

    print('original shape:', galaxy_Im.shape)

    #images = np.expand_dims(galaxy_Im, axis=0) #Add an extra dimension to the array from the front

    #print('expanded shape:', images.shape)

    Image_redshift = redshifting.add_noise(galaxy_Im)
    Image_redshift /= 2

    #print(type(Image_redshift[0][0]), Image_redshift[0][0].shape)
    #print(type(Image_redshift[1]))

    #print(Image_redshift[1]['shot noise'])
    data_shot = Image.fromarray((Image_redshift[1]['shot noise']* 255).astype(np.uint8))
    data_background = Image.fromarray((Image_redshift[1]['background noise']* 255).astype(np.uint8))

    data_shot.save('half_add_shot_noise.png')
    data_background.save('half_add_background_noise.png')

    
    #Images need to be in a numpy format [out of png using .asrray()]
    #Need to make sure code isn't expecting more dimensions than 1 image provides (can use .expand_dims)
    