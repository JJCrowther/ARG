import numpy as np
from numpy import asarray

import redshifting
from PIL import Image

print('\n\n')

if __name__ == '__main__':

    galaxy = Image.open("J000000.80+004200.0.png")
    galaxy_Im = asarray(galaxy)

    print('max value is:', np.max(galaxy_Im))

    #reformed_galaxy = Image.fromarray(((1 - galaxy_Im)*255).astype(np.uint8))
    #reformed_galaxy.save('reformed.png')

    #images = np.expand_dims(galaxy_Im, axis=0) #Add an extra dimension to the array from the front

    print('original shape:', galaxy_Im.shape)
    #print('Galaxy_Im original look:', galaxy_Im)

    for i in [0.125]:
        for fact in [True, False]:
            zoom_output = redshifting.zoom_contents(galaxy_Im, scale=i, image_axes=[0, 1], conserve_flux=fact, method='nearest')

            #print(zoom_output.shape)

            zoomed_pic = Image.fromarray(((1 - zoom_output)* 255).astype(np.uint8))
            zoomed_pic.save(f'Scale {i}, Conserve Flux {fact} - Zoomed_gal.png')


    #print(zoom_output.shape)
    #print(zoom_output)

   
