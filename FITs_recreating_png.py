import glob
import os
import FITs_to_PNG_MW

if __name__ == '__main__':

    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n')
    imgs = {}
    i=0
    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all png's within the desired directory
        i+=1
        imgs[i]=filename

    for i in range(len(imgs)):
        FITs_to_PNG_MW.make_png_from_fits(imgs[i+1], imgs[i+1] + '.png', 424)
