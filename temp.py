# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import glob
import os
import warnings
from astropy.io import fits
#import FITs_to_PNG_MW
#from PIL import Image
#from numpy import asarray
import numpy as np

def photon_counts_from_FITS(imgs, bands='grz', mn=0.1, mx=100.):
    """
    Convert a FITS data file into a photon counts array       
    """
    
    size = imgs[0].shape[1]
    
    soft_params = dict(g=(0, 0.9e-10),
                     r=(1, 1.2e-10),
                     z=(2, 7.4e-10) #We think this is the right plane order...hopefully
                     )
           
    #counts_data = [] #needs to be turned into a (3, x, x) array         
    img_counts = np.zeros((3, size, size), np.float32)
    for im, band in zip(imgs, bands):  #im is an (x, x)
        plane, softening_parameters = soft_params.get(band, (0, 1.))
        counts = flux_to_counts(im, softening_parameters, band) #im is a (3, x, x) array ; softening_parameters is an int
        img_counts[plane, :, :] = counts #counts is (x, x) in shape
        #if band == 'z':
        #    counts_data.append(img) #should make count_data a list of (3, x, x)
        #Scale to distance using d_I, d_O nd then add sqrt(n)
    
    new_position_counts = poisson_noise(img_counts, 2) #scaling by 2
    
    #scaled_data=[]
    img_scaled = np.zeros((3, size, size), np.float32)
    for count, band in zip(new_position_counts, bands): #count is a (x, x)
        plane, softening_parameters = soft_params.get(band, (0, 1.))
        nMgys = counts_to_flux(count, band)
        img_scaled[plane, :, :] = nMgys #nMgys is (x, x) in shape
        #if band == 'z': 
        #    scaled_data.append(img)

        
    return imgs, img_counts, img_scaled

def asinh_mag(img, soft_param):
    """
    Converts an image given by a (3, x, x) array of units nanomaggies into a 
    (3, x, x) array of asinh magnitudes as detailed in Lupton et al. (1999).
    
    Inputs;
        img - (3, x, x) numpy array[floats]
        soft_param - b-values associated with teh asinh magnitude method [float]
    """
    mag = -(2.5/np.log(10)) * np.arcsinh((img/(2*soft_param)) + np.log(soft_param)) #Need to check the maggie vs Jansky units here to be safe
    return mag

def asinh_mag_to_flux(Im, soft_param, m_0=22.5, f_0=1e9, a=2.5*np.log10(np.exp(1))):
    """
    
    """
    mag = asinh_mag(Im, soft_param)
    flux = 2*soft_param*f_0*(np.sinh((m_0 - 2.5*np.log10(soft_param*f_0) - mag)/a))
    return flux

def flux_to_counts(im, softening_parameters, band, exposure_time_seconds = 90. * 3.):
    """

    """
    photon_energy = dict(g=(0, 4.12e-19), #475nm
                         r=(1, 3.20e-19), #622nm
                         z=(2, 2.20e-19) #905nm
                         )# TODO assume 600nm mean freq. for gri bands, can improve this
    
    size = im.shape[1]
    img_nanomaggies_nonzero = np.clip(im, 1e-9, None) #Array with no values below 1e-9
    img_photons = np.zeros((size, size), np.float32)

    energy = photon_energy.get(band, 1)[1] #.get has inputs (key, value) where value is returned if key deos not exist
    #flux = asinh_mag_to_flux(Im, softening_parameters)
    flux = img_nanomaggies_nonzero * 3.631e-6
    img_photons[:, :] = np.multiply(flux, (exposure_time_seconds / energy)) #the flux values reach the upper limits of float manipulation - need to be scaled by some value for operations to be conducted
    
    
    return img_photons

def poisson_noise(photon_count, x):
    """
    Scales the photon count by 1/d^2 to account for decreased photon numbers at new position
    before adding a poissonly distributed random noise to each channel for each pixel
    """
    photon_at_distance_scale_x = photon_count * (1/x)**2
    photon_with_poisson = photon_at_distance_scale_x + np.random.poisson(np.sqrt(photon_at_distance_scale_x))
    return photon_with_poisson
    
def counts_to_flux(counts, band, exposure_time_seconds = 90. * 3.):
    """
        
    """
    photon_energy = dict(g=(0, 4.12 * 1e-19), #475nm
                         r=(1, 3.20 * 1e-19), #622nm
                         z=(2, 2.20 * 1e-19) #905nm
                         )# TODO assume 600nm mean freq. for gri bands, can improve this
    
    size = counts.shape[1]
    img_flux = np.zeros((size, size), np.float32)
    
    energy = photon_energy.get(band, 1)[1]
    img_flux[:, :] = counts / (exposure_time_seconds / energy)
    
    img_mgy = img_flux / 3.631e-6
        
    return img_mgy
    
def flux_to_mag(counts, soft_param, band, m_0=22.5, f_0=1e9, a=2.5*np.log10(np.exp(1))):
    """
    Takes in a (3, x, x) array[floats] of fluxes and outputs the corrosponding
    magnitudes for the asinh magnitude method (see Lupton et al. (1999))
    """
    flux = counts_to_flux(counts, band)
    sinh_mag = (m_0 - 2.5*np.log10(soft_param*f_0)) - (a * np.arcsinh(flux/(2*soft_param*f_0)))
    return sinh_mag
    
def asinh_mag_to_nMgy(counts, soft_param, band, m_0=22.5, f_0=1e9, a=2.5*np.log10(np.exp(1))):
    """
    Takes in a (3, x, x) array[floats] of asinh magnitudes and converts them into nanomaggie units
    """
    sinh_mag = flux_to_mag(counts, soft_param, band)
    nMgies = 2*soft_param*np.sinh(-((sinh_mag/a) + np.log(soft_param)))

    return nMgies
    
    
if __name__ == '__main__':
    
    dir_name='J000fitstest' #Sets the name of the file the input png's are stored in

    print('\n Begin \n')
    #imgs = {} 
    imgs = []

    for filename in glob.iglob(os.getcwd() + '/' + f'{dir_name}' + '/**/*.fits', recursive=True): #operates over all png's within the desired directory
        try:
            img, hdr = fits.getdata(filename, 0, header=True)
        except Exception:
            warnings.warn('Invalid fits at {}'.format(filename))
        #imgs[i]=img #Unsure if didctionary would be better here if required for a large quantity of data
        imgs.append(img)

    #print('\n finished appending np arrays:', imgs[0].shape)


    j=0
    final_data = {}
    rescaled_photon = {}

    for j in range(len(imgs)):
        final_data[j]=photon_counts_from_FITS(imgs[j])
        
        
        j+=1

    #print('\nCompleted converting to photon counts:', final_data[0])

    
    #print('\nFinished converting back into a numpy array ready for png-ing\n')

    #print('maggies:', imgs)
    #print('\n')
    #print('rescaled photon counts:', final_data[0])
    
    
    print('\n End \n')




    #image = Image.fromarray(((imgs_photon[0])* 255).astype(np.uint8))
    #image.save(f'photon_count_image.png')
