from astropy.io import fits
import sys
import warnings
from PIL import Image
import numpy as np


def make_png_from_fits(fits_loc, png_loc, png_size):
    '''
    Create png from multi-band fits
    Args:
        fits_loc (str): location of .fits to create png from
        png_loc (str): location to save png
    Returns:
        None
    '''
    try:
        img, hdr = fits.getdata(fits_loc, 0, header=True)
    except Exception:
        warnings.warn('Invalid fits at {}'.format(fits_loc))
    else:  # if no exception getting image

        # TODO wrap?

            # Set parameters for RGB image creation
        _scales = dict(
            g=(2, 0.008),
            r=(1, 0.014),
            z=(0, 0.019))
        _mnmx = (-0.5, 300)

        rgbimg = dr2_style_rgb(
            (img[0, :, :], img[1, :, :], img[2, :, :]),
            'grz',
            mnmx=_mnmx,
            arcsinh=1.,
            scales=_scales,
            desaturate=True)
        save_carefully_resized_png(png_loc, rgbimg, target_size=png_size)

def save_carefully_resized_png(png_loc, native_image, target_size):
    """
    # TODO
    Args:
        png_loc ():
        native_image ():
        target_size ():
    Returns:
    """
    native_pil_image = Image.fromarray(np.uint8(native_image * 255.), mode='RGB')
    nearest_image = native_pil_image.resize(size=(target_size, target_size), resample=Image.LANCZOS)
    nearest_image = nearest_image.transpose(Image.FLIP_TOP_BOTTOM)  # to align with north/east
    nearest_image.save(png_loc)

def lupton_rgb(imgs, bands='grz', arcsinh=1., mn=0.1, mx=100., desaturate=False, desaturate_factor=.01):
    """
    Create human-interpretable rgb image from multi-band pixel data
    Follow the comments of Lupton (2004) to preserve colour during rescaling
    1) linearly scale each band to have good colour spread (subjective choice)
    2) nonlinear rescale of total intensity using arcsinh
    3) linearly scale all pixel values to lie between mn and mx
    4) clip all pixel values to lie between 0 and 1
    Optionally, desaturate pixels with low signal/noise value to avoid speckled sky (partially implemented)
    Args:
        imgs (list): of 2-dim np.arrays, each with pixel data on a band # TODO refactor to one 3-dim array
        bands (str): ordered characters of bands of the 2-dim pixel arrays in imgs
        arcsinh (float): softening factor for arcsinh rescaling
        mn (float): min pixel value to set before (0, 1) clipping
        mx (float): max pixel value to set before (0, 1) clipping
        desaturate (bool): If True, reduce saturation on low S/N pixels to avoid speckled sky
        desaturate_factor (float): parameter controlling desaturation. Proportional to saturation.
    Returns:
        (np.array) of shape (H, W, 3) of pixel values for colour image
    """

    size = imgs[0].shape[1]
    grzscales = dict(g=(2, 0.00526),
                     r=(1, 0.008),
                     z=(0, 0.0135)
                     )

    # set the relative intensities of each band to be approximately equal
    img = np.zeros((size, size, 3), np.float32)
    for im, band in zip(imgs, bands):
        plane, scale = grzscales.get(band, (0, 1.))
        img[:, :, plane] = (im / scale).astype(np.float32)

    I = img.mean(axis=2, keepdims=True)

    if desaturate:
        img_nanomaggies = np.zeros((size, size, 3), np.float32)
        for im, band in zip(imgs, bands):
            plane, scale = grzscales.get(band, (0, 1.))
            img_nanomaggies[:, :, plane] = im.astype(np.float32)
        img_nanomaggies_nonzero = np.clip(img_nanomaggies, 1e-9, None)
        img_ab_mag = 22.5 - 2.5 * np.log10(img_nanomaggies_nonzero)
        img_flux = np.power(10, img_ab_mag / -2.5) * 3631
        # DR1 release paper quotes 90s exposure time per band, 900s on completion
        # TODO assume 3 exposures per band per image. exptime is per ccd, nexp per tile, will be awful to add
        exposure_time_seconds = 90. * 3.
        photon_energy = 600. * 1e-9  # TODO assume 600nm mean freq. for gri bands, can improve this
        img_photons = img_flux * exposure_time_seconds / photon_energy
        img_photons_per_pixel = np.sum(img_photons, axis=2, keepdims=True)

        #In order to work backwards to reproduce the image from photon counts, we will need to re-split the data 
        #back into its 3 bands. maybe do this by taking the rgz ratio initially and keeping that after correction?

        mean_all_bands = img.mean(axis=2, keepdims=True)
        deviation_from_mean = img - mean_all_bands
        signal_to_noise = np.sqrt(img_photons_per_pixel)
        saturation_factor = signal_to_noise * desaturate_factor
        # if that would imply INCREASING the deviation, do nothing
        saturation_factor[saturation_factor > 1] = 1.
        img = mean_all_bands + (deviation_from_mean * saturation_factor)

    rescaling = nonlinear_map(I, arcsinh=arcsinh)/I
    rescaled_img = img * rescaling

    #rescaled_img = (rescaled_img - mn) * (mx - mn)
    #rescaled_img = (rescaled_img - mn) * (mx - mn)

    return np.clip(rescaled_img, 0., 1.), rescaled_img

def dr2_style_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None, desaturate=False):
    '''
    Given a list of image arrays in the given bands, returns a scaled RGB image.
    Originally written by Dustin Lang and used by Kyle Willett for DECALS DR1/DR2 Galaxy Zoo subjects
    Args:
        imgs (list): numpy arrays, all the same size, in nanomaggies
        bands (list): strings, eg, ['g','r','z']
        mnmx (min,max), values that will become black/white *after* scaling. Default is (-3,10)):
        arcsinh (bool): if True, use nonlinear scaling (as in SDSS)
        scales (str): Override preset band scaling. Dict of form {band: (plane index, scale divider)}
        desaturate (bool): If [default=False] desaturate pixels dominated by a single colour
    Returns:
        (np.array) of shape (H, W, 3) with values between 0 and 1 of pixel values for colour image
    '''

    bands = ''.join(bands)  # stick list of bands into single string

    # first number is index of that band
    # second number is scale divisor - divide pixel values by scale divisor for rgb pixel value
    grzscales = dict(
        g=(2, 0.0066),
        r=(1, 0.01385),
        z=(0, 0.025),
    )

    if scales is None:
        if bands == 'grz':
            scales = grzscales
        elif bands == 'urz':
            scales = dict(
                u=(2, 0.0066),
                r=(1, 0.01),
                z=(0, 0.025),
            )
        elif bands == 'gri':
            scales = dict(
                g=(2, 0.002),
                r=(1, 0.004),
                i=(0, 0.005),
            )
        else:
            scales = grzscales

    #  create blank matrix to work with
    h, w = imgs[0].shape
    rgb = np.zeros((h, w, 3), np.float32)

    # Copy each band matrix into the rgb image, dividing by band scale divisor to increase pixel values
    for im, band in zip(imgs, bands):
        plane, scale = scales[band]
        rgb[:, :, plane] = (im / scale).astype(np.float32)

    # TODO mnmx -> (min, max)
    # cut-off values for non-linear arcsinh map
    if mnmx is None:
        mn, mx = -3, 10
    else:
        mn, mx = mnmx

    if arcsinh is not None:
        # image rescaled by single-pixel not image-pixel, which means colours depend on brightness
        rgb = nonlinear_map(rgb, arcsinh=arcsinh)
        mn = nonlinear_map(mn, arcsinh=arcsinh)
        mx = nonlinear_map(mx, arcsinh=arcsinh)

    # lastly, rescale image to be between min and max
    rgb = (rgb - mn) / (mx - mn)

    # default False, but downloader sets True
    if desaturate:
        # optionally desaturate pixels that are dominated by a single
        # colour to avoid colourful speckled sky

        # reshape rgb from (h, w, 3) to (3, h, w)
        RGBim = np.array([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
        a = RGBim.mean(axis=0)  # a is mean pixel value across all bands, (h, w) shape
        # putmask: given array and mask, set all mask=True values of array to new value
        np.putmask(a, a == 0.0, 1.0)  # set pixels with 0 mean value to mean of 1. Inplace?
        acube = np.resize(a, (3, h, w))  # copy mean value array (h,w) into 3 bands (3, h, w)
        bcube = (RGBim / acube) / 2.5  # bcube: divide image by mean-across-bands pixel value, and again by 2.5 (why?)
        mask = np.array(bcube)  # isn't bcube already an array?
        wt = np.max(mask, axis=0)  # maximum per pixel across bands of mean-band-normalised rescaled image
        # i.e largest relative deviation from mean
        np.putmask(wt, wt > 1.0, 1.0)  # clip largest allowed relative deviation to one (inplace?)
        wt = 1 - wt  # invert relative deviations
        wt = np.sin(wt*np.pi/2.0)  # non-linear rescaling of relative deviations
        temp = RGBim * wt + a*(1-wt) + a*(1-wt)**2 * RGBim  # multiply by weights in complicated fashion
        rgb = np.zeros((h, w, 3), np.float32)  # reset rgb to be blank
        for idx, im in enumerate((temp[0, :, :], temp[1, :, :], temp[2, :, :])):  # fill rgb with weight-rescaled rgb
            rgb[:, :, idx] = im

    clipped = np.clip(rgb, 0., 1.)  # set max/min to 0 and 1

    return clipped

def nonlinear_map(x, arcsinh=1.):
    """
    Apply non-linear map to input matrix. Useful to rescale telescope pixels for viewing.
    Args:
        x (np.array): array to have map applied
        arcsinh (np.float):
    Returns:
        (np.array) array with map applied
    """
    return np.arcsinh(x * arcsinh)

def ndarray_from_photon_counts(photon_array, initial_array, bands='grz'):
    """
    Reverses the process done by lupton_rgb to return an ndarray capable of being converted
    into a png from an initally photon count data set.

    inputs:
        photon_array - (x,x) array of ints for number 
        initial_array -
    """
    photon_energy = 600. * 1e-9
    exposure_time_seconds = 90. * 3.

    photon_array = np.transpose(photon_array) #need to reshape the array to (3, x, x) from (x, x, 3)
    photons_per_band =  photon_array * ratio_of_bands(initial_array)#Split back into bands (3) consider ratios?
    image_flux_back = (photon_energy * photons_per_band)/exposure_time_seconds #Find the flux from the number of photons
    image_absolute_mag_back = -2.5 * np.log10(image_flux_back/3631) #Absolute mag from flux
    image_nano_mag_back = np.power(10, (image_absolute_mag_back - 22.5)/(-2.5))#Nanomaggies from magnitude 

    size = initial_array[0].shape[0]
    grzscales = dict(g=(2, 0.00526),
                     r=(1, 0.008),
                     z=(0, 0.0135))

    returned_image = np.zeros((size, size, 3), np.float32)
    for im, band in zip(image_nano_mag_back, bands):
        plane, scale = grzscales.get(band, (0, 1.))
        returned_image[:, :, plane] = (im*scale).astype(np.float32)

    return returned_image

def ratio_of_bands(imgs):
    """
    Find the ratio in plane for each pixel

    inputs:
        imgs - (3, x, x) numpy array

    outputs
        (3, x, x) array of the ratio for each pixel
    """
    return imgs/np.sum(imgs, axis=0)

if __name__ == "__main__":
    print('hello')
    sys.exit()