from astropy.io import fits
import numpy as np
from astropy.modeling.models import Moffat2D
import glob, os
import matplotlib.pyplot as plt
from petrofit import fit_model, plot_fit, print_model_params

def combine_image_cubes(galaxy_fn, kilonova_fn):
    k = fits.open(galaxy_fn)
    x = (np.arange(k[0].header['NAXIS1'])*k[0].header['CDELT1'] +
             k[0].header['CRVAL1']) #
    y = (np.arange(k[0].header['NAXIS2'])*k[0].header['CDELT2'] +
             k[0].header['CRVAL2'])
    w = (np.arange(k[0].header['NAXIS3'])*k[0].header['CDELT3'] +
             k[0].header['CRVAL3']) # wavelengths
    xgrid, ygrid = np.meshgrid(x, y)

    seeing = 1.8 # seeing in arcseconds for the observation (I can get you these values, but it can start as a guess first)
    xc, yc = (np.mean(x), np.mean(y)) # get center coordinate (where target should be)
    data = k[0].data # the galaxy image cube
    
    # generate an image shape for the kilonova explosion
    M = Moffat2D(x_0=xc, y_0=yc, alpha=3.5)
    M.gamma.value = 0.5 * seeing / np.sqrt(2**(1./ M.alpha.value) - 1.)
    source_image = M(xgrid, ygrid)
    
    #import associated spectrum with kilonova
    input_spectrum = import_kilonova_spectrum(kilonova_fn)
    input_spectrum = np.interp(w, native_wave, native_flux) # rectify the input to the w solution
    
    #combines galaxy + kilonova image cubes
    simulated_cube = data + source_image[:, :, np.newaxis] * input_spectrum[np.newaxis, np.newaxis, :]
    return simulated_cube

def plot_image_cube(galaxy_fn):
    k = fits.open(galaxy_fn)
    x = (np.arange(k[0].header['NAXIS1'])*k[0].header['CDELT1'] +
             k[0].header['CRVAL1']) #
    y = (np.arange(k[0].header['NAXIS2'])*k[0].header['CDELT2'] +
             k[0].header['CRVAL2'])
    w_arr = (np.arange(k[0].header['NAXIS3'])*k[0].header['CDELT3'] +
             k[0].header['CRVAL3']) # wavelengths
    xgrid, ygrid = np.meshgrid(x, y)
    
    data = k[0].data
    
    prefix = galaxy_fn.split("/")[-1].split(".")[0]
    os.makedirs("figs/"+prefix, exist_ok=True)
    for e, w in enumerate(w_arr[::100]):
        plt.imshow(data[e], cmap="hot", extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.savefig("figs/"+prefix+"/"+str(w)+".pdf")
        plt.close()
    
    # collapse down spectrum into 1D around object
    fluxes_collapsed = np.sum(data, axis=0)
    fluxes_collapsed /= np.max(fluxes_collapsed)
    xc, yc = (len(xgrid)/2, len(ygrid)/2)
    M = Moffat2D(amplitude=1., x_0=xc,
                 y_0=yc, gamma=0.5*1.8 / np.sqrt(2**(1./ 3.5) - 1.),
                 alpha=3.5)
    fitted_model, fitter = fit_model(
        fluxes_collapsed,
        model = M,
        maxiter=10000,
        epsilon=1.4901161193847656e-08,
        acc=1e-09
    )
    print(fitted_model)
    axs, model_image, residual_image = plot_fit(fitted_model, fluxes_collapsed, return_images=True)
    #plt.imshow(model_image, cmap="hot", extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.savefig("figs/"+prefix+"/model.pdf")
    plt.close()
        
    # cut out source
    g = np.abs(fitted_model.gamma)
    x_min = round(fitted_model.x_0 - 1.*g)
    x_max = round(fitted_model.x_0 + 1.*g)
    y_min = round(fitted_model.y_0 - 1.*g)
    y_max = round(fitted_model.y_0 + 1.*g)
    
    image_cutout = fluxes_collapsed[y_min:y_max, x_min:x_max]
    print(y_min, y_max, x_min, x_max)
    print(image_cutout)
    plt.imshow(image_cutout, cmap="hot", extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.savefig("figs/"+prefix+"/collapsed_cutout.pdf")
    plt.close()
    
data_fn = glob.glob("/gpfs/group/vav5084/default/kdesoto/lighetr-data/*.fits")[-1]
plot_image_cube(data_fn)




