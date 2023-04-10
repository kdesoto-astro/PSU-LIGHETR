import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import os, glob, sys
from os import listdir, makedirs
from os.path import isfile, join

from astropy.io import fits
from astropy.table import Table, hstack
from scipy.optimize import curve_fit

np.set_printoptions(threshold=sys.maxsize)
H_0 = 70. # km/s/Mpc
c = 3e5 # km/s

HETDEX_SPECS_FN = "refined_hetdex_spec.fits"

def get_all_spec_files(fn_dir):
    """
    Return all spectral files for LIGO from specific
    directory.
    """
    return [join(fn_dir, f) for f in listdir(fn_dir) if (('spec' in f) & isfile(join(fn_dir, f)))]

def import_kn_spec_file(fn, t_lim=1e5, keep_frac=0.001):
    """
    Imports LIGO spectral file, with (t, viewing angle) pair
    as key and [wavelength arr, spectral arr] as values.
    """
    time_arr = []
    with open(fn, 'r') as fp:
        for l_no, line in enumerate(fp):
            # search string
            if '#' in line:
                time_arr.append(float(line.split()[-1]))

    data_dict = {}
    data = np.loadtxt(fn, comments='#')
    data = data.reshape(-1,1024,56)
    for t_idx in np.arange(len(data),dtype=int):
        t = time_arr[t_idx]
        if t >= t_lim:
            continue
        wv_left = data[t_idx, 1:, 0]
        wv_right = data[t_idx, 1:, 1]
        wv_center = (wv_left + wv_right) / 2.
        for angle in np.arange(2, np.shape(data)[-1],dtype=int):
            if np.random.rand() > keep_frac:
                continue
            # flux is at at R=10pc [erg/(s*Angstrom*cm)]
            data_dict[(t, angle)] = np.vstack([wv_center, data[t_idx,1:,angle]])
            #print(data_dict[(t, angle)])
    return data_dict

def calc_redshift(d):
    """
    Calculates redshift at a given distance d (in Mpc).
    Assumes c / H_0 >> d
    """
    return H_0 * d / c

def calc_distance(z):
    return c*z / H_0

def redshift_spectrum(wavelengths, spectrum, \
                      redshift_new, orig_distance=10.): # distances in pc
    """
    Redshifts the spectrum based on source being a certain
    distance away. Takes original distance to de-redshift
    and re-redshift the wavelengths.
    """
    redshift_orig = calc_redshift(orig_distance * 1e-6)
    #redshift_new = calc_redshift(distance * 1e-6)
    shifted_wv = wavelengths * (1. + redshift_new) / (1. + redshift_orig)
    scaled_lum = spectrum * (1. + redshift_orig) * orig_distance**2 # ignore constant factors
    scaled_flux = scaled_lum / (1. + redshift_new) / calc_distance(redshift_new)**2
    return shifted_wv, scaled_flux

def combine_kilonova_galaxy_spectrum(k_wv, k_fluxes, g_wv, g_fluxes_multi, g_z_multi, num=2):
    """
    Combine a galaxy and kilonova spectrum so that the redshifts align.
    Shifts to the galactic redshift. Applies corrections for multiple galaxies at once.
    """
    interped_k_fluxes = []
    idxs = np.random.randint(0,len(g_z_multi),num)
    for i in idxs:
        k_wv_shifted, k_flux_shifted = redshift_spectrum(k_wv, k_fluxes, g_z_multi[i])
        interped_k_fluxes.append(np.interp(g_wv, k_wv_shifted, k_flux_shifted))
    return g_wv, np.array(interped_k_fluxes) + g_fluxes_multi[idxs]

def import_gal_file_npy(fn):
    """
    Import galaxy spectrum file in numpy format.
    """
    npy_array = np.load(fn)
    return npy_array[0], npy_array[1], npy_array[2:20]

def refine_gal_specs(fn):
    """
    Import HETDEX spectral file. Returns wavelengths, spectra, and redshifts.
    """
    hdu = fits.open(fn)
    spec = hdu['SPEC'].data
    spec_err = hdu['SPEC_ERR'].data
    wave_rect = hdu['WAVELENGTH'].data
    z = np.array([x[7] for x in hdu['INFO'].data])
    
    w_z = ( z > 0.) & (z < 0.05)
    
    c1 = fits.Column(name='z', array=z[w_z], format='E')
    c2 = fits.Column(name='wv', array=wave_rect, format='E')
    c3 = fits.Column(name='flux', array=spec[w_z], format='1036E')
    
    coldefs = fits.ColDefs([c1, c2, c3])
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto('refined_hetdex_spec.fits')
    
    return z[w_z], wave_rect, spec[w_z]

def import_gal_specs(fn):
    hdu = fits.open(fn)
    data = hdu[1].data
    z = data['z']
    wv = data['wv']
    spec = data['flux']
    wv_shape = spec.shape[-1]
    return z, wv[:wv_shape], spec

def plot_redshifts(z):
    """
    Plot redshift distribution of sample. Overplots
    expected 1/(1+z)^3 trend
    """
    def z_curve_fit(z_arr, A):
        return A * (1+z_arr)**3
    
    n, bin_edges, _ = plt.hist(z, bins=50)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    popt, pcov = curve_fit(z_curve_fit, bin_centers, n)
    plt.plot(bin_centers, z_curve_fit(bin_centers, *popt), label="Expected Trend")
    plt.legend()
    plt.xlabel("Redshift")
    plt.ylabel("Count")
    plt.title("HETDEX Galaxy Distribution")
    plt.savefig("figs/z_dist_hetdex.pdf")
    plt.close()
    
def generate_samples_from_file_pair(kn_file, gal_file, out_folder, prefix):
    """
    Takes one kilonova file and one galaxy datafile and
    generates merged spectrum samples from them. Saves
    spectra to output folder.
    """
    makedirs(out_folder, exist_ok=True)
    kn_data_dict = import_kn_spec_file(kn_file, 2.)
    gal_z_all, gal_wv, gal_spec_all = import_gal_specs(gal_file)
    for k in kn_data_dict:
        kn_wv, kn_flux = kn_data_dict[k]
        combined_wv, combined_flux_all = combine_kilonova_galaxy_spectrum(kn_wv, kn_flux, \
                                                                     gal_wv, gal_spec_all, gal_z_all)
        for ct, f in enumerate(combined_flux_all):
            if len(f[np.isnan(f)]) > 0:
                continue
            save_suffix = str(prefix) + "_" + str(ct) + "_" + str(k[0]) + "_" + str(k[1])+".npz"
            np.savez_compressed(join(out_folder, save_suffix), np.column_stack((combined_wv, f)))
    
def plot_training_sample(fn):
    """
    Plots one combined spectrum used in the training sample.
    """
    npy_array = np.load(fn)
    true_arr = npy_array["arr_0"]
    plt.plot(true_arr[:,0], true_arr[:,1])
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.savefig("spectra/train_sample.png")
    plt.close()
    
def main():
    LIGO_DIR = "/gpfs/group/vav5084/default/ligo/kn_sim_cube_v1"
    SAVE_DIR = "/gpfs/group/vav5084/default/ligo/training_samples_kn_v2"
    
    ct = 0
    for kn_file in glob.glob(LIGO_DIR+"/*_spec_*.dat"):
        ct += 1
        print(ct)
        generate_samples_from_file_pair(kn_file, HETDEX_SPECS_FN, SAVE_DIR, ct)
        
    #kn_file = os.path.join(LIGO_DIR, "Run_TP_dyn_all_lanth_wind1_all_md0.001_vd0.05_mw0.001_vw0.05_spec_2020-03-19.dat")
    

if __name__ == "__main__":
    main()
    #z, wv, spec = import_gal_specs(HETDEX_SPECS_FN)
    #plot_redshifts(z)
    #plot_training_sample(test_fn)