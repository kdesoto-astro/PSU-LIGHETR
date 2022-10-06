import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import os, glob, sys
from os import listdir, makedirs
from os.path import isfile, join
np.set_printoptions(threshold=sys.maxsize)
H_0 = 70. # km/s/Mpc
c = 3e5 # km/s

def get_all_spec_files(fn_dir):
    """
    Return all spectral files for LIGO from specific
    directory.
    """
    return [join(fn_dir, f) for f in listdir(fn_dir) if (('spec' in f) & isfile(join(fn_dir, f)))]

def import_kn_spec_file(fn, t_lim=1e5):
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

def combine_kilonova_galaxy_spectrum(k_wv, k_fluxes, g_wv, g_fluxes_multi, g_z_multi):
    """
    Combine a galaxy and kilonova spectrum so that the redshifts align.
    Shifts to the galactic redshift.
    """
    interped_k_fluxes = []
    for i in range(len(g_z_multi)):
        k_wv_shifted, k_flux_shifted = redshift_spectrum(k_wv, k_fluxes, g_z_multi[i])
        interped_k_fluxes.append(np.interp(g_wv, k_wv_shifted, k_flux_shifted))
    return g_wv, np.array(interped_k_fluxes) + g_fluxes_multi

def import_gal_file_npy(fn):
    """
    Import galaxy spectrum file in numpy format.
    """
    npy_array = np.load(fn)
    return npy_array[0], npy_array[1], npy_array[2:20]

def generate_samples_from_file_pair(kn_file, gal_file, out_folder, prefix):
    """
    Takes one kilonova file and one galaxy datafile and
    generates merged spectrum samples from them. Saves
    spectra to output folder.
    """
    makedirs(out_folder, exist_ok=True)
    kn_data_dict = import_kn_spec_file(kn_file, 2.)
    gal_z_all, gal_wv, gal_spec_all = import_gal_file_npy(gal_file)
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
    
LIGO_DIR = "/gpfs/group/vav5084/default/ligo/kn_sim_cube_v1"
SAVE_DIR = "/gpfs/group/vav5084/default/ligo/training_samples_v1"
GAL_TEST_FILE = "/gpfs/group/vav5084/default/ligo/VIRUS_spectra.npy"
#AL_Z_FILE = "kaylee_zs.npy"
#GAL_TEST_Z = np.load(GAL_Z_FILE)[:20]

#GAL_TEST_Z = np.random.uniform(0.0, 0.05, 19)
kn_file = get_all_spec_files(LIGO_DIR)[0]
generate_samples_from_file_pair(kn_file, GAL_TEST_FILE, SAVE_DIR, "test")

#plot_training_sample(glob.glob(SAVE_DIR + "/*.npz")[0])