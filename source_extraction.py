import numpy as np

H_0 = 70. # km/s/Mpc
c = 3e5 # km/s
TEMPLATE_DIR = "/gpfs/group/vav5084/default/ligo/kn_sim_cube_v1"


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
            #print(data_dict
            
            
def calc_redshift(d):
    """
    Calculates redshift at a given distance d (in Mpc).
    Assumes c / H_0 >> d
    """
    return H_0 * d / c


def calc_distance(z):
    return c*z / H_0


def generate_template_and_params(fn):
    """
    Obtains the template wavelengths + spectra for a specific set of source parameters.
    
    fn_params include ejecta_type, wind_type (int), mass1, beta1, mass2, and beta2
    """
    data_dict = import_kn_spec_file(fn, t_lim=2.)
    fn_wo_path = fn.split("/")[-1]
    parsed_fn = fn.split("_")
    ejecta_type = parsed_fn[1]
    wind_type = int(parsed_fn[5][-1])
    mass1 = float(parsed_fn[7][2:])
    beta1 = float(parsed_fn[8][2:])
    mass2 = float(parsed_fn[9][2:])
    beta2 = float(parsed_fn[10][2:])
    
    source_params = [ejecta_type, wind_type, mass1, beta1, mass2, beta2]
    
    return data_dict, source_params
    

def shift_to_template_z(wvs, spectra, z):
    """
    Shifts true spectrum to match redshift of templates
    (10 pc)
    """
    z_new = calc_redshift(1e-5) # redshift corresponding to 10 pc
    orig_distance = calc_distance(z)
    shifted_wv = wavelengths * (1. + z_new) / (1. + z)
    scaled_lum = spectrum * (1. + z) * orig_distance**2 # ignore constant factors
    scaled_flux = scaled_lum / (1. + z_new) / (1e-5)**2
    return shifted_wv, scaled_flux


def calc_chisq(spec_obs, spec_template, snr):
    """
    Calculates the chi-squared value when comparing two
    sets of spectra, for template matching.
    """
    sigma = spec_obs / snr
    return np.sum((spec_template - spec_obs)**2 / sigma**2)


def get_nearest_template(wv, spectrum, redshift):
    """
    Return the best template given a kilonova spectrum
    and redshift.
    """
    best_chisq = np.inf
    best_s_params = None
    best_template = None
    wv_shifted, f_shifted = shift_to_template_z(wv, spectrum, redshift)
    all_templates = get_all_spec_files(TEMPLATE_DIR)
    for template_file in all_templates:
        data_dict, s_params = generate_template_and_params(template_file)
        for k in kn_data_dict:
            t, angle = k
            template_wv, template_flux = data_dict[k]
            template_flux_interped = np.interp(wv_shifted, template_wv, template_flux)
            chisq = calc_chisq(f_shifted, template_flux_interped, 5.)
            if chisq < best_chisq:
                best_chisq = chisq
                best_s_params = [*s_params, t, angle]
                best_template = template_file
    return best_s_params, best_template_file, best_chisq
                
            
            
    
    