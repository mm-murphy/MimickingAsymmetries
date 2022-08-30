## Importing the packages we'll use
import numpy as np              # numerical python
import matplotlib.pyplot as plt # plotting
import batman                   # homogeneous limb transit modeling
import catwoman                 # asymmetric limb transit modeling
import emcee                    # MCMC sampling package
from multiprocessing import Pool # easy multiprocessing with emcee
from HandyFunctions import *
import os, sys 

## Define the output file and destiations for outputting figures
output_file_path = './run1_synth_scatter250.txt'
figure_output_path = './figures/synth_scatter250/'

# create the file/directory if it doesn't exist already
output_file_exists = os.path.isfile(output_file_path)
if output_file_exists:
    print('these results already exist')
    print('option to not rewrite is not yet created')
    os.remove(output_file_path)
    with open(output_file_path, 'w') as f: pass # creates an empty output file
else:
    with open(output_file_path, 'w') as f: pass # creates an empty output file

fig_dir_exists = os.path.isdir(figure_output_path)
if not fig_dir_exists:
    os.mkdir(figure_output_path)

## toggle to send all print statements to the output file
print_all_to_file = True
if print_all_to_file:
    sys.stdout = open(output_file_path, 'wt')


print('Running program!')

### Setting our planet's parameters
# I'll use parameters for WASP-96 b
c = 1. / np.log(10.)  # useful constant for computing uncertainties of logarithmic quantities
lit_params = {
    # array order: 0 = value, 1 = uncertainty, 2 = unit, 3 = source
    't0':np.array([2459111.30170, 0.00031, 'days', 'Hellier+ 2014'],dtype=object),
    'P':np.array([3.4252565, 0.0000008, 'days', 'Kokori+ 2022'], dtype=object),
    'log10P':np.array([np.log10(3.4252565), ((c*0.0000008)/3.4252565), 'unitless', 'calculated'], dtype=object),
    'a':np.array([9.03, 0.3, 'Rs', 'Patel & Espinoza 2022'], dtype=object),
    'log10a':np.array([np.log10(9.03), ((c*0.3)/ 9.03), 'unitless', 'calculated'], dtype=object),
    'RpRs':np.array([0.1186, 0.0017, 'unitless', 'Patel & Espinoza 2022'], dtype=object),
    #'RpRs':np.array([0.1158, 0.0017, 'unitless', 'pre determined for this band'], dtype=object),
    'Rs':np.array([1.15, 0.03, 'Rsun', 'Gaia DR2'], dtype='object'),
    'Mp':np.array([0.49, 0.04, 'Mjupiter', 'Bonomo+ 2017'], dtype='object'),
    'Teq':np.array([1285., 40., 'K', 'Hellier+ 2014'], dtype='object'),
    'inc':np.array([85.6, 0.2, 'degrees', 'Hellier+ 2014'], dtype=object),
    'cosi':np.array([np.cos(85.6*(np.pi/180.)), np.sin(85.6*(np.pi/180.))*(0.2*(np.pi/180.)), 'unitless', 'calculated'], dtype=object),
    'u1':np.array([0.1777, 0.5, 'unitless', 'Claret+ 2011 tabulation'], dtype=object), # note: this uncertainty set arbitrarily
    'u2':np.array([0.2952, 0.5, 'unitless', 'Claret+ 2011 tabulation'], dtype=object) # note: this uncertainty set arbitrarily
}

### Load in data OR set up synthetic observations
##      (synthetic observations will start here but finish being set up later in the code)
loadData = False
if loadData:
    ...
    print('observed data loaded')
    # haven't written this yet
else:
    # if not, let's create a data set
    # it will be a single transit set some N periods ahead of the literature transit time, given above
    t0_true = lit_params['t0'][0]  # literature transit time [day]
    P_true = lit_params['P'][0]    # literature period [day]
    Ntransits_ahead = 10 # N transits ahead of lit. transit time to place our new data
                         # this will factor into the ephemeris uncertainty, as it grows with sqrt(N) [i think]
    t0_new_true = t0_true + Ntransits_ahead*P_true  # 'True' propagated transit time, just set to prev. measured ephemeris propagated forward
    t0_new_guess = t0_true + Ntransits_ahead*P_true # our guess of what the new transit time would be, based on prev. measured ephemeris
    t0_new_guess_uncertainty = np.sqrt( (lit_params['t0'][1]**2) + (Ntransits_ahead**2)*(lit_params['P'][1]**2))
    obs_window_size = 3. # [hours] before/after the transit midpoint to generate a model for
    Ndatapoints = 350    # number of light curve points
    # generate time axis
    time = np.linspace(t0_new_true-obs_window_size/24., t0_new_true+obs_window_size/24., Ndatapoints)
    # initialize arrays for the flux and flux uncertainty, which will be set later on
    syn_fluxes, syn_errs = np.ones(time.shape), np.ones(time.shape)
    scatter = 250. # [ppm], standard deviation of flux values about the model
    flux_uncertainty = scatter # [ppm], uncertainty on each flux point
    print('synthetic observed data initialized')

### Generating the intrinsic asymmetry model
## First, we need to define what asymmetry factor to use (or what range of values to use)
##   note: as of right now, the factor is defined the number of scale heights by which the two radii differ
asymmetry_factors_totest = np.array([5., 10., 15., 20., 25., 30., 35., 40., 45., 50.])
print('Testing asymmetry factors: ', asymmetry_factors_totest)

# create array of trailing limb RpRs vals (always the same value = the literature value)
rprs1_vals = np.ones(asymmetry_factors_totest.shape) * lit_params['RpRs'][0] 
# array of leading limb RpRs vals
rprs2_vals = np.zeros(asymmetry_factors_totest.shape)
# to calculate these ...
for i_factor, asymfactor in enumerate(asymmetry_factors_totest):
    # compute trailing limb radius in [Rjupiter]
    this_rp1 = convert_rprs_to_rpJ(rprs1_vals[i_factor], lit_params['Rs'][0]) 
    # compute corresponding scale height in [km]
    this_H1 = calc_scale_height(lit_params['Teq'][0], lit_params['Mp'][0], this_rp1, mm=2.5)
    # convert this scale height into [Rjupiter]
    this_H1_rJ = convert_km_to_rpJ(this_H1)
    # using the pre-defined asymmetry factor, compute the leading limb's radius in [Rjupiter]
    this_rp2 = this_rp1 + (asymfactor * this_H1_rJ)
    # convert this to an Rp/Rs ratio
    this_rprs2 = convert_rpJ_to_rprs(this_rp2, lit_params['Rs'][0])
    rprs2_vals[i_factor] = this_rprs2
print('Limb radii calculated')
    
## Creating an initialized CATWOMAN model for the asymmetric limb transit
# note - can change the rp2 argument later on to regenerate new asymmetry factor models
#        using this same initialized environment
InitAsymParams = catwoman.TransitParams()
InitAsymParams.t0 = t0_new_true              # true current transit midpoint in [day]
InitAsymParams.per = P_true                  # orbital period in [day]
InitAsymParams.rp = rprs1_vals[0]            # trailing limb RpRs
InitAsymParams.rp2 = rprs2_vals[0]           # leading limb RpRs
InitAsymParams.a = lit_params['a'][0]        # semi-major axis in [Rsol]
InitAsymParams.inc = lit_params['inc'][0]    # inclination in [deg]
InitAsymParams.ecc = 0.                      # eccentricity
InitAsymParams.w = 90.                       # argument of periastron?
InitAsymParams.u = [lit_params['u1'][0], lit_params['u2'][0]]  # limb darkening coefficients
InitAsymParams.phi = 90.                     # 90 - obliquity [deg]
InitAsymParams.limb_dark = 'quadratic'        # type of limb darkening law to use
InitAsymModel = catwoman.TransitModel(InitAsymParams, time)
init_asym_lc = InitAsymModel.light_curve(InitAsymParams)
print('True asymmetric model initialized')

## Creating an initialized BATMAN model environment for the homogeneous limb transit
InitHomogParams = batman.TransitParams()
InitHomogParams.t0 = t0_new_guess # transit midpoint in [day]
InitHomogParams.per = lit_params['P'][0] # orbital period in [day]
InitHomogParams.rp = np.mean((rprs1_vals[0], rprs2_vals[0]))
InitHomogParams.a = lit_params['a'][0] # semi-major axis in [Rsol]
InitHomogParams.inc = lit_params['inc'][0] # inclination in [deg]
InitHomogParams.ecc = 0. # eccentricity
InitHomogParams.w = 90. # argument of periastron?
InitHomogParams.u = [lit_params['u1'][0], lit_params['u2'][0]]  # limb darkening coefficients
InitHomogParams.limb_dark = 'quadratic' # type of limb darkening law to use
InitHomogModel = batman.TransitModel(InitHomogParams, time)
init_homog_lc = InitHomogModel.light_curve(InitHomogParams)
print('Homogeneous limb fitting model initialized')


### Setting up the MCMC fitting
## In this script, I'll only fit for the transit time
## and use its uniform prior bounds as our measure of the requisite precision
## Defining the info dict
fit_pars = {
    'Init':{
        # Initialization values
        't0':t0_new_guess#,
    #    'log10P':lit_params['log10P'][0]
    },
    'Prior':{
        # Bayesian priors
        # 0 = prior value or prior bounds if type = uniform, 1 = prior error (also initialization ball size), 2 = prior type
        't0':np.array([(t0_new_guess - t0_new_guess_uncertainty, t0_new_guess + t0_new_guess_uncertainty), lit_params['t0'][1], 'U'], dtype=object)#,
    #    'log10P':np.array([(lit_params['log10P'][0]-3.*lit_params['log10P'][1], lit_params['log10P'][0]+3.*lit_params['log10P'][1]), lit_params['log10P'][1], 'U'], dtype=object),
    }
}



def homog_transit_model(theta, InitModel, rprs1, rprs2):
    """
    Input to the MCMC. Given a set of parameters (theta) and an initialized BATMAN environment (InitModel),
        outputs the model lightcurve array\
    Note: Will need to adjust this function when changing what parameters are free in the MCMC
    """
    Params = batman.TransitParams()
    Params.t0 = theta[0]
    Params.per = lit_params['P'][0]
    Params.rp = np.mean((rprs1, rprs2))
    Params.a = lit_params['a'][0]
    Params.inc = lit_params['inc'][0]
    Params.ecc = 0.
    Params.w = 90.
    Params.u = [lit_params['u1'][0], lit_params['u2'][0]]  # these are set arbitrarily
    Params.limb_dark = 'quadratic'
    step_lc = InitModel.light_curve(Params)
    return step_lc

def lnPosterior(theta, flux, flux_errors, info_dict, init_transitmodel, rprs1, rprs2):
    """
    Input to the MCMC. Computes the Bayesian posterior
    """
    # compute and check priors
    lnPrior = logPriors(theta, info_dict)
    if not np.isfinite(lnPrior):
        return -np.inf

    # compute transit model and resulting likelihood
    transit_model_lc = homog_transit_model(theta, init_transitmodel, rprs1, rprs2)
    lnLikelihood = logLikelihood(flux, flux_errors, transit_model_lc)

    # compute posterior
    lnPost = lnPrior + lnLikelihood

    return lnPost


print('Entering MCMC loop ...')
## Now going into the script loop
for i_factor, asym_factor in enumerate(asymmetry_factors_totest):
    print('Testing asymmetry factor %.0f'%(asym_factor))
    this_rprs1, this_rprs2 = rprs1_vals[i_factor], rprs2_vals[i_factor]
    this_rpJ1 = convert_rprs_to_rpJ(this_rprs1, lit_params['Rs'][0])
    this_rpJ2 = convert_rprs_to_rpJ(this_rprs2, lit_params['Rs'][0])
    print('This trailing limb radius: RpRs = %.5f; Rp = %.3f RJupiter'%(this_rprs1, this_rpJ1))
    print('This leading limb radius: RpRs = %.5f; Rp = %.3f RJupiter'%(this_rprs2, this_rpJ2))
    print('Fitting for the transit time')
    t0_uncert_minutes = t0_new_guess_uncertainty * 24. * 60.
    print('t0 uncertainty (after propagation) = %.3f minutes'%(t0_uncert_minutes))
    
    # set up initial parameter array and walkers
    Nparams = len(fit_pars['Init'].keys())
    theta_init = np.zeros(Nparams)
    theta_init_errs = np.zeros(Nparams)
    for i, key in enumerate(fit_pars['Init'].keys()):
        theta_init[i] = fit_pars['Init'][key]
        theta_init_errs[i] = fit_pars['Prior'][key][1]
    print('Initial parameter array = ', theta_init)
    print('Bounds: ', t0_new_guess, ' +/- ', t0_new_guess_uncertainty)
    
    Nwalkers = 3*Nparams
    Nsteps = 30000
    Nburn = 1000
    pos = np.zeros((Nwalkers, Nparams))
    for j in range(Nparams):
        pos[:,j] = theta_init[j] + 0.5*np.random.normal(0., 0.5*theta_init_errs[j], Nwalkers)
    print(pos)
    
    # generate the true asymmetric model with these radii
    InitAsymParams.rp = this_rprs1 # trailing limb RpRs
    InitAsymParams.rp2 = this_rprs2 # leading limb RpRs
    this_true_asym_lc = InitAsymModel.light_curve(InitAsymParams)
    
    # generate the synthetic data
    if not loadData:
        for i_point, time_value in enumerate(time):
            # get the model value at this point
            model_val = this_true_asym_lc[i_point]
            # set the data point somewhere in a normal distribution about the model value
            flux_val = model_val + np.random.normal(loc=0, scale=(scatter/1.e6))
            syn_fluxes[i_point] = flux_val
            syn_errs[i_point] = (flux_uncertainty / 1.e6)
    
    # generate the initial homog light curve
    init_homog_lc = homog_transit_model(theta_init, InitHomogModel, this_rprs1, this_rprs2)
    
    # compute initial statistics
    print('N parameters = ', Nparams)
    initial_lnPrior = logPriors(theta_init, fit_pars)
    initial_lnPost = lnPosterior(theta_init, syn_fluxes, syn_errs, fit_pars, InitHomogModel, this_rprs1, this_rprs2)
    initial_lnLikelihood = initial_lnPost - initial_lnPrior
    print('Initial ln Prior = ', initial_lnPrior)
    print('Initial ln Likelihood = ', initial_lnLikelihood)
    print('Initial ln Posterior = ', initial_lnPost)
    
    # run the MCMC
    print('Running for %d steps, including %d step burn-in'%(Nsteps, Nburn))
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Nparams, lnPosterior, pool=pool, 
                                     args=(syn_fluxes, syn_errs, fit_pars, InitHomogModel, this_rprs1, this_rprs2))
        state = sampler.run_mcmc(pos, Nsteps, progress=True)
    print(' ... complete')
    # grab the outputs
    samples = sampler.get_chain(discard=Nburn)
    flatsamples = sampler.get_chain(discard=Nburn, flat=True)
    loglikelihoods = sampler.get_log_prob(discard=Nburn, flat=True)
    autocorrtimes = sampler.get_autocorr_time()
    print('sampler outputs grabbed')


    param_fits = np.asarray([np.median(flatsamples[:,i]) for i in range(samples.shape[2])])
    param_uperrs = np.asarray([np.percentile(flatsamples[:,i], 84) for i in range(samples.shape[2])]) - param_fits
    param_loerrs = param_fits - np.asarray([np.percentile(flatsamples[:,i], 16) for i in range(samples.shape[2])])
    param_errs = np.mean((param_uperrs, param_loerrs), axis=0)

    print('For the best fit Homogeneous model:')
    print('Best fit parameters = ', param_fits)
    print('errors = ', param_errs)
    print('True t0 was = ', t0_new_true)
    sigdiff = abs(param_fits[0] - t0_new_true)/param_errs[0]
    print('%.2f sigma difference'%(sigdiff))
    print('auto correlation times = ', autocorrtimes)
    bf_lnPrior = logPriors(param_fits, fit_pars)
    bf_lnPost = lnPosterior(param_fits, syn_fluxes, syn_errs, fit_pars, InitHomogModel, this_rprs1, this_rprs2)
    bf_model = homog_transit_model(param_fits, InitHomogModel, this_rprs1, this_rprs2)
    bf_lnLikelihood = logLikelihood(syn_fluxes, syn_errs, bf_model)
    bf_bic = compute_bic(len(param_fits), len(syn_fluxes), bf_lnLikelihood)
    bf_chi2red = compute_chi2(syn_fluxes, syn_errs, bf_model, reduced=True, Ndof=(len(time) - Nparams))


    data_asym_residuals = syn_fluxes - this_true_asym_lc
    mean_daresidual = np.mean(data_asym_residuals) 
    mean_abs_daresidual = np.mean(abs(data_asym_residuals))
    data_homog_residuals = syn_fluxes - bf_model
    mean_dhresidual = np.mean(data_homog_residuals)
    mean_abs_dhresidual = np.mean(abs(data_homog_residuals))
    asym_homog_residuals = this_true_asym_lc - bf_model
    mean_ahresidual = np.mean(asym_homog_residuals)
    mean_abs_ahresidual = np.mean(abs(asym_homog_residuals))

    print('Final ln Prior = ', bf_lnPrior)
    print('Final ln Likelihood = ', bf_lnLikelihood)
    print('Final ln Posterior = ', bf_lnPost)
    print('Final BIC = ', bf_bic)
    print('Final reduced chi2 = ', bf_chi2red)
    print('\n')

    print('For the true asymmetric model:')
    true_lnLikelihood = logLikelihood(syn_fluxes, syn_errs, this_true_asym_lc)
    true_bic = compute_bic(len(param_fits), len(syn_fluxes), true_lnLikelihood)
    true_chi2red = compute_chi2(syn_fluxes, syn_errs, this_true_asym_lc, reduced=True, Ndof=len(time))
    print('True ln Likelihood = ', true_lnLikelihood)
    print('True BIC = ', true_bic)
    print('True reduced chi2 = ', true_chi2red)
    print('\n')

    print('Mean Asym. Model - Homog. Model residual = %.0f ppm'%(1.e6*mean_ahresidual))
    print('    mean abs. above = %.0f ppm'%(1.e6*mean_abs_ahresidual))
    print('Mean Syn. Data - Homog. Model residual = %.0f ppm'%(1.e6*mean_dhresidual))
    print('    mean abs above = %.0f ppm'%(1.e6*mean_abs_dhresidual))
    print('Mean Syn. Data - Asym. Model residual = %.0f ppm'%(1.e6*mean_daresidual))
    print('    mean abs above = %.0f ppm'%(1.e6*mean_abs_daresidual))
    print('Mean Syn. Data Uncertainty = %.0f ppm'%(1.e6*np.mean(syn_errs)))
    print('Flux Scatter = %.0f ppm'%(scatter))

    ## make figure showing the lightcurves and residuals
    fig1, ax1 = plt.subplots(figsize=(10,6), nrows=2, sharex=True)
    plt.subplots_adjust(hspace=0.15)
    ax10, ax11 = ax1[0], ax1[1]
    # plotting the lightcurves:
    ax10.plot(time, this_true_asym_lc, c='green', lw=1, label='True Asym. Limb Model')
    ax10.plot(time, bf_model, c='blue', lw=1, label='B.F. Homog. Limb Model')
    ax10.errorbar(time, syn_fluxes, syn_errs, marker='o', ls='None', ms=2, c='black', label='Synth. Obs. Data')
    ax10.set(ylabel='Rel. Flux')
    ax10.legend(loc='lower right', fontsize=8)
    # plotting the residuals
    ax11.axhline(0., c='gray', lw=0.5, alpha=0.25)
    ax11.plot(time, 1.e6*asym_homog_residuals, lw=1, c='black', label='Asym. Model - Homog. Model')
    ax11.scatter(time, 1.e6*data_asym_residuals, c='green', s=2, label='Data - Asym. Model')
    ax11.scatter(time, 1.e6*data_homog_residuals, c='blue', s=2, label='Data - Homog. Model')
    ax11.set(xlabel='Time', ylabel='Residual [ppm]')
    ax11.legend(loc='lower right', fontsize=8)
    fig_name = 'lcplot_asym'+ str(int(asym_factor)) + '.png'
    plt.savefig(figure_output_path+fig_name, bbox_inches='tight')
    plt.close(fig1)

    ## make figure showing the distribution of t0 samples
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.axvline(t0_new_true, c='black', label='Truth')
    ax2.axvline(t0_new_guess - t0_new_guess_uncertainty, c='black', ls='--', label='Bounds')
    ax2.axvline(t0_new_guess + t0_new_guess_uncertainty, c='black', ls='--')
    ax2.hist(flatsamples, color='blue', edgecolor='black')
    ax2.set(xlabel='t0')
    ax2.yaxis.set_visible(False)
    ax2.legend(loc='upper right', fontsize=8)
    fig2_name = 'distplot_asym' + str(asym_factor) + '.png'
    plt.savefig(figure_output_path+fig2_name, bbox_inches='tight')
    plt.close(fig2)

    print('\n')
    print('---------------------')
    print('\n')





print('ran fine')
