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
## Parameters that are used in the code and also for naming:
runN = 1
Ntransits_ahead = 1000 # N transits ahead of lit. transit time to place our new data
                         # this will factor into the ephemeris uncertainty, as it grows with sqrt(N) [i think]
scatter = 100. # [ppm], standard deviation of flux values about the model
output_file_path = './output_files/gj1214b_synth_scatter'+str(int(scatter))+'_ahead'+str(int(Ntransits_ahead))+'LDscenA_nircCadence.txt'
figure_output_path = './figures/gj1214b_synth_scatter'+str(int(scatter))+'_ahead'+str(int(Ntransits_ahead))+'LDscenA_nircCadence/'
array_output_path = './output_arrays/gj1214b_synth_scatter'+str(int(scatter))+'_ahead'+str(int(Ntransits_ahead))+'LDscenA_nircCadence'

# print save locations to console
print('verbose output log will be saved to ', output_file_path)
print('figures will be saved to ', figure_output_path)
print('key results array will be saved to', array_output_path)




# create the file/directory if it doesn't exist already
output_file_exists = os.path.isfile(output_file_path)
rewrite = True
if output_file_exists:
    if rewrite:
        os.remove(output_file_path)
        with open(output_file_path, 'w') as f: pass # creates an empty output file
    else:
        runN += 1
        output_file_path = './run'+str(runN)+'_synth_scatter'+str(int(scatter))+'_ahead'+str(int(Ntransits_ahead))+'.txt'
        figure_output_path = './figures/synth_scatter'+str(int(scatter))+'_ahead'+str(int(Ntransits_ahead))+'/'
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
    't0':np.array([2455701.413328, 0.000066, 'days', 'Cloutier+ 2021'],dtype=object),
    'P':np.array([1.58040433, 0.00000013, 'days', 'Cloutier+ 2021'], dtype=object),
    'log10P':np.array([np.log10(1.58040433), ((c*0.00000013)/1.58040433), 'unitless', 'calculated'], dtype=object),
    'a':np.array([14.85, 0.16, 'Rs', 'Cloutier+ 2021'], dtype=object),
    'log10a':np.array([np.log10(14.85), ((c*0.16)/ 14.85), 'unitless', 'calculated'], dtype=object),
    'RpRs':np.array([0.1160, 0.0005, 'unitless', 'Berta+ 2012'], dtype=object),
    #'RpRs':np.array([0.1158, 0.0017, 'unitless', 'pre determined for this band'], dtype=object),
    'Rs':np.array([0.215, 0.008, 'Rsun', 'Cloutier+ 2021'], dtype='object'),
    'Mp':np.array([0.0257, 0.0014, 'Mjupiter', 'Cloutier+ 2021'], dtype='object'),
    'Teq':np.array([596, 19., 'K', 'Cloutier+ 2021'], dtype='object'),
    'inc':np.array([88.7, 0.16, 'degrees', 'Cloutier+ 2021'], dtype=object),
    'cosi':np.array([np.cos(88.7*(np.pi/180.)), np.sin(88.7*(np.pi/180.))*(0.16*(np.pi/180.)), 'unitless', 'calculated'], dtype=object),
    'u1':np.array([0.1777, 0.5, 'unitless', 'Claret+ 2011 tabulation'], dtype=object), # note: see below
    'u2':np.array([0.2952, 0.5, 'unitless', 'Claret+ 2011 tabulation'], dtype=object), # note: these vals are for another star!
    'T14':np.array([0.8688, 0.0029, 'hours', 'Berta+ 2012'], dtype=object)
}

## customize any parameters here:
# LD coeffs for scen B
#lit_params['u1'] = np.array([0.25, 0.5, 'unitless', 'custom'], dtype=object)
#lit_params['u2'] = np.array([0.45, 0.5, 'unitless', 'custom'], dtype=object)
# LD coeffs for scen C
#lit_params['u1'] = np.array([0.4, 0.5, 'unitless', 'custom'], dtype=object)
#lit_params['u1'] = np.array([0.6, 0.5, 'unitless', 'custom'], dtype=object)




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
 #   Ntransits_ahead = 10 # N transits ahead of lit. transit time to place our new data
 #                        # this will factor into the ephemeris uncertainty, as it grows with sqrt(N) [i think]
    t0_new_true = t0_true + Ntransits_ahead*P_true  # 'True' propagated transit time, just set to prev. measured ephemeris propagated forward
    t0_new_guess = t0_true + Ntransits_ahead*P_true # our guess of what the new transit time would be, based on prev. measured ephemeris
    t0_new_guess_uncertainty = np.sqrt( (lit_params['t0'][1]**2) + (Ntransits_ahead**2)*(lit_params['P'][1]**2))
    obs_window_size = 2.*lit_params['T14'][0] # [hours] before/after the transit midpoint to generate a model for
    #Ndatapoints = 350    # number of light curve points
    tint=17.5
    # generate time axis
#    time = np.linspace(t0_new_true-obs_window_size/24., t0_new_true+obs_window_size/24., Ndatapoints)
    time = np.arange((t0_new_true-obs_window_size/24.), (t0_new_true + obs_window_size/24.), (tint/60./60./24.))
    idxs_intransit = np.where((time >= (t0_new_true - 0.5*lit_params['T14'][0]/24.)) & (time <= (t0_new_true + 0.5*lit_params['T14'][0]/24.)))[0]
    # initialize arrays for the flux and flux uncertainty, which will be set later on
    syn_fluxes, syn_errs = np.ones(time.shape), np.ones(time.shape)
#    scatter = 250. # [ppm], standard deviation of flux values about the model
    flux_uncertainty = scatter # [ppm], uncertainty on each flux point
    print('synthetic observed data initialized')

### Generating the intrinsic asymmetry model
## First, we need to define what asymmetry factor to use (or what range of values to use)
##   note: as of right now, the factor is defined the number of scale heights by which the two radii differ
asymmetry_factors_totest = np.array([1., 3., 5., 7.5, 10., 15., 20., 25., 30., 35., 40., 45., 50.])
print('Testing asymmetry factors: ', asymmetry_factors_totest)

print('parameters being used:')
for key in lit_params:
    print(key, lit_params[key])

# create array of trailing limb RpRs vals (always the same value = the literature value)
rprs1_vals = np.ones(asymmetry_factors_totest.shape) * lit_params['RpRs'][0] 
# array of leading limb RpRs vals
rprs2_vals = np.zeros(asymmetry_factors_totest.shape)
# to calculate these ...
for i_factor, asymfactor in enumerate(asymmetry_factors_totest):
    # compute trailing limb radius in [Rjupiter]
    this_rp1 = convert_rprs_to_rpJ(rprs1_vals[i_factor], lit_params['Rs'][0]) 
    # compute corresponding scale height in [km]
    this_H1 = calc_scale_height(lit_params['Teq'][0], lit_params['Mp'][0], this_rp1, mm=5.)
    # convert this scale height into [Rjupiter]
    this_H1_rJ = convert_km_to_rpJ(this_H1)
    # using the pre-defined asymmetry factor, compute the leading limb's radius in [Rjupiter]
    this_rp2 = this_rp1 + (asymfactor * this_H1_rJ)
    # convert this to an Rp/Rs ratio
    this_rprs2 = convert_rpJ_to_rprs(this_rp2, lit_params['Rs'][0])
    rprs2_vals[i_factor] = this_rprs2
#print('Limb radii calculated')
    
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
#print('True asymmetric model initialized')

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
#print('Homogeneous limb fitting model initialized')


### Setting up the MCMC fitting
## In this script, I'll only fit for the transit time
## and use its uniform prior bounds as our measure of the requisite precision
## Defining the info dict
fit_pars = {
    'Init':{
        # Initialization values
        't0':t0_new_guess
    },
    'Prior':{
        # Bayesian priors
        # 0 = prior value or prior bounds if type = uniform, 1 = prior error (also initialization ball size), 2 = prior type
        't0':np.array([(t0_new_guess - 100*t0_new_guess_uncertainty, t0_new_guess + 100*t0_new_guess_uncertainty), lit_params['t0'][1], 'U'], dtype=object)
    }
}



def homog_transit_model(theta, InitModel, rprs):
    """
    Input to the MCMC. Given a set of parameters (theta) and an initialized BATMAN environment (InitModel),
        outputs the model lightcurve array\
    Note: Will need to adjust this function when changing what parameters are free in the MCMC
    """
    Params = batman.TransitParams()
    Params.t0 = theta[0]
    Params.per = lit_params['P'][0]
    Params.rp = rprs
    Params.a = lit_params['a'][0]
    Params.inc = lit_params['inc'][0]
    Params.ecc = 0.
    Params.w = 90.
    Params.u = [lit_params['u1'][0], lit_params['u2'][0]]  # these are set arbitrarily
    Params.limb_dark = 'quadratic'
    step_lc = InitModel.light_curve(Params)
    return step_lc

def lnPosterior(theta, flux, flux_errors, info_dict, init_transitmodel, rprs_homog):
    """
    Input to the MCMC. Computes the Bayesian posterior
    """
    # compute and check priors
    lnPrior = logPriors(theta, info_dict)
    if not np.isfinite(lnPrior):
        return -np.inf

    # compute transit model and resulting likelihood
    transit_model_lc = homog_transit_model(theta, init_transitmodel, rprs_homog)
    lnLikelihood = logLikelihood(flux, flux_errors, transit_model_lc)

    # compute posterior
    lnPost = lnPrior + lnLikelihood

    return lnPost

## defining arrays of quantities to save
t0_diff_seconds_arr = np.array([])
t0_diff_seconds_err_arr = np.array([])
chi2red_homog_arr = np.array([])
chi2red_asym_arr = np.array([])
fasym_arr = np.copy(asymmetry_factors_totest)

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
    print('t0 uncertainty (after propagation) = %f minutes'%(t0_uncert_minutes))
    
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
    Nsteps = 20000
    Nburn = 1000
    pos = np.zeros((Nwalkers, Nparams))
    for j in range(Nparams):
        pos[:,j] = theta_init[j] + 0.5*np.random.normal(0., 0.5*theta_init_errs[j], Nwalkers)
    print(pos)
    
    # generate the true asymmetric model with these radii
    InitAsymParams.rp = this_rprs1 # trailing limb RpRs
    InitAsymParams.rp2 = this_rprs2 # leading limb RpRs
    this_true_asym_lc = InitAsymModel.light_curve(InitAsymParams)
    rprs_homog = np.sqrt(np.mean([this_rprs1**2, this_rprs2**2]))
    
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
    init_homog_lc = homog_transit_model(theta_init, InitHomogModel, rprs_homog)
    
    # compute initial statistics
    print('N parameters = ', Nparams)
    initial_lnPrior = logPriors(theta_init, fit_pars)
    initial_lnPost = lnPosterior(theta_init, syn_fluxes, syn_errs, fit_pars, InitHomogModel, rprs_homog)
    initial_lnLikelihood = initial_lnPost - initial_lnPrior
    print('Initial ln Prior = ', initial_lnPrior)
    print('Initial ln Likelihood = ', initial_lnLikelihood)
    print('Initial ln Posterior = ', initial_lnPost)
    
    # run the MCMC
    print('Running for %d steps, including %d step burn-in'%(Nsteps, Nburn))
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalkers, Nparams, lnPosterior, pool=pool, 
                                     args=(syn_fluxes, syn_errs, fit_pars, InitHomogModel, rprs_homog))
        state = sampler.run_mcmc(pos, Nsteps, progress=True)
    #print(' ... complete')
    # grab the outputs
    samples = sampler.get_chain(discard=Nburn)
    flatsamples = sampler.get_chain(discard=Nburn, flat=True)
    loglikelihoods = sampler.get_log_prob(discard=Nburn, flat=True)
    autocorrtimes = sampler.get_autocorr_time()
    #print('sampler outputs grabbed')


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
    tdiff = param_fits[0] - t0_new_true # diff between b.f. time and true time in [day]
    print('time difference = %f minutes'%(tdiff*24.*60.))
    print('auto correlation times = ', autocorrtimes)
    bf_lnPrior = logPriors(param_fits, fit_pars)
    bf_lnPost = lnPosterior(param_fits, syn_fluxes, syn_errs, fit_pars, InitHomogModel, rprs_homog)
    bf_model = homog_transit_model(param_fits, InitHomogModel, rprs_homog)
    bf_lnLikelihood = logLikelihood(syn_fluxes, syn_errs, bf_model)
    bf_bic = compute_bic(len(param_fits), len(syn_fluxes), bf_lnLikelihood)
    bf_chi2red = compute_chi2(syn_fluxes, syn_errs, bf_model, reduced=True, Ndof=(len(time)))


    data_asym_residuals = syn_fluxes - this_true_asym_lc
    mean_daresidual, mean_daresidual_it = np.mean(data_asym_residuals), np.mean(data_asym_residuals[idxs_intransit])
    mean_abs_daresidual, mean_abs_daresidual_it = np.mean(abs(data_asym_residuals)), np.mean(abs(data_asym_residuals[idxs_intransit]))
    data_homog_residuals = syn_fluxes - bf_model
    mean_dhresidual, mean_dhresidual_it = np.mean(data_homog_residuals), np.mean(data_homog_residuals[idxs_intransit])
    mean_abs_dhresidual, mean_abs_dhresidual_it = np.mean(abs(data_homog_residuals)), np.mean(abs(data_homog_residuals[idxs_intransit]))
    asym_homog_residuals = this_true_asym_lc - bf_model
    mean_ahresidual, mean_ahresidual_it = np.mean(asym_homog_residuals), np.mean(asym_homog_residuals[idxs_intransit])
    mean_abs_ahresidual, mean_abs_ahresidual_it = np.mean(abs(asym_homog_residuals)), np.mean(abs(asym_homog_residuals[idxs_intransit]))
    
    max_daresidual = max(abs(data_asym_residuals))
    max_dhresidual = max(abs(data_homog_residuals))
    max_ahresidual = max(abs(asym_homog_residuals))

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
    lnLdiff = true_lnLikelihood - bf_lnLikelihood
    # try to avoid overflows when computing exp for bayes factor:
    if lnLdiff >= 100:
        bayes_factor = np.inf
    else:
        bayes_factor = np.exp(lnLdiff)
    # if bayes_factor <= 1 - homog model preferred or can't be ruled out
    print('Bayes factor = exp(lnL_asym - lnL_homog) =  ', bayes_factor)
    if bayes_factor >= 2.0:
        print('    this prefers the asymmetric model!')
        print('    so asymmetry may be retrieved given these parameters')
        print('    must manually decide whether this evidence is strong enough')
    elif (1. <= bayes_factor < 2.):
        print('    this prefers the asymmetric model')
        print('    evidence is only slight though')
    else:
        print('    homogeneous model cannot be ruled out')
        print('    asymmetry may not be retrieved given these parameters')
    print('\n')

    print('Full array residuals:')
    print('Mean Asym. Model - Homog. Model residual = %.0f ppm'%(1.e6*mean_ahresidual))
    print('    mean abs. above = %.0f ppm'%(1.e6*mean_abs_ahresidual))
    print('Mean Syn. Data - Homog. Model residual = %.0f ppm'%(1.e6*mean_dhresidual))
    print('    mean abs above = %.0f ppm'%(1.e6*mean_abs_dhresidual))
    print('Mean Syn. Data - Asym. Model residual = %.0f ppm'%(1.e6*mean_daresidual))
    print('    mean abs above = %.0f ppm'%(1.e6*mean_abs_daresidual))
    print('Mean Syn. Data Uncertainty = %.0f ppm'%(1.e6*np.mean(syn_errs)))
    print('Flux Scatter = %.0f ppm'%(scatter))
    print('\n')
    print('In-transit residuals:')
    print('Mean Asym. Model - Homog. Model residual = %.0f ppm'%(1.e6*mean_ahresidual_it))
    print('    mean abs. above = %.0f ppm'%(1.e6*mean_abs_ahresidual_it))
    print('Mean Syn. Data - Homog. Model residual = %.0f ppm'%(1.e6*mean_dhresidual_it))
    print('    mean abs above = %.0f ppm'%(1.e6*mean_abs_dhresidual_it))
    print('Mean Syn. Data - Asym. Model residual = %.0f ppm'%(1.e6*mean_daresidual_it))
    print('    mean abs above = %.0f ppm'%(1.e6*mean_abs_daresidual_it))
    print('Mean Syn. Data Uncertainty = %.0f ppm'%(1.e6*np.mean(syn_errs[idxs_intransit])))
    print('Flux Scatter = %.0f ppm'%(scatter))
    print('\n')
    print('Max residuals:')
    print('Max Asym. Model - Homog. Model residual = %.0f ppm'%(1.e6*max_ahresidual))
    print('Max Data - Asym. Model residual = %.0f ppm'%(1.e6*max_dhresidual))
    print('Max Data - Asym. Model residual = %.0f ppm'%(1.e6*max_daresidual))


    ## append results to our output arrays
    t0_diff_seconds_arr = np.append(t0_diff_seconds_arr, (tdiff*24.*60.*60.))
    t0_diff_seconds_err_arr = np.append(t0_diff_seconds_err_arr, (param_errs[0]*24.*60.*60.))
    chi2red_homog_arr = np.append(chi2red_homog_arr, bf_chi2red)
    chi2red_asym_arr = np.append(chi2red_asym_arr, true_chi2red)


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
    ax2.axvline(t0_new_guess, c='blue', label='Prop. Guess')
    unc_extent = t0_new_guess_uncertainty * 24. * 60.
    ax2.axvline(t0_new_guess - t0_new_guess_uncertainty, c='blue', ls='--', label='Prop. Uncert. (Bounds); +/- %.3f min'%(unc_extent))
    ax2.axvline(t0_new_guess + t0_new_guess_uncertainty, c='blue', ls='--')
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

output_arr_name = array_output_path+'keyoutputs.npz'
np.savez(output_arr_name, 
        fasyms=fasym_arr,
        tdiff_seconds=t0_diff_seconds_arr, tdiff_err_seconds=t0_diff_seconds_err_arr,
        chi2red_homog=chi2red_homog_arr, chi2red_asym=chi2red_asym_arr)

# print save locations to verbose output log
print('verbose output log saved to ', output_file_path)
print('figures saved to ', figure_output_path)
print('key results array saved to', output_arr_name)


print('ran fine')
