## Importing the packages we'll use
import numpy as np              # numerical python
import matplotlib.pyplot as plt # plotting
import batman                   # homogeneous limb transit modeling
import catwoman                 # asymmetric limb transit modeling
import emcee                    # MCMC sampling package
from multiprocessing import Pool # easy multiprocessing with emcee
from HandyFunctions import *

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
    # haven't written this yet
else:
    # if not, let's create a data set
    # it will be a single transit set some N periods ahead of the literature transit time, given above
    t0_true = lit_params['t0'][0]  # literature transit time [day]
    P_true = lit_params['P'][0]    # literature period [day]
    Ntransits_ahead = 10 # N transits ahead of lit. transit time to place our new data
                         # this will factor into the ephemeris uncertainty, as it grows with sqrt(N) [i think]
    t0_new_true = t0_true + Ntransits_ahead*P_true  # 'True' propagated transit time 
    obs_window_size = 3. # [hours] before/after the transit midpoint to generate a model for
    Ndatapoints = 350    # number of light curve points
    # generate time axis
    time = np.linspace(t0_new_true-obs_window_size/24., t0_new_true+obs_window_size/24., Ndatapoints)
    # initialize arrays for the flux and flux uncertainty, which will be set later on
    syn_fluxes, syn_errs = np.ones(time.shape), np.ones(time.shape)
    scatter = 250. # [ppm], standard deviation of flux values about the model
    flux_uncertainty = scatter # [ppm], uncertainty on each flux point
    
### Generating the intrinsic asymmetry model
## First, we need to define what asymmetry factor to use (or what range of values to use)
##   note: as of right now, the factor is defined the number of scale heights by which the two radii differ
asymmetry_factors_totest = np.array([5., 10., 15., 20., 25., 30., 35.])

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
    
## Creating an initialized CATWOMAN model for the asymmetric limb transit
# note - can change the rp2 argument later on to regenerate new asymmetry factor models
#        using this same initialized environment
InitAsymParams = catwoman.TransitParams()
InitAsymParams.t0 = t0_new_true # transit midpoint in [day]
InitAsymParams.per = P_true # orbital period in [day]
InitAsymParams.rp = rprs1 # trailing limb RpRs
InitAsymParams.rp2 = rprs2 # leading limb RpRs
InitAsymParams.a = lit_params['a'][0] # semi-major axis in [Rsol]
InitAsymParams.inc = lit_params['inc'][0] # inclination in [deg]
InitAsymParams.ecc = 0. # eccentricity
InitAsymParams.w = 90. # argument of periastron?
InitAsymParams.u = [lit_params['u1'][0], lit_params['u2'][0]]  # limb darkening coefficients
InitAsymParams.phi = 90. # 90 - obliquity [deg]
InitAsymParams.limb_dark = 'quadratic' # type of limb darkening law to use
InitAsymModel = catwoman.TransitModel(InitAsymParams, time)
init_asym_lc = InitAsymModel.light_curve(InitAsymParams)

## Creating an initialized BATMAN model environment for the homogeneous limb transit
InitHomogParams = batman.TransitParams()
InitHomogParams.t0 = lit_params['t0'][0] # transit midpoint in [day]
InitHomogParams.per = lit_params['P'][0] # orbital period in [day]
InitHomogParams.rp = np.mean((rprs1_vals[0], rprs2_vals[0]))
InitHomogParams.a = lit_params['a'][0] # semi-major axis in [Rsol]
InitHomogParams.inc = lit_params['inc'][0] # inclination in [deg]
InitHomogParams.ecc = 0. # eccentricity
InitHomogParams.w = 90. # argument of periastron?
InitHomogParams.u = [lit_params['u1'][0], lit_params['u2'][0]]  # limb darkening coefficients
InitHomogParams.phi = 90. # 90 - obliquity [deg]
InitHomogParams.limb_dark = 'quadratic' # type of limb darkening law to use
InitHomogModel = batman.TransitModel(InitHomogParams, time)
init_homog_lc = InitHomogModel.light_curve(InitHomogParams)

### Setting up the MCMC fitting
## In this script, I'll only fit for the transit time
## and use its uniform prior bounds as our measure of the requisite precision
## Defining the info dict
t0_uncertainty = lit_params['t0'][1]
fit_pars = {
    'Init':{
        # Initialization values
        't0':lit_params['t0'][0]#,
    #    'log10P':lit_params['log10P'][0]
    },
    'Prior':{
        # Bayesian priors
        # 0 = prior value or prior bounds if type = uniform, 1 = prior error (also initialization ball size), 2 = prior type
        't0':np.array([(fit_pars['Init']['t0']-t0_uncertainty, fit_pars['Init']['t0']+t0_uncertainty), lit_params['t0'][1], 'U'], dtype=object)#,
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


## Now going into the script loop
for i_factor, asym_factor in enumerate(asymmetry_factors_totest):
    print('Testing asymmetry factor %.0f'%(asym_factor))
    this_rprs1, this_rprs2 = rprs1_vals[i_factor], rprs2_vals[i_factor]
    this_rpJ1 = convert_rprs_to_rpJ(this_rprs1, lit_params['Rs'][0])
    this_rpJ2 = convert_rprs_to_rpJ(this_rprs2, lit_params['Rs'][0])
    print('This trailing limb radius: RpRs = %.5f; Rp = %.3f RJupiter'%(this_rprs1, this_rpJ1))
    print('This leading limb radius: RpRs = %.5f; Rp = %.3f RJupiter'%(this_rprs2, this_rpJ2))
    print('Fitting for the transit time')
    t0_uncert_minutes = t0_uncertainty * 24. * 60.
    print('t0 uncertainty = %.3f minutes'%(t0_uncert_minutes))
    
    # set up initial parameter array and walkers
    Nparams = len(fit_pars['Init'].keys())
    theta_init = np.zeros(Nparams)
    theta_init_errs = np.zeros(Nparams)
    for i, key in enumerate(fit_pars['Init'].keys()):
        theta_init[i] = fit_pars['Init'][key]
        theta_init_errs[i] = fit_pars['Prior'][key][1]
    print('Initial parameter array = ', theta_init)
    
    Nwalkers = 3*Nparams
    Nsteps = 50000
    Nburn = 1000
    pos = np.zeros((Nwalkers, Nparams))
    for j in range(Nparams):
        pos[:,j] = theta_init[j] + 0.5*np.random.normal(0., theta_init_errs[j], Nwalkers)

    
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
        
    # grab the outputs
    samples = sampler.get_chain(discard=Nburn)
    flatsamples = sampler.get_chain(discard=Nburn, flat=True)
    loglikelihoods = sampler.get_log_prob(discard=Nburn, flat=True)
    autocorrtimes = sampler.get_autocorr_time()