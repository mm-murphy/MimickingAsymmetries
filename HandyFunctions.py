## Functions that will be used ...
def calc_scale_height(T, M, R, mm=2):
    """ Calculates the approximate scale height of a planet's atmosphere, using the equation
     scale height = kT / mg
    
    Inputs: T = the atmospheric temperature in [K]; M = the planet's mass in [Mjupiter]; 
            R = the planet's radius in [Rjupiter]; mm = mean mass of a molecule in the atmosphere [amu], this is
                   default set to 1 amu = 1 proton mass (for now)
    Outputs: H = the scale height in [km]
    """
    # constants:
    amu = 1.67e-27 # [kg]; atomic mass unit in [kg]
    k = 1.38e-23 # [Joule/K]; Boltzmann constant
    G = 6.674e-11 # [m^3/kg/s^2]; Gravitational constant
    Mjupiter = 1.9e27 # [kg]; mass of Jupiter
    Rjupiter = 69911000.0 # [m]; approx. radius of Jupiter
    
    # computing the numerator for the scale height equation:
    E_thermal = k*T # [Joule]
    
    # computing the denominator:
    M_kg, R_m = M*Mjupiter, R*Rjupiter # convert planet quantities into SI units
    g = G*M_kg/(R_m**2) # gravitational acceleration in [m/s^2]
    meanmass = mm*amu
    denominator = meanmass*g # [kg*m/s^2]
    
    # compute the scale height:
    H = E_thermal / denominator # [meters]
    H /= 1000. # convert to [km] from [m]
    
    return H

def convert_rprs_to_rpJ(rprs, rs):
    """ Converts the planet-star radius ratio into the planet's radius in Jupiter radii
    Inputs: rprs = planet-star radius ratio; rs = stellar radius in [Rsun]
    Outputs: rp = planet radius in [RJupiter]
    """
    # compute planet radius in [solar radii]
    rp_s = rprs*rs # [Rsol]
    # convert [solar radii] to [jupiter radii]
    rp_J = rp_s * 9.73116
    
    return rp_J

def convert_rpJ_to_rprs(rpJ, rs):
    """ Converts a planet radius to the planet-star radius ratio
    Inputs: rpJ = planet radius in [Rjupiter], rs = stellar radius in [Rsun]
    Outputs: rprs = planet-star radius ratio
    """
    # convert planet radius to [Rsol]
    rp_s = rpJ / 9.73116
    # divide by stellar radius (in [Rsol])
    rprs = rp_s / rs
    
    return rprs

def convert_km_to_rpJ(km):
    """ Converts a quantity in [km] to [Jupiter radii]"""
    d = km / 71492.
    return d

def logLikelihood(ydata, yerr, modely):
    """ Computes the Bayesian likelihood of a model, given the data (or is it the other way around?)
    Inputs: ydata = your data, yerr= uncertainties on your data, modely = same size array of the model's values
    outputs: ln( the likelihood )
    """
    lnL = 0.
    chi_array = ((ydata - modely) ** 2. / yerr ** 2.) + np.log(2. * np.pi * yerr ** 2.)
    lnL += -0.5 * np.sum(chi_array)
    
    return lnL

def logPriors(theta, info_dict):
    """ Computes the Bayesian prior value, given the current set of parameter values
    Inputs: theta = array of parameter values, info_dict = a dictionary that defines which parameters are having
        priors enforced, which type, and of what bounds/values
    Output: ln(the prior value)
    """
    lnP_runsum = 0.
    for i, key in enumerate(info_dict['Prior'].keys()):
        if info_dict['Prior'][key][2] == 'U':
            l1, l2 = info_dict['Prior'][key][0][0], info_dict['Prior'][key][0][1]
            if not (l1 <= theta[i] <= l2):
                return -np.inf
        elif info_dict['Prior'][key][2] == 'N':
            pval, perr = info_dict['Prior'][key][0], info_dict['Prior'][key][1]
            lnP_runsum += -(theta[i] - pval) ** 2. / (2. * perr ** 2.) - np.log(np.sqrt(2. *perr **2. * np.pi))
    
    return lnP_runsum

# def homog_transit_model(theta, InitModel):
#     """
#     Input to the MCMC. Given a set of parameters (theta) and an initialized BATMAN environment (InitModel),
#         outputs the model lightcurve array\
#     Note: Will need to adjust this function when changing what parameters are free in the MCMC
#     """
#     Params = batman.TransitParams()
#     Params.t0 = theta[0]
#     Params.per = lit_params['P'][0]
#     Params.rp = np.mean((rprs1, rprs2))
#     Params.a = lit_params['a'][0]
#     Params.inc = lit_params['inc'][0]
#     Params.ecc = 0.
#     Params.w = 90.
#     Params.u = [lit_params['u1'][0], lit_params['u2'][0]]  # these are set arbitrarily
#     Params.limb_dark = 'quadratic'
#     step_lc = InitModel.light_curve(Params)
#     return step_lc

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

def compute_bic(Nparams, Ndata, max_lnL):
    """computes the bayesian information criterion"""
    bic = Nparams*np.log(Ndata) - 2.*max_lnL
    return bic

def compute_chi2(ydata, yerr, modely, reduced=True, Ndof=1):
    chi2vals = (ydata - modely)**2 / yerr**2
    chi2 = np.sum(chi2vals)
    
#     runningSum = 0.
#     for i, y in enumerate(ydata):
#         runningSum += ((ydata[i] - modely[i])**2 / (yerr[i]**2))
#     chi2 = runningSum
    if reduced:
        chi2red = chi2 / Ndof
        return chi2red
    else:
        return chi2