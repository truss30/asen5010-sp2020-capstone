# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:17:32 2020

@author: Tim Russell

Calculates the angular rates of reference frames with respect to the inertial
frame.
"""

def angrate_orbit(RAAN, inc, TA0, n, t):
    import numpy as np
    from project_dcms import dcm_orbit
    
    O_omega_ON = np.array([[0],[0],[n]])
    DCM_NO = np.transpose(dcm_orbit(RAAN, inc, TA0, n, t))
    N_omega_ON = DCM_NO @ O_omega_ON
    
    return N_omega_ON

def angrate_sun():
    import numpy as np
    
    N_omega_SN = np.array([[0],[0],[0]])
    
    return N_omega_SN

def angrate_nadir(RAAN, inc, TA0, n, t):
    """
    Calculate the angular rate of the nadir-pointing reference frame with
    respect to the inertial frame. Note that this orbit is circular and
    unperturbed.
    Parameters
    ----------
    RAAN : float (deg)
        Right ascension of the ascending node of the spacecraft orbit.
    inc : float (deg)
        Inclination of the spacecraft orbit.
    TA0 : float (deg)
        True anomaly of the spacecraft in its orbit at epoch t=0.
    n : float (rad/s)
        Mean motion of the spacecraft orbit.
    t : float (s)
        Time elapsed since t=0.
    
    Returns
    -------
    ang_rate : float (rad/s)
        Angular rate of the nadir-pointing reference frame with respect to the
        inertial frame. Represented in inertial frame coordinates.
    """
    import numpy as np
    from project_dcms import dcm_nadir

    G_omega_GN = np.array([[0],[0],[-n]])
    DCM_NG = np.transpose(dcm_nadir(RAAN, inc, TA0, n, t))
    
    N_omega_GN = DCM_NG @ G_omega_GN
    
    return N_omega_GN

def angrate_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t):
    """
    Calculate the angular rate of the GMO-pointing reference frame with
    respect to the inertial frame. Note that the orbits of the two spacecraft
    are circular and unperturbed.
    Parameters
    ----------
    RAAN_LMO : float (deg)
        Right ascension of the ascending node of the LMO spacecraft orbit.
    inc_LMO : float (deg)
        Inclination of the LMO spacecraft orbit.
    TA0_LMO : float (deg)
        True anomaly of the LMO spacecraft in its orbit at epoch t=0.
    r_LMO : float (km)
        Raidus of the LMO spacecraft orbit.
    n_LMO : float (rad/s)
        Mean motion of the LMO spacecraft orbit.
    RAAN_GMO : float (deg)
        Right ascension of the ascending node of the GMO spacecraft orbit.
    inc_GMO : float (deg)
        Inclination of the GMO spacecraft orbit.
    TA0_GMO : float (deg)
        True anomaly of the GMO spacecraft in its orbit at epoch t=0.
    r_GMO : float (km)
        Raidus of the GMO spacecraft orbit.
    n_GMO : float (rad/s)
        Mean motion of the GMO spacecraft orbit.
    t : float (s)
        Time elapsed since t=0.

    Returns
    -------
    ang_rate : float (rad/s)
        Angular rate of the GMO-pointing reference frame with respect to the
        inertial frame. Represented in inertial frame coordinates.
    """
    import numpy as np
    from project_dcms import dcm_gmo
    
    # Calculate [MN](t)
    DCM_MN = dcm_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                     RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t)
    
    # Calculate [MN](t-dt)
    mDCM_MN = dcm_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                      RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t-0.001)
    
    # Calculate [MN](t+dt)
    pDCM_MN = dcm_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                      RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t+0.001)
    
    # Calculate d[MN]/dt(t)
    DCMdot_MN = (pDCM_MN - mDCM_MN)/0.002
    
    # Calculate w_tilde = -d[MN]/dt*[MN]'
    omega_tilde_MN = -DCMdot_MN @ np.transpose(DCM_MN)
    
    # Calculate w in M frame
    M_omega_MN = np.array([[omega_tilde_MN[2][1]],[omega_tilde_MN[0][2]],
                           [omega_tilde_MN[1][0]]])
    
    # Calculate w in N frame
    N_omega_MN = np.transpose(DCM_MN) @ M_omega_MN
    
    return N_omega_MN
