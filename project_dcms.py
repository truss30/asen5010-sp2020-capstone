# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:36:29 2020

@author: Tim Russell

Calculates the DCMs to convert from the inertial frame to reference frames.
"""

def dcm_orbit(RAAN, inc, TA0, n, t):
    import numpy as np
    import math
    from calc_TA import calc_TA
    
    TA = calc_TA(TA0, n, t)
    
    c1 = math.cos(RAAN * math.pi/180)
    s1 = math.sin(RAAN * math.pi/180)
    c2 = math.cos(inc * math.pi/180)
    s2 = math.sin(inc * math.pi/180)
    c3 = math.cos(TA * math.pi/180)
    s3 = math.sin(TA * math.pi/180)
    
    DCM_ON = np.array([[c3*c1-s3*c2*s1, c3*s1+s3*c2*c1, s3*s2],
                       [-s3*c1-c3*c2*s1, -s3*s1+c3*c2*c1, c3*s2],
                       [s2*s1, -s2*c1, c2]])
    
    return DCM_ON
    
def dcm_sun():
    import numpy as np
    
    DCM_SN = np.array([[-1, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]])

    return DCM_SN

def dcm_nadir(RAAN, inc, TA0, n, t):
    import numpy as np
    
    DCM_ON = dcm_orbit(RAAN, inc, TA0, n, t)
    
    DCM_GO = np.array([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -1]])
    
    DCM_GN = DCM_GO @ DCM_ON
    
    return DCM_GN

def dcm_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
            RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t):
    import numpy as np
    from numpy.linalg import norm
    from cross_matrix import cross_matrix
    
    O_r_LMO = np.array([[r_LMO],[0],[0]])
    DCM_NO_LMO = np.transpose(dcm_orbit(RAAN_LMO, inc_LMO, TA0_LMO, n_LMO, t))
    N_r_LMO = DCM_NO_LMO @ O_r_LMO
    
    O_r_GMO = np.array([[r_GMO],[0],[0]])
    DCM_NO_GMO = np.transpose(dcm_orbit(RAAN_GMO, inc_GMO, TA0_GMO, n_GMO, t))
    N_r_GMO = DCM_NO_GMO @ O_r_GMO
    
    N_delr = N_r_GMO - N_r_LMO
    
    m1 = -N_delr / norm(N_delr)
    
    m2 = cross_matrix(N_delr) @ np.array([[0],[0],[1]])
    m2 = m2 / norm(m2)
    
    m3 = cross_matrix(m1) @ m2
    m3 = m3 / norm(m3)
    
    DCM_MN = np.transpose(m1)
    DCM_MN = np.append(DCM_MN, np.transpose(m2), axis=0)
    DCM_MN = np.append(DCM_MN, np.transpose(m3), axis=0)
    
    return DCM_MN

def dcm_body(sigma_BN):
    import numpy as np
    from numpy.linalg import norm
    from cross_matrix import cross_matrix
    
    sigma_tilde_BN = cross_matrix(sigma_BN)
    
    DCM_BN = np.identity(3) + (8*sigma_tilde_BN@sigma_tilde_BN - \
                               4*(1-norm(sigma_BN)**2)*sigma_tilde_BN) / \
                              (1+norm(sigma_BN)**2)**2
    
    return DCM_BN
