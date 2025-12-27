# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:55:54 2020

@author: Tim Russell

Calculate attitude and rate errors of a body frame with respect to a reference
frame.
"""

def attitude_error(sigma_BN, DCM_RN):
    import numpy as np
    from numpy.linalg import norm
    import math
    from cross_matrix import cross_matrix
    
    sigma_tilde_BN = cross_matrix(sigma_BN)
    
    DCM_BN = np.identity(3) + (8*sigma_tilde_BN@sigma_tilde_BN - \
                               4*(1-norm(sigma_BN)**2)*sigma_tilde_BN) / \
                              (1+norm(sigma_BN)**2)**2
    
    DCM_BR = DCM_BN @ np.transpose(DCM_RN)
    
    xi = math.sqrt(np.trace(DCM_BR)+1)
    
    sigma_BR = 1/(xi*(xi+2))*np.array([[DCM_BR[1][2]-DCM_BR[2][1]],
                                       [DCM_BR[2][0]-DCM_BR[0][2]],
                                       [DCM_BR[0][1]-DCM_BR[1][0]]])
    
    return sigma_BR

def rate_error(sigma_BN, B_omega_BN, DCM_RN, N_omega_RN):
    import numpy as np
    from numpy.linalg import norm
    from cross_matrix import cross_matrix
    
    sigma_BN_tilde = cross_matrix(sigma_BN)
    
    DCM_BN = np.identity(3) + (8*sigma_BN_tilde@sigma_BN_tilde - \
                               4*(1-norm(sigma_BN)**2)*sigma_BN_tilde) / \
                              (1+norm(sigma_BN)**2)**2
    
    B_omega_RN = DCM_BN @ N_omega_RN
    B_omega_BR = B_omega_BN - B_omega_RN
    
    return B_omega_BR
