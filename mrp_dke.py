# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:32:06 2020

@author: Tim Russell

Calculates the instantaneous attitude rates in terms of MRPs. Inputs are an
MRP attitude description and an attitude rate vector (in rad/s).
"""

def mrp_dke(sigma, omega):
    import numpy as np
    from numpy.linalg import norm
    from cross_matrix import cross_matrix
    
    sigma_tilde = cross_matrix(sigma)
    B_sigma = (1 - norm(sigma)**2)*np.identity(3) + 2*sigma_tilde + \
              2*sigma@sigma.T
    sigma_dot = 0.25*B_sigma@omega
    
    return sigma_dot
