# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:20:32 2020

@author: Tim Russell
"""
def mrp_shadow_set(sigma):
    from numpy.linalg import norm

    sigma_s = -sigma/norm(sigma)**2

    return sigma_s