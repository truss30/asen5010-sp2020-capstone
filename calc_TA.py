# -*- coding: utf-8 -*-
"""
Created on Sat Apr 4 14:50:52 2020

@author: Tim Russell

Calculates the true anomaly of a circular orbit given initial true anomaly
(deg), mean motion (rad/s), and time elapsed (s). Normalizes to [0, 360) deg.
"""

def calc_TA(TA0, n, t):
    import math
    n = n * 180/math.pi
    TA = (TA0 + n*t) % 360
    
    return TA
