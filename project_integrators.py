# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:19:33 2020

@author: Tim Russell
"""

def torque_free_integrator(X0, tspan, tstep, I):
    import numpy as np
    from numpy.linalg import norm
    from mrp_dke import mrp_dke
    from euler_eom import euler_eom
    from mrp_shadow_set import mrp_shadow_set
    import matplotlib.pyplot as plt
    
    steps = int(tspan/tstep)
    
    X = np.zeros((6, steps+1))
    t = np.zeros(steps+1)
    
    X[:,0] = X0.T
    t[0] = 0
    
    for i in range(0, steps):
        # Current state
        sigma = np.array([[X[0,i]],[X[1,i]],[X[2,i]]])
        omega = np.array([[X[3,i]],[X[4,i]],[X[5,i]]])
        
        # Uncomment the following if you want to ensure angular momentum and
        # kinetic energy preserved at each step
        H = norm(I@omega)
        T = 0.5*(I[0,0]*omega[0,0]**2 + I[1,1]*omega[1,0]**2 +
                  I[2,2]*omega[2,0]**2)
        print(f'H = {H}, T = {T}\n')
        
        # k1
        omega1 = omega
        omega_dot1 = euler_eom(I, omega1, 0)
        k1_omega = tstep*omega_dot1
        sigma1 = sigma
        sigma_dot1 = mrp_dke(sigma1, omega1)
        k1_sigma = tstep*sigma_dot1
        
        # k2
        omega2 = omega + 0.5*k1_omega
        omega_dot2 = euler_eom(I, omega2, 0)
        k2_omega = tstep*omega_dot2
        sigma2 = sigma + 0.5*k1_sigma
        sigma_dot2 = mrp_dke(sigma2, omega2)
        k2_sigma = tstep*sigma_dot2
        
        # k3
        omega3 = omega + 0.5*k2_omega
        omega_dot3 = euler_eom(I, omega3, 0)
        k3_omega = tstep*omega_dot3
        sigma3 = sigma + 0.5*k2_sigma
        sigma_dot3 = mrp_dke(sigma3, omega3)
        k3_sigma = tstep*sigma_dot3
        
        # k4
        omega4 = omega + k3_omega
        omega_dot4 = euler_eom(I, omega4, 0)
        k4_omega = tstep*omega_dot4
        sigma4 = sigma + k3_sigma
        sigma_dot4 = mrp_dke(sigma4, omega4)
        k4_sigma = tstep*sigma_dot4
        
        # Next state
        sigma_next = sigma + 1/6*(k1_sigma + 2*k2_sigma +
                                      2*k3_sigma + k4_sigma)
        omega_next = omega + 1/6*(k1_omega + 2*k2_omega +
                                      2*k3_omega + k4_omega)
        
        # Shadow set conversion if necessary
        if norm(sigma_next) > 1:
            sigma_next = mrp_shadow_set(sigma_next)
        
        X[:,i+1] = np.append(sigma_next, omega_next).T
        t[i+1] = tstep*(i+1)
    
    plt.figure(num=None, figsize=(12, 8), dpi=300, facecolor='w', edgecolor='k')

    plt.subplot(2, 3, 1)
    plt.plot(t, X[0,:], '-k')
    plt.title('$\sigma_1$')

    plt.subplot(2, 3, 2)
    plt.plot(t, X[1,:], '-k')
    plt.title('$\sigma_2$')

    plt.subplot(2, 3, 3)
    plt.plot(t, X[2,:], '-k')
    plt.title('$\sigma_3$')

    plt.subplot(2, 3, 4)
    plt.plot(t, X[3,:], '-k')
    plt.title('$\omega_1$')
    plt.xlabel('$s$')
    plt.ylabel('$rad/s$')

    plt.subplot(2, 3, 5)
    plt.plot(t, X[4,:], '-k')
    plt.title('$\omega_2$')
    plt.xlabel('$s$')

    plt.subplot(2, 3, 6)
    plt.plot(t, X[5,:], '-k')
    plt.title('$\omega_3$')
    plt.xlabel('$s$')

    return X, t
 
def constant_torque_integrator(X0, tspan, tstep, I):
    import numpy as np
    from numpy.linalg import norm
    from mrp_dke import mrp_dke
    from euler_eom import euler_eom
    from mrp_shadow_set import mrp_shadow_set
    import matplotlib.pyplot as plt
    
    steps = int(tspan/tstep)
    
    X = np.zeros((6, steps+1))
    t = np.zeros(steps+1)
    u = np.array([[0.01],[-0.01],[0.02]])
    
    X[:,0] = X0.T
    t[0] = 0
    
    for i in range(0, steps):
        # Current state
        sigma = np.array([[X[0,i]],[X[1,i]],[X[2,i]]])
        omega = np.array([[X[3,i]],[X[4,i]],[X[5,i]]])
        
        # Uncomment the following if you want to ensure angular momentum and
        # kinetic energy preserved at each step
        H = norm(I@omega)
        T = 0.5*(I[0,0]*omega[0,0]**2 + I[1,1]*omega[1,0]**2 +
                  I[2,2]*omega[2,0]**2)
        print(f'H = {H}, T = {T}\n')
        
        # k1
        omega1 = omega
        omega_dot1 = euler_eom(I, omega1, u)
        k1_omega = tstep*omega_dot1
        sigma1 = sigma
        sigma_dot1 = mrp_dke(sigma1, omega1)
        k1_sigma = tstep*sigma_dot1
        
        # k2
        omega2 = omega + 0.5*k1_omega
        omega_dot2 = euler_eom(I, omega2, u)
        k2_omega = tstep*omega_dot2
        sigma2 = sigma + 0.5*k1_sigma
        sigma_dot2 = mrp_dke(sigma2, omega2)
        k2_sigma = tstep*sigma_dot2
        
        # k3
        omega3 = omega + 0.5*k2_omega
        omega_dot3 = euler_eom(I, omega3, u)
        k3_omega = tstep*omega_dot3
        sigma3 = sigma + 0.5*k2_sigma
        sigma_dot3 = mrp_dke(sigma3, omega3)
        k3_sigma = tstep*sigma_dot3
        
        # k4
        omega4 = omega + k3_omega
        omega_dot4 = euler_eom(I, omega4, u)
        k4_omega = tstep*omega_dot4
        sigma4 = sigma + k3_sigma
        sigma_dot4 = mrp_dke(sigma4, omega4)
        k4_sigma = tstep*sigma_dot4
        
        # Next state
        sigma_next = sigma + 1/6*(k1_sigma + 2*k2_sigma +
                                      2*k3_sigma + k4_sigma)
        omega_next = omega + 1/6*(k1_omega + 2*k2_omega +
                                      2*k3_omega + k4_omega)
        
        # Shadow set conversion if necessary
        if norm(sigma_next) > 1:
            sigma_next = mrp_shadow_set(sigma_next)
        X[:,i+1] = np.append(sigma_next, omega_next).T
        t[i+1] = tstep*(i+1)
    
    plt.figure(num=None, figsize=(12, 8), dpi=300, facecolor='w', edgecolor='k')
    
    plt.subplot(2, 3, 1)
    plt.plot(t, X[0,:], '-k')
    plt.title('$\sigma_1$')
    
    plt.subplot(2, 3, 2)
    plt.plot(t, X[1,:], '-k')
    plt.title('$\sigma_2$')
    
    plt.subplot(2, 3, 3)
    plt.plot(t, X[2,:], '-k')
    plt.title('$\sigma_3$')
    
    plt.subplot(2, 3, 4)
    plt.plot(t, X[3,:], '-k')
    plt.title('$\omega_1$')
    plt.xlabel('$s$')
    plt.ylabel('$rad/s$')
    
    plt.subplot(2, 3, 5)
    plt.plot(t, X[4,:], '-k')
    plt.title('$\omega_2$')
    plt.xlabel('$s$')
    
    plt.subplot(2, 3, 6)
    plt.plot(t, X[5,:], '-k')
    plt.title('$\omega_3$')
    plt.xlabel('$s$')
    
    return X, t

def pd_feedback_integrator(X0, tspan, tstep, I, K, P, RAAN_LMO, inc_LMO,
                           TA0_LMO, n_LMO, r_LMO, RAAN_GMO, inc_GMO, TA0_GMO,
                           n_GMO, r_GMO):
    import numpy as np
    from numpy.linalg import norm
    from mrp_dke import mrp_dke
    from euler_eom import euler_eom
    from mrp_shadow_set import mrp_shadow_set
    from project_dcms import dcm_orbit, dcm_sun, dcm_nadir, dcm_gmo, dcm_body
    from project_angular_rates import angrate_sun, angrate_nadir, angrate_gmo
    from project_errors import attitude_error, rate_error
    import math
    import matplotlib.pyplot as plt
    
    steps = int(tspan/tstep)
    
    X = np.zeros((6, steps+1))
    X[:,0] = X0.T
    
    t = np.zeros(steps+1)
    
    u = np.zeros((3, steps+1))
    
    sigma_SN = np.zeros((3, steps+1))
    omega_SN = np.zeros((3, steps+1))
    
    sigma_GN = np.zeros((3, steps+1))
    omega_GN = np.zeros((3, steps+1))
    
    sigma_MN = np.zeros((3, steps+1))
    omega_MN = np.zeros((3, steps+1))
    
    O_r_LMO = np.array([[r_LMO],[0],[0]])
    O_r_GMO = np.array([[r_GMO],[0],[0]])
    
    for i in range(0, steps):
        # Current state
        sigma = np.array([[X[0,i]],[X[1,i]],[X[2,i]]])
        omega = np.array([[X[3,i]],[X[4,i]],[X[5,i]]])
        
        # Determine inertial positions of each spacecraft
        DCM_ON_LMO = dcm_orbit(RAAN_LMO, inc_LMO, TA0_LMO, n_LMO, t[i])
        N_r_LMO = np.transpose(DCM_ON_LMO) @ O_r_LMO
        DCM_ON_GMO = dcm_orbit(RAAN_GMO, inc_GMO, TA0_GMO, n_GMO, t[i])
        N_r_GMO = np.transpose(DCM_ON_GMO) @ O_r_GMO
        
        # Determine if LMO spacecraft is on shaded side
        sun_mode = bool((np.transpose(N_r_LMO) @ np.array([[0],[1],[0]])
                           > 0))
        
        # Determine if LMO and GMO comm link closed
        gmo_mode = bool(math.acos((np.transpose(N_r_LMO) @ N_r_GMO)/(r_LMO*r_GMO))
                        < 35*math.pi/180)
        
        # Feedback control torque
        if sun_mode:
            DCM_SN = dcm_sun()
            sigma_BS = attitude_error(sigma, DCM_SN)
            angrate_SN = angrate_sun()
            omega_BS = rate_error(sigma, omega, DCM_SN, angrate_SN)
            torque = -K*sigma_BS - P*omega_BS
        elif gmo_mode:
            DCM_MN = dcm_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
            RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t[i])
            sigma_BM = attitude_error(sigma, DCM_MN)
            
            angrate_MN = angrate_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                                     RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO,
                                     t[i])
            omega_BM = rate_error(sigma, omega, DCM_MN, angrate_MN)
            torque = -K*sigma_BM - P*omega_BM
        else:
            DCM_GN = dcm_nadir(RAAN_LMO, inc_LMO, TA0_LMO, n_LMO, t[i])
            sigma_BG = attitude_error(sigma, DCM_GN)
            angrate_GN = angrate_nadir(RAAN_LMO, inc_LMO, TA0_LMO, n_LMO, t[i])
            omega_BG = rate_error(sigma, omega, DCM_GN, angrate_GN)
            torque = -K*sigma_BG - P*omega_BG
          
        # Uncomment the following if you want to ensure angular momentum and
        # kinetic energy preserved at each step
        # H = norm(I@omega)
        # T = 0.5*(I[0,0]*omega[0,0]**2 + I[1,1]*omega[1,0]**2 +
        # I[2,2]*omega[2,0]**2)
        # print(f'H = {H}, T = {T}\n, u = {norm(u)}\n')
        
        # k1
        omega1 = omega
        omega_dot1 = euler_eom(I, omega1, torque)
        k1_omega = tstep*omega_dot1
        sigma1 = sigma
        sigma_dot1 = mrp_dke(sigma1, omega1)
        k1_sigma = tstep*sigma_dot1
        
        # k2
        omega2 = omega + 0.5*k1_omega
        omega_dot2 = euler_eom(I, omega2, torque)
        k2_omega = tstep*omega_dot2
        sigma2 = sigma + 0.5*k1_sigma
        sigma_dot2 = mrp_dke(sigma2, omega2)
        k2_sigma = tstep*sigma_dot2
        
        # k3
        omega3 = omega + 0.5*k2_omega
        omega_dot3 = euler_eom(I, omega3, torque)
        k3_omega = tstep*omega_dot3
        sigma3 = sigma + 0.5*k2_sigma
        sigma_dot3 = mrp_dke(sigma3, omega3)
        k3_sigma = tstep*sigma_dot3
        
        # k4
        omega4 = omega + k3_omega
        omega_dot4 = euler_eom(I, omega4, torque)
        k4_omega = tstep*omega_dot4
        sigma4 = sigma + k3_sigma
        sigma_dot4 = mrp_dke(sigma4, omega4)
        k4_sigma = tstep*sigma_dot4
        
        # Next state
        sigma_next = sigma + 1/6*(k1_sigma + 2*k2_sigma +
                                      2*k3_sigma + k4_sigma)
        omega_next = omega + 1/6*(k1_omega + 2*k2_omega +
                                      2*k3_omega + k4_omega)
                                      
        # Shadow set conversion if necessary
        if norm(sigma_next) > 1:
            sigma_next = mrp_shadow_set(sigma_next)
        
        # Populate next step state
        X[:,i+1] = np.append(sigma_next, omega_next).T
        t[i+1] = tstep*(i+1)
        
        # Calculate non-state variables for making cool* plots
        # *unofficial opinion of author
        
        # Control torque
        u[0,i] = torque[0]
        u[1,i] = torque[1]
        u[2,i] = torque[2]
        
        # Body-Inertial DCM
        DCM_BN = dcm_body(sigma)
        
        # Sun frame attitude and rates
        sigma_SN[0,i] = 0
        sigma_SN[1,i] = 0.7071
        sigma_SN[2,i] = 0.7071
        
        N_omega_SN = angrate_sun()
        B_omega_SN = DCM_BN @ N_omega_SN
        
        omega_SN[0,i] = B_omega_SN[0]
        omega_SN[1,i] = B_omega_SN[1]
        omega_SN[2,i] = B_omega_SN[2]
        
        # Nadir frame attitude and rates
        DCM_GN = dcm_nadir(RAAN_LMO, inc_LMO, TA0_LMO, n_LMO, t[i])
        xi = math.sqrt(np.trace(DCM_GN)+1)
        MRP_GN = 1/(xi*(xi+2))*np.array([[DCM_GN[1][2]-DCM_GN[2][1]],
                                         [DCM_GN[2][0]-DCM_GN[0][2]],
                                         [DCM_GN[0][1]-DCM_GN[1][0]]])
        sigma_GN[0,i] = MRP_GN[0]
        sigma_GN[1,i] = MRP_GN[1]
        sigma_GN[2,i] = MRP_GN[2]
        
        N_omega_GN = angrate_nadir(RAAN_LMO, inc_LMO, TA0_LMO, n_LMO, t[i])
        B_omega_GN = DCM_BN @ N_omega_GN
        omega_GN[0,i] = B_omega_GN[0]
        omega_GN[1,i] = B_omega_GN[1]
        omega_GN[2,i] = B_omega_GN[2]
        
        # GMO frame attitude and rates
        DCM_MN = dcm_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                         RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t[i])
        xi = math.sqrt(np.trace(DCM_MN)+1)
        MRP_MN = 1/(xi*(xi+2))*np.array([[DCM_MN[1][2]-DCM_MN[2][1]],
                                         [DCM_MN[2][0]-DCM_MN[0][2]],
                                         [DCM_MN[0][1]-DCM_MN[1][0]]])
        sigma_MN[0,i] = MRP_MN[0]
        sigma_MN[1,i] = MRP_MN[1]
        sigma_MN[2,i] = MRP_MN[2]

        N_omega_MN = angrate_gmo(RAAN_LMO, inc_LMO, TA0_LMO, r_LMO, n_LMO,
                                 RAAN_GMO, inc_GMO, TA0_GMO, r_GMO, n_GMO, t[i])

        B_omega_MN = DCM_BN @ N_omega_MN
        omega_MN[0,i] = B_omega_MN[0]
        omega_MN[1,i] = B_omega_MN[1]
        omega_MN[2,i] = B_omega_MN[2]

    plt.figure(num=None, figsize=(12, 8), dpi=300, facecolor='w', edgecolor='k')
    plt.title('Spacecraft Attitude and Rates (w/r/t Inertial Frame)')

    plt.subplot(2, 3, 1)
    plt.plot(t, X[0,:], '-k')
    plt.plot(t[0:-2], sigma_SN[0,0:-2], '--r')
    plt.plot(t[0:-2], -sigma_SN[0,0:-2], '--r')
    plt.plot(t[0:-2], sigma_GN[0,0:-2], ':g')
    plt.plot(t[0:-2], sigma_MN[0,0:-2], '-.b')
    plt.title('$\sigma_1$')

    plt.subplot(2, 3, 2)
    plt.plot(t, X[1,:], '-k')
    plt.plot(t[0:-2], sigma_SN[1,0:-2], '--r')
    plt.plot(t[0:-2], -sigma_SN[1,0:-2], '--r')
    plt.plot(t[0:-2], sigma_GN[1,0:-2], ':g')
    plt.plot(t[0:-2], sigma_MN[1,0:-2], '-.b')
    plt.title('$\sigma_2$')

    plt.subplot(2, 3, 3)
    plt.plot(t, X[2,:], '-k')
    plt.plot(t[0:-2], sigma_SN[2,0:-2], '--r')
    plt.plot(t[0:-2], -sigma_SN[2,0:-2], '--r')
    plt.plot(t[0:-2], sigma_GN[2,0:-2], ':g')
    plt.plot(t[0:-2], sigma_MN[2,0:-2], '-.b')
    plt.title('$\sigma_3$')

    plt.subplot(2, 3, 4)
    plt.plot(t, X[3,:], '-k')
    plt.plot(t[0:-2], omega_SN[0,0:-2], '--r')
    plt.plot(t[0:-2], omega_GN[0,0:-2], ':g')
    plt.plot(t[0:-2], omega_MN[0,0:-2], '-.b')
    plt.title('$\omega_1$')
    plt.xlabel('$s$')
    plt.ylabel('$rad/s$')

    plt.subplot(2, 3, 5)
    plt.plot(t, X[4,:], '-k')
    plt.plot(t[0:-2], omega_SN[1,0:-2], '--r')
    plt.plot(t[0:-2], omega_GN[1,0:-2], ':g')
    plt.plot(t[0:-2], omega_MN[1,0:-2], '-.b')
    plt.title('$\omega_2$')
    plt.xlabel('$s$')

    ax = plt.subplot(2, 3, 6)
    ax.plot(t, X[5,:], '-k', label='S/C Telemetry')
    ax.plot(t[0:-2], omega_SN[2,0:-2], '--r', label='Sun Reference')
    ax.plot(t[0:-2], omega_GN[2,0:-2], ':g', label='Nadir Reference')
    ax.plot(t[0:-2], omega_MN[2,0:-2], '-.b', label='GMO Reference')
    plt.title('$\omega_3$')
    plt.xlabel('$s$')
    ax.legend()
    
    plt.figure(num=None, figsize=(12, 8), dpi=300, facecolor='w', edgecolor='k')
    plt.title('Control Torque (Body Frame)')
    
    plt.subplot(1, 3, 1)
    plt.plot(t[0:-2], u[0,0:-2], '-k')
    plt.title('$u_1$')
    plt.xlabel('$s$')
    plt.ylabel('$Nm$')
    
    plt.subplot(1, 3, 2)
    plt.plot(t[0:-2], u[1,0:-2], '-k')
    plt.title('$u_2$')
    plt.xlabel('$s$')
    
    plt.subplot(1, 3, 3)
    plt.plot(t[0:-2], u[2,0:-2], '-k')
    plt.title('$u_3$')
    plt.xlabel('$s$')
    
    return X, t, u, sigma_SN, omega_SN, sigma_GN, omega_GN, sigma_MN, omega_MN
