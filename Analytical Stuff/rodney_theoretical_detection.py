import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

E_alum = 73.1*10**9

t_wall_1 = 0.02 * 2.54 / 100
t_wall_2 = 0.035 * 2.54 / 100
t_wall_3 = 0.049 * 2.54 / 100
t_wall_4 = 0.058 * 2.54 / 100
t_wall_5 = 0.065 * 2.54 / 100
t_wall_6 = 0.08 * 2.54 / 100

d_out = .5 * 2.54 / 100
d_in_1 = d_out - 2*t_wall_1
d_in_2 = d_out - 2*t_wall_2
d_in_3 = d_out - 2*t_wall_3
d_in_4 = d_out - 2*t_wall_4
d_in_5 = d_out - 2*t_wall_5
d_in_6 = d_out - 2*t_wall_6

I_solid = np.pi/4*(d_out/2)**4
I_empty_1 = np.pi/4*(d_in_1/2)**4
I_empty_2 = np.pi/4*(d_in_2/2)**4
I_empty_3 = np.pi/4*(d_in_3/2)**4
I_empty_4 = np.pi/4*(d_in_4/2)**4
I_empty_5 = np.pi/4*(d_in_5/2)**4
I_empty_6 = np.pi/4*(d_in_6/2)**4

I_1 = I_solid - I_empty_1
I_2 = I_solid - I_empty_2
I_3 = I_solid - I_empty_3
I_4 = I_solid - I_empty_4
I_5 = I_solid - I_empty_5
I_6 = I_solid - I_empty_6

EI_stalk_low = 10
EI_stalk_med = 20
EI_stalk_hi = 40

height = .7
pre_delta = .05
gamma = .85
k_theta = 2.65
c_theta = 1.24
rod_pos = np.arange(.12, .22, .005)
angles = [0, 5, 10, 15, 20, 25, 30, 35]

def equations(x, delta):
    cap_theta, L = x
    eqn1 = gamma*L*np.sin(cap_theta) - delta
    eqn2 = L*(1-gamma*(1-np.cos(cap_theta))) - height
    return eqn1, eqn2

backward_moms_a_low = np.zeros((len(angles), len(rod_pos)))
backward_moms_a_med = np.zeros((len(angles), len(rod_pos)))
backward_moms_a_hi = np.zeros((len(angles), len(rod_pos)))
backward_moms_b_low = np.zeros((len(angles), len(rod_pos)))
backward_moms_b_med = np.zeros((len(angles), len(rod_pos)))
backward_moms_b_hi = np.zeros((len(angles), len(rod_pos)))
backward_a_low_I_1_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_med_I_1_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_hi_I_1_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_low_I_2_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_med_I_2_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_hi_I_2_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_low_I_3_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_med_I_3_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_hi_I_3_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_low_I_4_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_med_I_4_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_hi_I_4_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_low_I_5_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_med_I_5_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_hi_I_5_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_low_I_6_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_med_I_6_strain = np.zeros((len(angles), len(rod_pos)))
backward_a_hi_I_6_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_low_I_1_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_med_I_1_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_hi_I_1_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_low_I_2_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_med_I_2_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_hi_I_2_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_low_I_3_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_med_I_3_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_hi_I_3_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_low_I_4_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_med_I_4_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_hi_I_4_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_low_I_5_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_med_I_5_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_hi_I_5_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_low_I_6_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_med_I_6_strain = np.zeros((len(angles), len(rod_pos)))
backward_b_hi_I_6_strain = np.zeros((len(angles), len(rod_pos)))


forward_moms_a_low = np.zeros((len(angles), len(rod_pos)))
forward_moms_a_med = np.zeros((len(angles), len(rod_pos)))
forward_moms_a_hi = np.zeros((len(angles), len(rod_pos)))
forward_moms_b_low = np.zeros((len(angles), len(rod_pos)))
forward_moms_b_med = np.zeros((len(angles), len(rod_pos)))
forward_moms_b_hi = np.zeros((len(angles), len(rod_pos)))
forward_a_low_I_1_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_med_I_1_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_hi_I_1_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_low_I_2_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_med_I_2_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_hi_I_2_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_low_I_3_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_med_I_3_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_hi_I_3_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_low_I_4_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_med_I_4_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_hi_I_4_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_low_I_5_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_med_I_5_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_hi_I_5_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_low_I_6_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_med_I_6_strain = np.zeros((len(angles), len(rod_pos)))
forward_a_hi_I_6_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_low_I_1_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_med_I_1_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_hi_I_1_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_low_I_2_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_med_I_2_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_hi_I_2_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_low_I_3_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_med_I_3_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_hi_I_3_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_low_I_4_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_med_I_4_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_hi_I_4_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_low_I_5_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_med_I_5_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_hi_I_5_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_low_I_6_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_med_I_6_strain = np.zeros((len(angles), len(rod_pos)))
forward_b_hi_I_6_strain = np.zeros((len(angles), len(rod_pos)))

for i in range(len(angles)):
    angle = angles[i] * np.pi/180
    for j in range(len(rod_pos)):
        backward_delta = pre_delta + (.125*np.sin(angle) - (rod_pos[j] - .1)*np.sin(angle))
        backward_cap_theta, backward_L = fsolve(equations, np.array([.3, .75]), args=(backward_delta))
    
        backward_theta_0 = c_theta*backward_cap_theta
        if i == 5:
            print(f'backward delta: {backward_delta}')
            print(f'backward position: {rod_pos[j]}')
            print(f'backward theta_0: {backward_theta_0*180/np.pi}')
    
        backward_phi = backward_theta_0 + np.pi/2
        backward_n = -1/np.tan(backward_phi)
        
        backward_F_low = EI_stalk_low*backward_cap_theta*gamma*k_theta/(backward_L*(height + backward_n*backward_delta))
        backward_F_med = EI_stalk_med*backward_cap_theta*gamma*k_theta/(backward_L*(height + backward_n*backward_delta))
        backward_F_hi = EI_stalk_hi*backward_cap_theta*gamma*k_theta/(backward_L*(height + backward_n*backward_delta))
        backward_Fp_low = backward_F_low*np.cos(angle)
        backward_Fp_med = backward_F_med*np.cos(angle)
        backward_Fp_hi = backward_F_hi*np.cos(angle)
        if i == 5:
            print(f'Fp_hi: {backward_Fp_hi}')

        backward_moms_a_low[i, j] = backward_Fp_low * (rod_pos[j]-0.01651)
        backward_moms_a_med[i, j] = backward_Fp_med * (rod_pos[j]-0.01651)
        backward_moms_a_hi[i, j] = backward_Fp_hi * (rod_pos[j]-0.01651)
        backward_moms_b_low[i, j] = backward_Fp_low * (rod_pos[j]-0.0635)
        backward_moms_b_med[i, j] = backward_Fp_med * (rod_pos[j]-0.0635)
        backward_moms_b_hi[i, j] = backward_Fp_hi * (rod_pos[j]-0.0635)

        backward_a_low_I_1_strain[i, j] = (backward_moms_a_low[i, j] * d_out/2 / I_1 + backward_F_low*np.sin(angle))/E_alum 
        backward_a_med_I_1_strain[i, j] = (backward_moms_a_med[i, j] * d_out/2 / I_1 + backward_F_med*np.sin(angle))/E_alum
        backward_a_hi_I_1_strain[i, j] = (backward_moms_a_hi[i, j] * d_out/2 / I_1 + backward_F_hi*np.sin(angle))/E_alum
        backward_a_low_I_2_strain[i, j] = (backward_moms_a_low[i, j] * d_out/2 / I_2 + backward_F_low*np.sin(angle))/E_alum
        backward_a_med_I_2_strain[i, j] = (backward_moms_a_med[i, j] * d_out/2 / I_2 + backward_F_med*np.sin(angle))/E_alum
        backward_a_hi_I_2_strain[i, j] = (backward_moms_a_hi[i, j] * d_out/2 / I_2 + backward_F_hi*np.sin(angle))/E_alum
        backward_a_low_I_3_strain[i, j] = (backward_moms_a_low[i, j] * d_out/2 / I_3 + backward_F_low*np.sin(angle))/E_alum
        backward_a_med_I_3_strain[i, j] = (backward_moms_a_med[i, j] * d_out/2 / I_3 + backward_F_med*np.sin(angle))/E_alum
        backward_a_hi_I_3_strain[i, j] = (backward_moms_a_hi[i, j] * d_out/2 / I_3 + backward_F_hi*np.sin(angle))/E_alum
        backward_a_low_I_4_strain[i, j] = (backward_moms_a_low[i, j] * d_out/2 / I_4 + backward_F_low*np.sin(angle))/E_alum
        backward_a_med_I_4_strain[i, j] = (backward_moms_a_med[i, j] * d_out/2 / I_4 + backward_F_med*np.sin(angle))/E_alum
        backward_a_hi_I_4_strain[i, j] = (backward_moms_a_hi[i, j] * d_out/2 / I_4 + backward_F_hi*np.sin(angle))/E_alum
        backward_a_low_I_5_strain[i, j] = (backward_moms_a_low[i, j] * d_out/2 / I_5 + backward_F_low*np.sin(angle))/E_alum
        backward_a_med_I_5_strain[i, j] = (backward_moms_a_med[i, j] * d_out/2 / I_5 + backward_F_med*np.sin(angle))/E_alum
        backward_a_hi_I_5_strain[i, j] = (backward_moms_a_hi[i, j] * d_out/2 / I_5 + backward_F_hi*np.sin(angle))/E_alum
        backward_a_low_I_6_strain[i, j] = (backward_moms_a_low[i, j] * d_out/2 / I_6 + backward_F_low*np.sin(angle))/E_alum
        backward_a_med_I_6_strain[i, j] = (backward_moms_a_med[i, j] * d_out/2 / I_6 + backward_F_med*np.sin(angle))/E_alum
        backward_a_hi_I_6_strain[i, j] = (backward_moms_a_hi[i, j] * d_out/2 / I_6 + backward_F_hi*np.sin(angle))/E_alum
        backward_b_low_I_1_strain[i, j] = (backward_moms_b_low[i, j] * d_out/2 / I_1 + backward_F_low*np.sin(angle))/E_alum
        backward_b_med_I_1_strain[i, j] = (backward_moms_b_med[i, j] * d_out/2 / I_1 + backward_F_med*np.sin(angle))/E_alum
        backward_b_hi_I_1_strain[i, j] = (backward_moms_b_hi[i, j] * d_out/2 / I_1 + backward_F_hi*np.sin(angle))/E_alum
        backward_b_low_I_2_strain[i, j] = (backward_moms_b_low[i, j] * d_out/2 / I_2 + backward_F_low*np.sin(angle))/E_alum
        backward_b_med_I_2_strain[i, j] = (backward_moms_b_med[i, j] * d_out/2 / I_2 + backward_F_med*np.sin(angle))/E_alum
        backward_b_hi_I_2_strain[i, j] = (backward_moms_b_hi[i, j] * d_out/2 / I_2 + backward_F_hi*np.sin(angle))/E_alum
        backward_b_low_I_3_strain[i, j] = (backward_moms_b_low[i, j] * d_out/2 / I_3 + backward_F_low*np.sin(angle))/E_alum
        backward_b_med_I_3_strain[i, j] = (backward_moms_b_med[i, j] * d_out/2 / I_3 + backward_F_med*np.sin(angle))/E_alum
        backward_b_hi_I_3_strain[i, j] = (backward_moms_b_hi[i, j] * d_out/2 / I_3 + backward_F_hi*np.sin(angle))/E_alum
        backward_b_low_I_4_strain[i, j] = (backward_moms_b_low[i, j] * d_out/2 / I_4 + backward_F_low*np.sin(angle))/E_alum
        backward_b_med_I_4_strain[i, j] = (backward_moms_b_med[i, j] * d_out/2 / I_4 + backward_F_med*np.sin(angle))/E_alum
        backward_b_hi_I_4_strain[i, j] = (backward_moms_b_hi[i, j] * d_out/2 / I_4 + backward_F_hi*np.sin(angle))/E_alum
        backward_b_low_I_5_strain[i, j] = (backward_moms_b_low[i, j] * d_out/2 / I_5 + backward_F_low*np.sin(angle))/E_alum
        backward_b_med_I_5_strain[i, j] = (backward_moms_b_med[i, j] * d_out/2 / I_5 + backward_F_med*np.sin(angle))/E_alum
        backward_b_hi_I_5_strain[i, j] = (backward_moms_b_hi[i, j] * d_out/2 / I_5 + backward_F_hi*np.sin(angle))/E_alum
        backward_b_low_I_6_strain[i, j] = (backward_moms_b_low[i, j] * d_out/2 / I_6 + backward_F_low*np.sin(angle))/E_alum
        backward_b_med_I_6_strain[i, j] = (backward_moms_b_med[i, j] * d_out/2 / I_6 + backward_F_med*np.sin(angle))/E_alum
        backward_b_hi_I_6_strain[i, j] = (backward_moms_b_hi[i, j] * d_out/2 / I_6 + backward_F_hi*np.sin(angle))/E_alum


        forward_delta = pre_delta + (rod_pos[j]-0.1)*np.sin(angle)
        forward_cap_theta, forward_L = fsolve(equations, np.array([.3, .75]), args=(forward_delta))
    
        forward_theta_0 = c_theta*forward_cap_theta
        #if i == 0:
        #    print(f'forward delta: {forward_delta}')
        #    print(f'forward position: {rod_pos[j]}')
        #    print(f'forward theta_0: {forward_theta_0*180/np.pi}')
    
        forward_phi = forward_theta_0 + np.pi/2
        forward_n = -1/np.tan(forward_phi)
        
        forward_F_low = EI_stalk_low*forward_cap_theta*gamma*k_theta/(forward_L*(height + forward_n*forward_delta))
        forward_F_med = EI_stalk_med*forward_cap_theta*gamma*k_theta/(forward_L*(height + forward_n*forward_delta))
        forward_F_hi = EI_stalk_hi*forward_cap_theta*gamma*k_theta/(forward_L*(height + forward_n*forward_delta))
        forward_Fp_low = forward_F_low*np.cos(angle)
        forward_Fp_med = forward_F_med*np.cos(angle)
        forward_Fp_hi = forward_F_hi*np.cos(angle)
        if i == 0:
            print(f'Fp_hi: {forward_Fp_hi}')

        forward_moms_a_low[i, j] = forward_Fp_low * (rod_pos[j]-0.01651)
        forward_moms_a_med[i, j] = forward_Fp_med * (rod_pos[j]-0.01651)
        forward_moms_a_hi[i, j] = forward_Fp_hi * (rod_pos[j]-0.01651)
        forward_moms_b_low[i, j] = forward_Fp_low * (rod_pos[j]-0.0635)
        forward_moms_b_med[i, j] = forward_Fp_med * (rod_pos[j]-0.0635)
        forward_moms_b_hi[i, j] = forward_Fp_hi * (rod_pos[j]-0.0635)

        forward_a_low_I_1_strain[i, j] = (forward_moms_a_low[i, j] * d_out/2 / I_1 - forward_F_low*np.sin(angle))/E_alum
        forward_a_med_I_1_strain[i, j] = (forward_moms_a_med[i, j] * d_out/2 / I_1 - forward_F_med*np.sin(angle))/E_alum
        forward_a_hi_I_1_strain[i, j] = (forward_moms_a_hi[i, j] * d_out/2 / I_1 - forward_F_hi*np.sin(angle))/E_alum
        forward_a_low_I_2_strain[i, j] = (forward_moms_a_low[i, j] * d_out/2 / I_2 - forward_F_low*np.sin(angle))/E_alum
        forward_a_med_I_2_strain[i, j] = (forward_moms_a_med[i, j] * d_out/2 / I_2 - forward_F_med*np.sin(angle))/E_alum
        forward_a_hi_I_2_strain[i, j] = (forward_moms_a_hi[i, j] * d_out/2 / I_2 - forward_F_hi*np.sin(angle))/E_alum
        forward_a_low_I_3_strain[i, j] = (forward_moms_a_low[i, j] * d_out/2 / I_3 - forward_F_low*np.sin(angle))/E_alum
        forward_a_med_I_3_strain[i, j] = (forward_moms_a_med[i, j] * d_out/2 / I_3 - forward_F_med*np.sin(angle))/E_alum
        forward_a_hi_I_3_strain[i, j] = (forward_moms_a_hi[i, j] * d_out/2 / I_3 - forward_F_hi*np.sin(angle))/E_alum
        forward_a_low_I_4_strain[i, j] = (forward_moms_a_low[i, j] * d_out/2 / I_4 - forward_F_low*np.sin(angle))/E_alum
        forward_a_med_I_4_strain[i, j] = (forward_moms_a_med[i, j] * d_out/2 / I_4 - forward_F_med*np.sin(angle))/E_alum
        forward_a_hi_I_4_strain[i, j] = (forward_moms_a_hi[i, j] * d_out/2 / I_4 - forward_F_hi*np.sin(angle))/E_alum
        forward_a_low_I_5_strain[i, j] = (forward_moms_a_low[i, j] * d_out/2 / I_5 - forward_F_low*np.sin(angle))/E_alum
        forward_a_med_I_5_strain[i, j] = (forward_moms_a_med[i, j] * d_out/2 / I_5 - forward_F_med*np.sin(angle))/E_alum
        forward_a_hi_I_5_strain[i, j] = (forward_moms_a_hi[i, j] * d_out/2 / I_5 - forward_F_hi*np.sin(angle))/E_alum
        forward_a_low_I_6_strain[i, j] = (forward_moms_a_low[i, j] * d_out/2 / I_6 - forward_F_low*np.sin(angle))/E_alum
        forward_a_med_I_6_strain[i, j] = (forward_moms_a_med[i, j] * d_out/2 / I_6 - forward_F_med*np.sin(angle))/E_alum
        forward_a_hi_I_6_strain[i, j] = (forward_moms_a_hi[i, j] * d_out/2 / I_6 - forward_F_hi*np.sin(angle))/E_alum
        forward_b_low_I_1_strain[i, j] = (forward_moms_b_low[i, j] * d_out/2 / I_1 - forward_F_low*np.sin(angle))/E_alum
        forward_b_med_I_1_strain[i, j] = (forward_moms_b_med[i, j] * d_out/2 / I_1 - forward_F_med*np.sin(angle))/E_alum
        forward_b_hi_I_1_strain[i, j] = (forward_moms_b_hi[i, j] * d_out/2 / I_1 - forward_F_hi*np.sin(angle))/E_alum
        forward_b_low_I_2_strain[i, j] = (forward_moms_b_low[i, j] * d_out/2 / I_2 - forward_F_low*np.sin(angle))/E_alum
        forward_b_med_I_2_strain[i, j] = (forward_moms_b_med[i, j] * d_out/2 / I_2 - forward_F_med*np.sin(angle))/E_alum
        forward_b_hi_I_2_strain[i, j] = (forward_moms_b_hi[i, j] * d_out/2 / I_2 - forward_F_hi*np.sin(angle))/E_alum
        forward_b_low_I_3_strain[i, j] = (forward_moms_b_low[i, j] * d_out/2 / I_3 - forward_F_low*np.sin(angle))/E_alum
        forward_b_med_I_3_strain[i, j] = (forward_moms_b_med[i, j] * d_out/2 / I_3 - forward_F_med*np.sin(angle))/E_alum
        forward_b_hi_I_3_strain[i, j] = (forward_moms_b_hi[i, j] * d_out/2 / I_3 - forward_F_hi*np.sin(angle))/E_alum
        forward_b_low_I_4_strain[i, j] = (forward_moms_b_low[i, j] * d_out/2 / I_4 - forward_F_low*np.sin(angle))/E_alum
        forward_b_med_I_4_strain[i, j] = (forward_moms_b_med[i, j] * d_out/2 / I_4 - forward_F_med*np.sin(angle))/E_alum
        forward_b_hi_I_4_strain[i, j] = (forward_moms_b_hi[i, j] * d_out/2 / I_4 - forward_F_hi*np.sin(angle))/E_alum
        forward_b_low_I_5_strain[i, j] = (forward_moms_b_low[i, j] * d_out/2 / I_5 - forward_F_low*np.sin(angle))/E_alum
        forward_b_med_I_5_strain[i, j] = (forward_moms_b_med[i, j] * d_out/2 / I_5 - forward_F_med*np.sin(angle))/E_alum
        forward_b_hi_I_5_strain[i, j] = (forward_moms_b_hi[i, j] * d_out/2 / I_5 - forward_F_hi*np.sin(angle))/E_alum
        forward_b_low_I_6_strain[i, j] = (forward_moms_b_low[i, j] * d_out/2 / I_6 - forward_F_low*np.sin(angle))/E_alum
        forward_b_med_I_6_strain[i, j] = (forward_moms_b_med[i, j] * d_out/2 / I_6 - forward_F_med*np.sin(angle))/E_alum
        forward_b_hi_I_6_strain[i, j] = (forward_moms_b_hi[i, j] * d_out/2 / I_6 - forward_F_hi*np.sin(angle))/E_alum

        print(I_1)


plt.plot(rod_pos*100, backward_moms_a_low[0], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[0], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[0], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[0], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[0], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[0], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 0 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos0Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[1], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[1], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[1], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[1], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[1], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[1], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 5 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos5Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[2], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[2], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[2], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[2], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[2], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[2], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 10 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos10Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[3], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[3], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[3], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[3], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[3], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[3], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 15 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos15Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[4], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[4], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[4], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[4], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[4], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[4], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 20 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos20Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[5], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[5], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[5], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[5], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[5], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[5], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 25 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos25Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[6], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[6], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[6], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[6], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[6], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[6], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 30 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos30Degrees.png')
plt.show()

plt.plot(rod_pos*100, backward_moms_a_low[7], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_low[7], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_med[7], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_med[7], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, backward_moms_a_hi[7], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, backward_moms_b_hi[7], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Backward Deflection Moment vs Position at 35 Degrees')
plt.legend()
plt.savefig('BackwardMomVsPos35Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[0], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[0], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[0], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[0], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[0], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[0], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 0 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos0Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[1], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[1], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[1], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[1], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[1], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[1], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 5 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos5Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[2], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[2], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[2], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[2], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[2], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[2], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 10 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos10Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[3], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[3], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[3], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[3], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[3], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[3], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 15 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos15Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[4], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[4], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[4], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[4], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[4], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[4], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 20 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos20Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[5], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[5], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[5], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[5], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[5], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[5], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 25 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos25Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[6], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[6], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[6], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[6], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[6], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[6], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 30 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos30Degrees.png')
plt.show()

plt.plot(rod_pos*100, forward_moms_a_low[7], label='Moment at A Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_low[7], label='Moment at B Gauges with Stiffness of 10 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_med[7], label='Moment at A Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_med[7], label='Moment at B Gauges with Stiffness of 20 N*m^2')
plt.plot(rod_pos*100, forward_moms_a_hi[7], label='Moment at A Gauges with Stiffness of 40 N*m^2')
plt.plot(rod_pos*100, forward_moms_b_hi[7], label='Moment at B Gauges with Stiffness of 40 N*m^2')
plt.xlabel('Position (cm)')
plt.ylabel('Moment (Nm)')
plt.title('Forward Deflection Moment vs Position at 35 Degrees')
plt.legend()
plt.savefig('forwardMomVsPos35Degrees.png')
plt.show()


### Strain Plots ###
fig1, axs1 = plt.subplots(2, 2, sharex=True, sharey=True)
fig1.suptitle('Strain Vs Position with 0.02" Wall Thickness', fontsize=16)

axs1[0, 0].plot(rod_pos*100, backward_a_low_I_1_strain[0], label='A Strain with 10 Nm^2')
axs1[0, 0].plot(rod_pos*100, backward_b_low_I_1_strain[0], label='B Strain with 10 Nm^2')
axs1[0, 0].plot(rod_pos*100, backward_a_med_I_1_strain[0], label='A Strain with 20 Nm^2')
axs1[0, 0].plot(rod_pos*100, backward_b_med_I_1_strain[0], label='B Strain with 20 Nm^2')
axs1[0, 0].plot(rod_pos*100, backward_a_hi_I_1_strain[0], label='A Strain with 40 Nm^2')
axs1[0, 0].plot(rod_pos*100, backward_b_hi_I_1_strain[0], label='B Strain with 40 Nm^2')
axs1[0, 0].set_title('0 Degrees')
axs1[0, 1].plot(rod_pos*100, backward_a_low_I_1_strain[7], label='A Strain with 10 Nm^2')
axs1[0, 1].plot(rod_pos*100, backward_b_low_I_1_strain[7], label='B Strain with 10 Nm^2')
axs1[0, 1].plot(rod_pos*100, backward_a_med_I_1_strain[7], label='A Strain with 20 Nm^2')
axs1[0, 1].plot(rod_pos*100, backward_b_med_I_1_strain[7], label='B Strain with 20 Nm^2')
axs1[0, 1].plot(rod_pos*100, backward_a_hi_I_1_strain[7], label='A Strain with 40 Nm^2')
axs1[0, 1].plot(rod_pos*100, backward_b_hi_I_1_strain[7], label='B Strain with 40 Nm^2')
axs1[0, 1].set_title('35 Degrees')
axs1[1, 0].plot(rod_pos*100, forward_a_low_I_1_strain[0], label='A Strain with 10 Nm^2')
axs1[1, 0].plot(rod_pos*100, forward_b_low_I_1_strain[0], label='B Strain with 10 Nm^2')
axs1[1, 0].plot(rod_pos*100, forward_a_med_I_1_strain[0], label='A Strain with 20 Nm^2')
axs1[1, 0].plot(rod_pos*100, forward_b_med_I_1_strain[0], label='B Strain with 20 Nm^2')
axs1[1, 0].plot(rod_pos*100, forward_a_hi_I_1_strain[0], label='A Strain with 40 Nm^2')
axs1[1, 0].plot(rod_pos*100, forward_b_hi_I_1_strain[0], label='B Strain with 40 Nm^2')
axs1[1, 1].plot(rod_pos*100, forward_a_low_I_1_strain[7], label='A Strain with 10 Nm^2')
axs1[1, 1].plot(rod_pos*100, forward_b_low_I_1_strain[7], label='B Strain with 10 Nm^2')
axs1[1, 1].plot(rod_pos*100, forward_a_med_I_1_strain[7], label='A Strain with 20 Nm^2')
axs1[1, 1].plot(rod_pos*100, forward_b_med_I_1_strain[7], label='B Strain with 20 Nm^2')
axs1[1, 1].plot(rod_pos*100, forward_a_hi_I_1_strain[7], label='A Strain with 40 Nm^2')
axs1[1, 1].plot(rod_pos*100, forward_b_hi_I_1_strain[7], label='B Strain with 40 Nm^2')

axs1[0, 0].set(ylabel='Backward')
axs1[1, 0].set(ylabel='Forward')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs1.flat:
    ax.label_outer()

# Create a single legend for the entire figure
handles_1, labels_1 = axs1[0, 0].get_legend_handles_labels()
fig1.legend(handles_1, labels_1, loc='upper right', )

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, .9])  # Adjust rect to make space for the legend
#fig1.text()
plt.show()

fig2, axs2 = plt.subplots(2, 2, sharex=True, sharey=True)
fig2.suptitle('Strain Vs Position with 0.035" Wall Thickness', fontsize=16)

axs2[0, 0].plot(rod_pos*100, backward_a_low_I_2_strain[0], label='A Strain with 10 Nm^2')
axs2[0, 0].plot(rod_pos*100, backward_b_low_I_2_strain[0], label='B Strain with 10 Nm^2')
axs2[0, 0].plot(rod_pos*100, backward_a_med_I_2_strain[0], label='A Strain with 20 Nm^2')
axs2[0, 0].plot(rod_pos*100, backward_b_med_I_2_strain[0], label='B Strain with 20 Nm^2')
axs2[0, 0].plot(rod_pos*100, backward_a_hi_I_2_strain[0], label='A Strain with 40 Nm^2')
axs2[0, 0].plot(rod_pos*100, backward_b_hi_I_2_strain[0], label='B Strain with 40 Nm^2')
axs2[0, 0].set_title('0 Degrees')
axs2[0, 1].plot(rod_pos*100, backward_a_low_I_2_strain[7], label='A Strain with 10 Nm^2')
axs2[0, 1].plot(rod_pos*100, backward_b_low_I_2_strain[7], label='B Strain with 10 Nm^2')
axs2[0, 1].plot(rod_pos*100, backward_a_med_I_2_strain[7], label='A Strain with 20 Nm^2')
axs2[0, 1].plot(rod_pos*100, backward_b_med_I_2_strain[7], label='B Strain with 20 Nm^2')
axs2[0, 1].plot(rod_pos*100, backward_a_hi_I_2_strain[7], label='A Strain with 40 Nm^2')
axs2[0, 1].plot(rod_pos*100, backward_b_hi_I_2_strain[7], label='B Strain with 40 Nm^2')
axs2[0, 1].set_title('35 Degrees')
axs2[1, 0].plot(rod_pos*100, forward_a_low_I_2_strain[0], label='A Strain with 10 Nm^2')
axs2[1, 0].plot(rod_pos*100, forward_b_low_I_2_strain[0], label='B Strain with 10 Nm^2')
axs2[1, 0].plot(rod_pos*100, forward_a_med_I_2_strain[0], label='A Strain with 20 Nm^2')
axs2[1, 0].plot(rod_pos*100, forward_b_med_I_2_strain[0], label='B Strain with 20 Nm^2')
axs2[1, 0].plot(rod_pos*100, forward_a_hi_I_2_strain[0], label='A Strain with 40 Nm^2')
axs2[1, 0].plot(rod_pos*100, forward_b_hi_I_2_strain[0], label='B Strain with 40 Nm^2')
axs2[1, 1].plot(rod_pos*100, forward_a_low_I_2_strain[7], label='A Strain with 10 Nm^2')
axs2[1, 1].plot(rod_pos*100, forward_b_low_I_2_strain[7], label='B Strain with 10 Nm^2')
axs2[1, 1].plot(rod_pos*100, forward_a_med_I_2_strain[7], label='A Strain with 20 Nm^2')
axs2[1, 1].plot(rod_pos*100, forward_b_med_I_2_strain[7], label='B Strain with 20 Nm^2')
axs2[1, 1].plot(rod_pos*100, forward_a_hi_I_2_strain[7], label='A Strain with 40 Nm^2')
axs2[1, 1].plot(rod_pos*100, forward_b_hi_I_2_strain[7], label='B Strain with 40 Nm^2')

axs2[0, 0].set(ylabel='Backward')
axs2[1, 0].set(ylabel='Forward')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs2.flat:
    ax.label_outer()

# Create a single legend for the entire figure
handles_2, labels_2 = axs2[0, 0].get_legend_handles_labels()
fig2.legend(handles_2, labels_2, loc='upper right', )

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, .9])  # Adjust rect to make space for the legend
#fig1.text()
plt.show()

fig3, axs3 = plt.subplots(2, 2, sharex=True, sharey=True)
fig3.suptitle('Strain Vs Position with 0.049" Wall Thickness', fontsize=16)

axs3[0, 0].plot(rod_pos*100, backward_a_low_I_3_strain[0], label='A Strain with 10 Nm^2')
axs3[0, 0].plot(rod_pos*100, backward_b_low_I_3_strain[0], label='B Strain with 10 Nm^2')
axs3[0, 0].plot(rod_pos*100, backward_a_med_I_3_strain[0], label='A Strain with 20 Nm^2')
axs3[0, 0].plot(rod_pos*100, backward_b_med_I_3_strain[0], label='B Strain with 20 Nm^2')
axs3[0, 0].plot(rod_pos*100, backward_a_hi_I_3_strain[0], label='A Strain with 40 Nm^2')
axs3[0, 0].plot(rod_pos*100, backward_b_hi_I_3_strain[0], label='B Strain with 40 Nm^2')
axs3[0, 0].set_title('0 Degrees')
axs3[0, 1].plot(rod_pos*100, backward_a_low_I_3_strain[7], label='A Strain with 10 Nm^2')
axs3[0, 1].plot(rod_pos*100, backward_b_low_I_3_strain[7], label='B Strain with 10 Nm^2')
axs3[0, 1].plot(rod_pos*100, backward_a_med_I_3_strain[7], label='A Strain with 20 Nm^2')
axs3[0, 1].plot(rod_pos*100, backward_b_med_I_3_strain[7], label='B Strain with 20 Nm^2')
axs3[0, 1].plot(rod_pos*100, backward_a_hi_I_3_strain[7], label='A Strain with 40 Nm^2')
axs3[0, 1].plot(rod_pos*100, backward_b_hi_I_3_strain[7], label='B Strain with 40 Nm^2')
axs3[0, 1].set_title('35 Degrees')
axs3[1, 0].plot(rod_pos*100, forward_a_low_I_3_strain[0], label='A Strain with 10 Nm^2')
axs3[1, 0].plot(rod_pos*100, forward_b_low_I_3_strain[0], label='B Strain with 10 Nm^2')
axs3[1, 0].plot(rod_pos*100, forward_a_med_I_3_strain[0], label='A Strain with 20 Nm^2')
axs3[1, 0].plot(rod_pos*100, forward_b_med_I_3_strain[0], label='B Strain with 20 Nm^2')
axs3[1, 0].plot(rod_pos*100, forward_a_hi_I_3_strain[0], label='A Strain with 40 Nm^2')
axs3[1, 0].plot(rod_pos*100, forward_b_hi_I_3_strain[0], label='B Strain with 40 Nm^2')
axs3[1, 1].plot(rod_pos*100, forward_a_low_I_3_strain[7], label='A Strain with 10 Nm^2')
axs3[1, 1].plot(rod_pos*100, forward_b_low_I_3_strain[7], label='B Strain with 10 Nm^2')
axs3[1, 1].plot(rod_pos*100, forward_a_med_I_3_strain[7], label='A Strain with 20 Nm^2')
axs3[1, 1].plot(rod_pos*100, forward_b_med_I_3_strain[7], label='B Strain with 20 Nm^2')
axs3[1, 1].plot(rod_pos*100, forward_a_hi_I_3_strain[7], label='A Strain with 40 Nm^2')
axs3[1, 1].plot(rod_pos*100, forward_b_hi_I_3_strain[7], label='B Strain with 40 Nm^2')

axs3[0, 0].set(ylabel='Backward')
axs3[1, 0].set(ylabel='Forward')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs3.flat:
    ax.label_outer()

# Create a single legend for the entire figure
handles_3, labels_3 = axs3[0, 0].get_legend_handles_labels()
fig3.legend(handles_3, labels_3, loc='upper right', )

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, .9])  # Adjust rect to make space for the legend
#fig1.text()
plt.show()

fig4, axs4 = plt.subplots(2, 2, sharex=True, sharey=True)
fig4.suptitle('Strain Vs Position with 0.058" Wall Thickness', fontsize=16)

axs4[0, 0].plot(rod_pos*100, backward_a_low_I_4_strain[0], label='A Strain with 10 Nm^2')
axs4[0, 0].plot(rod_pos*100, backward_b_low_I_4_strain[0], label='B Strain with 10 Nm^2')
axs4[0, 0].plot(rod_pos*100, backward_a_med_I_4_strain[0], label='A Strain with 20 Nm^2')
axs4[0, 0].plot(rod_pos*100, backward_b_med_I_4_strain[0], label='B Strain with 20 Nm^2')
axs4[0, 0].plot(rod_pos*100, backward_a_hi_I_4_strain[0], label='A Strain with 40 Nm^2')
axs4[0, 0].plot(rod_pos*100, backward_b_hi_I_4_strain[0], label='B Strain with 40 Nm^2')
axs4[0, 0].set_title('0 Degrees')
axs4[0, 1].plot(rod_pos*100, backward_a_low_I_4_strain[7], label='A Strain with 10 Nm^2')
axs4[0, 1].plot(rod_pos*100, backward_b_low_I_4_strain[7], label='B Strain with 10 Nm^2')
axs4[0, 1].plot(rod_pos*100, backward_a_med_I_4_strain[7], label='A Strain with 20 Nm^2')
axs4[0, 1].plot(rod_pos*100, backward_b_med_I_4_strain[7], label='B Strain with 20 Nm^2')
axs4[0, 1].plot(rod_pos*100, backward_a_hi_I_4_strain[7], label='A Strain with 40 Nm^2')
axs4[0, 1].plot(rod_pos*100, backward_b_hi_I_4_strain[7], label='B Strain with 40 Nm^2')
axs4[0, 1].set_title('35 Degrees')
axs4[1, 0].plot(rod_pos*100, forward_a_low_I_4_strain[0], label='A Strain with 10 Nm^2')
axs4[1, 0].plot(rod_pos*100, forward_b_low_I_4_strain[0], label='B Strain with 10 Nm^2')
axs4[1, 0].plot(rod_pos*100, forward_a_med_I_4_strain[0], label='A Strain with 20 Nm^2')
axs4[1, 0].plot(rod_pos*100, forward_b_med_I_4_strain[0], label='B Strain with 20 Nm^2')
axs4[1, 0].plot(rod_pos*100, forward_a_hi_I_4_strain[0], label='A Strain with 40 Nm^2')
axs4[1, 0].plot(rod_pos*100, forward_b_hi_I_4_strain[0], label='B Strain with 40 Nm^2')
axs4[1, 1].plot(rod_pos*100, forward_a_low_I_4_strain[7], label='A Strain with 10 Nm^2')
axs4[1, 1].plot(rod_pos*100, forward_b_low_I_4_strain[7], label='B Strain with 10 Nm^2')
axs4[1, 1].plot(rod_pos*100, forward_a_med_I_4_strain[7], label='A Strain with 20 Nm^2')
axs4[1, 1].plot(rod_pos*100, forward_b_med_I_4_strain[7], label='B Strain with 20 Nm^2')
axs4[1, 1].plot(rod_pos*100, forward_a_hi_I_4_strain[7], label='A Strain with 40 Nm^2')
axs4[1, 1].plot(rod_pos*100, forward_b_hi_I_4_strain[7], label='B Strain with 40 Nm^2')

axs4[0, 0].set(ylabel='Backward')
axs4[1, 0].set(ylabel='Forward')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs4.flat:
    ax.label_outer()

# Create a single legend for the entire figure
handles_4, labels_4 = axs4[0, 0].get_legend_handles_labels()
fig4.legend(handles_4, labels_4, loc='upper right', )

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, .9])  # Adjust rect to make space for the legend
#fig1.text()
plt.show()

fig5, axs5 = plt.subplots(2, 2, sharex=True, sharey=True)
fig5.suptitle('Strain Vs Position with 0.065" Wall Thickness', fontsize=16)

axs5[0, 0].plot(rod_pos*100, backward_a_low_I_5_strain[0], label='A Strain with 10 Nm^2')
axs5[0, 0].plot(rod_pos*100, backward_b_low_I_5_strain[0], label='B Strain with 10 Nm^2')
axs5[0, 0].plot(rod_pos*100, backward_a_med_I_5_strain[0], label='A Strain with 20 Nm^2')
axs5[0, 0].plot(rod_pos*100, backward_b_med_I_5_strain[0], label='B Strain with 20 Nm^2')
axs5[0, 0].plot(rod_pos*100, backward_a_hi_I_5_strain[0], label='A Strain with 40 Nm^2')
axs5[0, 0].plot(rod_pos*100, backward_b_hi_I_5_strain[0], label='B Strain with 40 Nm^2')
axs5[0, 0].set_title('0 Degrees')
axs5[0, 1].plot(rod_pos*100, backward_a_low_I_5_strain[7], label='A Strain with 10 Nm^2')
axs5[0, 1].plot(rod_pos*100, backward_b_low_I_5_strain[7], label='B Strain with 10 Nm^2')
axs5[0, 1].plot(rod_pos*100, backward_a_med_I_5_strain[7], label='A Strain with 20 Nm^2')
axs5[0, 1].plot(rod_pos*100, backward_b_med_I_5_strain[7], label='B Strain with 20 Nm^2')
axs5[0, 1].plot(rod_pos*100, backward_a_hi_I_5_strain[7], label='A Strain with 40 Nm^2')
axs5[0, 1].plot(rod_pos*100, backward_b_hi_I_5_strain[7], label='B Strain with 40 Nm^2')
axs5[0, 1].set_title('35 Degrees')
axs5[1, 0].plot(rod_pos*100, forward_a_low_I_5_strain[0], label='A Strain with 10 Nm^2')
axs5[1, 0].plot(rod_pos*100, forward_b_low_I_5_strain[0], label='B Strain with 10 Nm^2')
axs5[1, 0].plot(rod_pos*100, forward_a_med_I_5_strain[0], label='A Strain with 20 Nm^2')
axs5[1, 0].plot(rod_pos*100, forward_b_med_I_5_strain[0], label='B Strain with 20 Nm^2')
axs5[1, 0].plot(rod_pos*100, forward_a_hi_I_5_strain[0], label='A Strain with 40 Nm^2')
axs5[1, 0].plot(rod_pos*100, forward_b_hi_I_5_strain[0], label='B Strain with 40 Nm^2')
axs5[1, 1].plot(rod_pos*100, forward_a_low_I_5_strain[7], label='A Strain with 10 Nm^2')
axs5[1, 1].plot(rod_pos*100, forward_b_low_I_5_strain[7], label='B Strain with 10 Nm^2')
axs5[1, 1].plot(rod_pos*100, forward_a_med_I_5_strain[7], label='A Strain with 20 Nm^2')
axs5[1, 1].plot(rod_pos*100, forward_b_med_I_5_strain[7], label='B Strain with 20 Nm^2')
axs5[1, 1].plot(rod_pos*100, forward_a_hi_I_5_strain[7], label='A Strain with 40 Nm^2')
axs5[1, 1].plot(rod_pos*100, forward_b_hi_I_5_strain[7], label='B Strain with 40 Nm^2')

axs5[0, 0].set(ylabel='Backward')
axs5[1, 0].set(ylabel='Forward')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs5.flat:
    ax.label_outer()

# Create a single legend for the entire figure
handles_5, labels_5 = axs5[0, 0].get_legend_handles_labels()
fig5.legend(handles_5, labels_5, loc='upper right', )

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, .9])  # Adjust rect to make space for the legend
#fig1.text()
plt.show()

fig6, axs6 = plt.subplots(2, 2, sharex=True, sharey=True)
fig6.suptitle('Strain Vs Position with 0.08" Wall Thickness', fontsize=16)

axs6[0, 0].plot(rod_pos*100, backward_a_low_I_6_strain[0], label='A Strain with 10 Nm^2')
axs6[0, 0].plot(rod_pos*100, backward_b_low_I_6_strain[0], label='B Strain with 10 Nm^2')
axs6[0, 0].plot(rod_pos*100, backward_a_med_I_6_strain[0], label='A Strain with 20 Nm^2')
axs6[0, 0].plot(rod_pos*100, backward_b_med_I_6_strain[0], label='B Strain with 20 Nm^2')
axs6[0, 0].plot(rod_pos*100, backward_a_hi_I_6_strain[0], label='A Strain with 40 Nm^2')
axs6[0, 0].plot(rod_pos*100, backward_b_hi_I_6_strain[0], label='B Strain with 40 Nm^2')
axs6[0, 0].set_title('0 Degrees')
axs6[0, 1].plot(rod_pos*100, backward_a_low_I_6_strain[7], label='A Strain with 10 Nm^2')
axs6[0, 1].plot(rod_pos*100, backward_b_low_I_6_strain[7], label='B Strain with 10 Nm^2')
axs6[0, 1].plot(rod_pos*100, backward_a_med_I_6_strain[7], label='A Strain with 20 Nm^2')
axs6[0, 1].plot(rod_pos*100, backward_b_med_I_6_strain[7], label='B Strain with 20 Nm^2')
axs6[0, 1].plot(rod_pos*100, backward_a_hi_I_6_strain[7], label='A Strain with 40 Nm^2')
axs6[0, 1].plot(rod_pos*100, backward_b_hi_I_6_strain[7], label='B Strain with 40 Nm^2')
axs6[0, 1].set_title('35 Degrees')
axs6[1, 0].plot(rod_pos*100, forward_a_low_I_6_strain[0], label='A Strain with 10 Nm^2')
axs6[1, 0].plot(rod_pos*100, forward_b_low_I_6_strain[0], label='B Strain with 10 Nm^2')
axs6[1, 0].plot(rod_pos*100, forward_a_med_I_6_strain[0], label='A Strain with 20 Nm^2')
axs6[1, 0].plot(rod_pos*100, forward_b_med_I_6_strain[0], label='B Strain with 20 Nm^2')
axs6[1, 0].plot(rod_pos*100, forward_a_hi_I_6_strain[0], label='A Strain with 40 Nm^2')
axs6[1, 0].plot(rod_pos*100, forward_b_hi_I_6_strain[0], label='B Strain with 40 Nm^2')
axs6[1, 1].plot(rod_pos*100, forward_a_low_I_6_strain[7], label='A Strain with 10 Nm^2')
axs6[1, 1].plot(rod_pos*100, forward_b_low_I_6_strain[7], label='B Strain with 10 Nm^2')
axs6[1, 1].plot(rod_pos*100, forward_a_med_I_6_strain[7], label='A Strain with 20 Nm^2')
axs6[1, 1].plot(rod_pos*100, forward_b_med_I_6_strain[7], label='B Strain with 20 Nm^2')
axs6[1, 1].plot(rod_pos*100, forward_a_hi_I_6_strain[7], label='A Strain with 40 Nm^2')
axs6[1, 1].plot(rod_pos*100, forward_b_hi_I_6_strain[7], label='B Strain with 40 Nm^2')

axs6[0, 0].set(ylabel='Backward')
axs6[1, 0].set(ylabel='Forward')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs6.flat:
    ax.label_outer()

# Create a single legend for the entire figure
handles_6, labels_6 = axs6[0, 0].get_legend_handles_labels()
fig6.legend(handles_6, labels_6, loc='upper right', )

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, .9])  # Adjust rect to make space for the legend
#fig1.text()
plt.show()