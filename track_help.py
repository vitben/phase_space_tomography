import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cpymad.madx import Madx
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import curve_fit
# from cpymad import libmadx
import copy
import pickle
import os


def get_twiss(data, plane):
    """
    The get_twiss function takes a dataframe and a plane as input.
    It returns the emittance, alpha, beta and gamma of that plane.


    :param data: Specify which data is used to calculate the twiss parameters
    :param plane: Specify which plane to calculate the twiss parameters for
    :return: The emittance, alpha, beta and gamma for the given plane
    :doc-author: Trelent
    """
    u = data[plane]
    pu = data["p" + plane]
    mu_u = np.mean(u)
    mu_pu = np.mean(pu)
    s_u = np.std(u)
    s_pu = np.std(pu)

    cross = np.mean((u - mu_u) * (pu - mu_pu))
    eg = np.sqrt(s_u ** 2 * s_pu ** 2 - cross ** 2) 
    beta = np.mean(u**2)/eg
    alpha = -np.mean(u*pu)/eg
    gamma = (1+alpha**2)/beta
    return eg, alpha, beta, gamma



def gauss(x ,x0, sigma):
    return np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def scale_emit(s, data, plane, goal_p, opt = True):
    scale = s[0]
    beam_s = data.copy()
    if plane == 'x':
        beam_s['x'] = data['x']*scale
        beam_s['px'] = data['px']*scale
        e, alfx, betx, gammx = get_twiss(beam_s, plane)
    elif plane == 'y':
        beam_s['y'] = data['y']*scale
        beam_s['py'] = data['py']*scale
        e, alfx, betx, gammx = get_twiss(beam_s, plane)
    elif  plane == 't':
        beam_s['pt'] = data['pt']*scale
        e = np.std(data['pt'])*scale
    goal  = abs(e*1e10-goal_p*1e10)
    if opt == True:
        return goal
    else:
        return beam_s

def shift_twiss(beam_m, goal_twiss):
    """
    Description: Transform beam distribution to goal Twiss parameters

    Input: - beam_m [DataFrame] - Input beam distribution with columns ['x', 'y', 'px', 'py', 't', 'pt'] (madx units)
        - goal_twiss - [dict] - goal twiss parameters in the form: 
        ['alfx':[], 'betx':[], 'alfx':[], 'betx':[], 'epsx':[], 'epsy':[], 'dpp':[]]
        emittances are geometrical
    Return: beam_out [DataFrame] - tranformed distribution 
    """

    ex, alfx, betx, gammx = get_twiss(beam_m, 'x')
    ey, alfy, bety, gammy = get_twiss(beam_m, 'y')
    gammtx = (1+goal_twiss['alfx']**2)/goal_twiss['betx']
    gammty = (1+goal_twiss['alfy']**2)/goal_twiss['bety']
    scale_target = [goal_twiss['epsx'], goal_twiss['epsy'], goal_twiss['dpp']]
    twiss_target_x = [goal_twiss['alfx'], goal_twiss['betx'], gammtx]
    twiss_target_y = [goal_twiss['alfy'], goal_twiss['bety'], gammty]
    twiss_init_x = [alfx, betx, gammx]
    twiss_init_y = [alfy, bety, gammy]


    res_emit_x = minimize(scale_emit, 
                        x0 = [1], 
                        method = 'Nelder-Mead', 
                        args  = (beam_m, 'x', scale_target[0]))
    res_emit_y = minimize(scale_emit, 
                    x0 = [1], 
                    method = 'Nelder-Mead', 
                    args  = (beam_m, 'y', scale_target[1]))
    res_emit_p = minimize(scale_emit, 
                    x0 = [1], 
                    method = 'Nelder-Mead', 
                    args  = (beam_m, 't',scale_target[2]))
    

    beam_m = scale_emit(res_emit_x.x,
                        beam_m, 'x', scale_target[0] , opt = False)
    beam_m = scale_emit(res_emit_y.x,
                        beam_m, 'y', scale_target[1] , opt = False)
    beam_m = scale_emit(res_emit_p.x,
                        beam_m, 't', scale_target[2] , opt = False)
    # Change alpha and betas

    def func_t(tw, goal_t, goal_i, opt = True):
        c = tw[0]
        s = tw[1]
        ci = tw[2]
        si = tw[3]

        eq1 = c**2*goal_i[1]-2*c*s*goal_i[0]+s**2*goal_i[2]-goal_t[1]
        eq2 = -c*ci*goal_i[1]+(s*ci+si*c)*goal_i[0]-s*si*goal_i[2]-goal_t[0]
        eq3 = ci**2*goal_i[1]-2*ci*si*goal_i[0]+si**2*goal_i[2]-goal_t[2]
        goal = abs(eq1)+abs(eq2)+abs(eq3)
        # print(tw)
        
        return goal


    res_tx = minimize(func_t, x0 = [1, 0, 0, 1], method = 'Powell', tol = 1e-6, args = (twiss_target_x,twiss_init_x), options = {'maxiter': 10000})
    # res_tx = minimize(func_t, x0 = res_tx.x, method = 'Powell', tol = 1e-6, args = (twiss_target_x,twiss_init_x))
    res_tx = minimize(func_t, x0 = res_tx.x, method = 'Nelder-Mead', tol = 1e-6, args = (twiss_target_x,twiss_init_x))
    res_ty = minimize(func_t, x0 = [1, 0, 0, 1], method = 'Powell', tol = 1e-6, args = (twiss_target_y,twiss_init_y))
    # res_ty = minimize(func_t, x0 = res_ty.x, method = 'Powell', tol = 1e-6, args = (twiss_target_y,twiss_init_y))
    res_ty = minimize(func_t, x0 = res_ty.x, method = 'Nelder-Mead', tol = 1e-6, args = (twiss_target_y,twiss_init_y))
    # print(res_tx.fun)
    # print(res_ty.fun)

    def change_twiss(data, resx, resy):
        out_t = data.copy()
        out_t['x'] = data['x']*resx[0]+data['px']*resx[1]
        out_t['px'] = data['x']*resx[2]+data['px']*resx[3]
        out_t['y'] = data['y']*resy[0]+data['py']*resy[1]
        out_t['py'] = data['y']*resy[2]+data['py']*resy[3]
    
    # def change_twiss_x(data, resx, resy):
    #     out_t = data.copy()
        
    #     out_t['x'] = data['x']*resx[0]+data['px']*resx[1]
    #     out_t['px'] = data['x']*resx[2]+data['px']*resx[3]
    #     out_t['y'] = data['y']*resy[0]+data['py']*resy[1]
    #     out_t['py'] = data['y']*resy[2]+data['py']*resy[3]
    
        return out_t
    beam_out = change_twiss(beam_m, res_tx.x, res_ty.x)
    # print(get_twiss(beam_out, 'x'))
    # print(twiss_target_x)
    # print(get_twiss(beam_out, 'y'))
    # print(twiss_target_y)
    return beam_out