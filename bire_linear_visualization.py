#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:31:17 2021

@author: christian
"""
import numpy as np
import aero_trim
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import json

trim_case = aero_trim.TrimCase(222., 0.0023084)
trim_case.import_aero_data("./misc/TODatabase_BIRE_body344.csv", model='bire', d_B=np.linspace(-90, 90, 44))
# trim_case.trim(climb_angle=0., bank_angle=0.0)

d_B = np.linspace(-90, 90, 44)
CL0 = 0.0876
CL_alpha = 4.0314
CL_qbar = 3.7263
CL_de = 0.6341
CS_beta = -0.5512
CS_pbar = 0.0165
CS_rbar = 0.6411
CS_da = 0.1011
CS_dr = 0.2052
CD0 = 0.0687
CD1 = 0.0029
CD2 = 0.1051
CD3 = 0.2148
CD_qbar = 0.0366
CD_de = -0.0077
Cell_beta = -0.0939
Cell_pbar = -0.4419
Cell_rbar = 0.0975
Cell_da = -0.1223
Cell_dr = 0.0332
Cm0 = -0.0048
Cm_alpha = -0.5889
Cm_qbar = -5.0267
Cm_de = -0.7826
Cn_beta = 0.2958
Cn_pbar = 0.0057
Cn_rbar = -0.3041
Cn_da = -0.0432
Cn_dr = -0.1071


# LIFT COEFFICIENT
def lift_fits():
    coefficients = {"CL0": {"Value": None,
                            "Type": "Sine"},
                    "CL_alpha": {"Value": None,
                                 "Type": "Sine"},
                    "CL_beta": {"Value": None,
                                "Type": "Sine"},
                    "CL_pbar": {"Value": None,
                                "Type": "Poly"},
                    "CL_qbar": {"Value": None,
                                "Type": "Sine"},
                    "CL_rbar": {"Value": None,
                                "Type": "Sine"},
                    "CL_da": {"Value": None,
                              "Type": "Sine"},
                    "CL_de": {"Value": None,
                              "Type": "Sine"}}
    d_Bfine = np.linspace(-90, 90, 200)
    fig, axs = plt.subplots(4, 2, sharex='col')
    fig.suptitle('BIRE Lift Coefficient Fits', fontsize=16)

    axs[0, 0].scatter(d_B, trim_case.CL0, label='BIRE Fits')
    axs[0, 0].axhline(CL0, c='r', label='Baseline Value')
    axs[0, 0].set_title(r"$C_{L0}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL0
    A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["CL0"]["Value"] = (A, w, phi, z)
    CL0_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 0].plot(d_Bfine, CL0_fit, c='g', label='Coefficient Fit')
    axs[0, 0].tick_params(axis='y', labelsize= 14)
    print(A, w, phi, z)

    axs[0, 1].scatter(d_B, trim_case.CL_alpha, label='BIRE Fits')
    axs[0, 1].axhline(CL_alpha, c='r', label='Baseline Value')
    axs[0, 1].set_title(r"$C_{L,\alpha}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL_alpha
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CL_alpha"]["Value"] = (A, w, phi, z)
    CLalpha_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 1].plot(d_Bfine, CLalpha_fit, c='g', label='Coefficient Fit')
    axs[0, 1].tick_params(axis='y', labelsize= 14)

    axs[1, 0].scatter(d_B, trim_case.CL_beta, label='BIRE Fits')
    axs[1, 0].axhline(0, c='r', label='Baseline Value')
    axs[1, 0].set_title(r"$C_{L,\beta}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL_beta
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CL_beta"]["Value"] = (A, w, phi, z)
    CLbeta_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 0].plot(d_Bfine, CLbeta_fit, c='g', label='Coefficient Fit')
    axs[1, 0].tick_params(axis='y', labelsize= 14)

    axs[1, 1].scatter(d_B, trim_case.CL_pbar, label='BIRE Fits')
    axs[1, 1].axhline(0, c='r', label='Baseline Value')
    axs[1, 1].set_title(r"$C_{L,\overline{p}}$", fontsize=14)
    CLpbar_fit = np.full(200, np.mean(trim_case.CL_pbar))
    coefficients["CL_pbar"]["Value"] = (A, w, phi, z)
    axs[1, 1].plot(d_Bfine, CLpbar_fit, c='g', label='Coefficient Fit')
    axs[1, 1].tick_params(axis='y', labelsize= 14)

    axs[2, 0].scatter(d_B, trim_case.CL_qbar, label='BIRE Fits')
    axs[2, 0].axhline(CL_qbar, c='r', label='Baseline Value')
    axs[2, 0].set_title(r"$C_{L,\overline{q}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL_qbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CL_qbar"]["Value"] = (A, w, phi, z)
    CLqbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 0].plot(d_Bfine, CLqbar_fit, c='g', label='Coefficient Fit')
    axs[2, 0].tick_params(axis='y', labelsize= 14)

    axs[2, 1].scatter(d_B, trim_case.CL_rbar, label='BIRE Fits')
    axs[2, 1].axhline(0, c='r', label='Baseline Value')
    axs[2, 1].set_title(r"$C_{L,\overline{r}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL_rbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CL_rbar"]["Value"] = (A, w, phi, z)
    CLrbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 1].plot(d_Bfine, CLrbar_fit, c='g', label='Coefficient Fit')
    axs[2, 1].tick_params(axis='y', labelsize= 14)

    axs[3, 0].scatter(d_B, trim_case.CL_da, label='BIRE Fits')
    axs[3, 0].axhline(0, c='r', label='Baseline Value')
    axs[3, 0].set_title(r"$C_{L,\delta_a}$", fontsize=14)
    axs[3, 0].set_xticks(np.arange(-90, 120, 30))
    axs[3, 0].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL_da
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CL_da"]["Value"] = (A, w, phi, z)
    CLda_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 0].plot(d_Bfine, CLda_fit, c='g', label='Coefficient Fit')
    axs[3, 0].tick_params(axis='y', labelsize= 14)
    axs[3, 0].tick_params(axis='x', labelsize= 14)

    axs[3, 1].scatter(d_B, trim_case.CL_de, label='BIRE Fits')
    axs[3, 1].axhline(CL_de, c='r', label='Baseline Value')
    axs[3, 1].set_title(r"$C_{L,\delta_e}$", fontsize=14)
    axs[3, 1].set_xticks(np.arange(-90, 120, 30))
    axs[3, 1].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CL_de
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CL_de"]["Value"] = (A, w, phi, z)
    CLde_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 1].plot(d_Bfine, CLde_fit, c='g', label='Coefficient Fit')
    axs[3, 1].tick_params(axis='y', labelsize= 14)
    axs[3, 1].tick_params(axis='x', labelsize= 14)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return coefficients

# DRAG COEFFICIENTS
def drag_fits():
    coefficients = {"CD0": {"Value": None,
                            "Type": "Sine"},
                    "CD1": {"Value": None,
                            "Type": "Poly"},
                    "CD2": {"Value": None,
                            "Type": "Poly"},
                    "CD3": {"Value": None,
                            "Type": "Sine"},
                    "CD_pbar": {"Value": None,
                                "Type": "Sine"},
                    "CD_qbar": {"Value": None,
                                "Type": "Sine"},
                    "CD_rbar": {"Value": None,
                                "Type": "Sine"},
                    "CD_da": {"Value": None,
                              "Type": "Sine"},
                    "CD_de": {"Value": None,
                              "Type": "Sine"}}
    d_Bfine = np.linspace(-90, 90, 200)
    fig, axs = plt.subplots(5, 2, sharex='col')
    fig.suptitle('BIRE Drag Coefficient Fits', fontsize=16)

    axs[0, 0].scatter(d_B, trim_case.CD0, label='BIRE Fits')
    axs[0, 0].axhline(CD0, c='r', label='Baseline Value')
    axs[0, 0].set_title(r"$C_{D0}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD0
    A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["CD0"]["Value"] = (A, w, phi, z)
    CD0_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 0].plot(d_Bfine, CD0_fit, c='g', label='Coefficient Fit')
    axs[0, 0].tick_params(axis='y', labelsize= 14)

    axs[0, 1].scatter(d_B, trim_case.CD1, label='BIRE Fits')
    axs[0, 1].axhline(CD1, c='r', label='Baseline Value')
    axs[0, 1].set_title(r"$C_{D1}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) + x[2]*np.power(d_B, 1) + x[3] - trim_case.CD1
    a3, a2, a1, a0 = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["CD1"]["Value"] = (a3, a2, a1, a0)
    CD1_fit = a3*np.power(d_Bfine, 3) + a2*np.power(d_Bfine, 2) + a1*np.power(d_Bfine, 1) + a0
    axs[0, 1].plot(d_Bfine, CD1_fit, c='g', label='Coefficient Fit')
    axs[0, 1].tick_params(axis='y', labelsize= 14)

    axs[1, 0].scatter(d_B, trim_case.CD2, label='BIRE Fits')
    axs[1, 0].axhline(CD2, c='r', label='Baseline Value')
    axs[1, 0].set_title(r"$C_{D2}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.power(d_B, 4) + x[1]*np.power(d_B, 3) + x[2]*np.power(d_B, 2) + x[3]*np.power(d_B, 1) + x[4] - trim_case.CD2
    a4, a3, a2, a1, a0 = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087, 0.1])[0]
    coefficients["CD2"]["Value"] = (a4, a3, a2, a1, a0)
    CD2_fit = a4*np.power(d_Bfine, 4) + a3*np.power(d_Bfine, 3) + a2*np.power(d_Bfine, 2) + a1*np.power(d_Bfine, 1) + a0
    axs[1, 0].plot(d_Bfine, CD2_fit, c='g', label='Coefficient Fit')
    axs[1, 0].tick_params(axis='y', labelsize= 14)

    axs[1, 1].scatter(d_B, trim_case.CD3, label='BIRE Fits')
    axs[1, 1].axhline(CD3, c='r', label='Baseline Value')
    axs[1, 1].set_title(r"$C_{D3}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD3
    A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.1, -0.66, 0.087])[0]
    coefficients["CD3"]["Value"] = (A, w, phi, z)
    CD3_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 1].plot(d_Bfine, CD3_fit, c='g', label='Coefficient Fit')
    axs[1, 1].tick_params(axis='y', labelsize= 14)

    axs[2, 0].scatter(d_B, trim_case.CD_pbar, label='BIRE Fits')
    axs[2, 0].axhline(0, c='r', label='Baseline Value')
    axs[2, 0].set_title(r"$C_{D,\overline{p}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD_pbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.02, -0.66, 0.087])[0]
    coefficients["CD_pbar"]["Value"] = (A, w, phi, z)
    CDpbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 0].plot(d_Bfine, CDpbar_fit, c='g', label='Coefficient Fit')
    axs[2, 0].tick_params(axis='y', labelsize= 14)

    axs[2, 1].scatter(d_B, trim_case.CD_qbar, label='BIRE Fits')
    axs[2, 1].axhline(CD_qbar, c='r', label='Baseline Value')
    axs[2, 1].set_title(r"$C_{D,\overline{q}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD_qbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.05, -0.66, 0.087])[0]
    coefficients["CD_qbar"]["Value"] = (A, w, phi, z)
    CDqbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 1].plot(d_Bfine, CDqbar_fit, c='g', label='Coefficient Fit')
    axs[2, 1].tick_params(axis='y', labelsize= 14)

    axs[3, 0].scatter(d_B, trim_case.CD_rbar, label='BIRE Fits')
    axs[3, 0].axhline(0, c='r', label='Baseline Value')
    axs[3, 0].set_title(r"$C_{D,\overline{r}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD_rbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.02, -0.66, 0.087])[0]
    coefficients["CD_rbar"]["Value"] = (A, w, phi, z)
    CDrbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 0].plot(d_Bfine, CDrbar_fit, c='g', label='Coefficient Fit')
    axs[3, 0].tick_params(axis='y', labelsize= 14)

    axs[3, 1].scatter(d_B, trim_case.CD_da, label='BIRE Fits')
    axs[3, 1].axhline(0, c='r', label='Baseline Value')
    axs[3, 1].set_title(r"$C_{D,\delta_a}$", fontsize=14)
    axs[3, 1].set_xticks(np.arange(-90, 120, 30))
    axs[3, 1].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD_da
    A, w, phi, z = optimize.leastsq(sine_fit, [0.06, 0.01, -0.66, 0.087])[0]
    coefficients["CD_da"]["Value"] = (A, w, phi, z)
    CDda_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 1].plot(d_Bfine, CDda_fit, c='g', label='Coefficient Fit')
    axs[3, 1].tick_params(axis='y', labelsize= 14)
    axs[3, 1].tick_params(axis='x', labelsize= 14)

    axs[4, 0].scatter(d_B, trim_case.CD_de, label='BIRE Fits')
    axs[4, 0].axhline(CD_de, c='r', label='Baseline Value')
    axs[4, 0].set_title(r"$C_{D,\delta_e}$", fontsize=14)
    axs[4, 0].set_xticks(np.arange(-90, 120, 30))
    axs[4, 0].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CD_de
    A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.02, -0.66, 0.087])[0]
    coefficients["CD_de"]["Value"] = (A, w, phi, z)
    CDde_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[4, 0].plot(d_Bfine, CDde_fit, c='g', label='Coefficient Fit')
    axs[4, 0].tick_params(axis='y', labelsize= 14)
    axs[4, 0].tick_params(axis='x', labelsize= 14)

    axs[4, 1].set_xticks([])
    axs[4, 1].set_yticks([])
    axs[4, 1].axis('off')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return coefficients

# SIDE FORCE COEFFICIENTS
def side_fits():
    coefficients = {"CS0": {"Value": None,
                            "Type": "Sine"},
                    "CS_alpha": {"Value": None,
                                 "Type": "Sine"},
                    "CS_beta": {"Value": None,
                                "Type": "Sine"},
                    "CS_pbar": {"Value": None,
                                "Type": "Sine"},
                    "CS_qbar": {"Value": None,
                                "Type": "Sine"},
                    "CS_rbar": {"Value": None,
                                "Type": "Sine"},
                    "CS_da": {"Value": None,
                              "Type": "Sine"},
                    "CS_de": {"Value": None,
                              "Type": "Sine"}}
    d_Bfine = np.linspace(-90, 90, 200)
    fig, axs = plt.subplots(4, 2, sharex='col')
    fig.suptitle('BIRE Side Force Coefficient Fits', fontsize=16)

    axs[0, 0].scatter(d_B, trim_case.CS0, label='BIRE Fits')
    axs[0, 0].axhline(0., c='r', label='Baseline Value')
    axs[0, 0].set_title(r"$C_{S0}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS0
    A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["CS0"]["Value"] = (A, w, phi, z)
    CS0_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 0].plot(d_Bfine, CS0_fit, c='g', label='Coefficient Fit')
    axs[0, 0].tick_params(axis='y', labelsize= 14)

    axs[0, 1].scatter(d_B, trim_case.CS_alpha, label='BIRE Fits')
    axs[0, 1].axhline(0., c='r', label='Baseline Value')
    axs[0, 1].set_title(r"$C_{S,\alpha}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_alpha
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CS_alpha"]["Value"] = (A, w, phi, z)
    CSalpha_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 1].plot(d_Bfine, CSalpha_fit, c='g', label='Coefficient Fit')
    axs[0, 1].tick_params(axis='y', labelsize= 14)

    axs[1, 0].scatter(d_B, trim_case.CS_beta, label='BIRE Fits')
    axs[1, 0].axhline(CS_beta, c='r', label='Baseline Value')
    axs[1, 0].set_title(r"$C_{S,\beta}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_beta
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CS_beta"]["Value"] = (A, w, phi, z)
    CSbeta_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 0].plot(d_Bfine, CSbeta_fit, c='g', label='Coefficient Fit')
    axs[1, 0].tick_params(axis='y', labelsize= 14)

    axs[1, 1].scatter(d_B, trim_case.CS_pbar, label='BIRE Fits')
    axs[1, 1].axhline(CS_pbar, c='r', label='Baseline Value')
    axs[1, 1].set_title(r"$C_{S,\overline{p}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_pbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CS_pbar"]["Value"] = (A, w, phi, z)
    CSpbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 1].plot(d_Bfine, CSpbar_fit, c='g', label='Coefficient Fit')
    axs[1, 1].tick_params(axis='y', labelsize= 14)

    axs[2, 0].scatter(d_B, trim_case.CS_qbar, label='BIRE Fits')
    axs[2, 0].axhline(0., c='r', label='Baseline Value')
    axs[2, 0].set_title(r"$C_{S,\overline{q}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_qbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.1, 1.42, 3.69])[0]
    coefficients["CS_qbar"]["Value"] = (A, w, phi, z)
    CSqbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 0].plot(d_Bfine, CSqbar_fit, c='g', label='Coefficient Fit')
    axs[2, 0].tick_params(axis='y', labelsize= 14)

    axs[2, 1].scatter(d_B, trim_case.CS_rbar, label='BIRE Fits')
    axs[2, 1].axhline(CS_rbar, c='r', label='Baseline Value')
    axs[2, 1].set_title(r"$C_{S,\overline{r}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_rbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
    coefficients["CS_rbar"]["Value"] = (A, w, phi, z)
    CSrbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 1].plot(d_Bfine, CSrbar_fit, c='g', label='Coefficient Fit')
    axs[2, 1].tick_params(axis='y', labelsize= 14)

    axs[3, 0].scatter(d_B, trim_case.CS_da, label='BIRE Fits')
    axs[3, 0].axhline(CS_da, c='r', label='Baseline Value')
    axs[3, 0].set_title(r"$C_{S,\delta_a}$", fontsize=14)
    axs[3, 0].set_xticks(np.arange(-90, 120, 30))
    axs[3, 0].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_da
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CS_da"]["Value"] = (A, w, phi, z)
    CSda_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 0].plot(d_Bfine, CSda_fit, c='g', label='Coefficient Fit')
    axs[3, 0].tick_params(axis='y', labelsize= 14)
    axs[3, 0].tick_params(axis='x', labelsize= 14)

    axs[3, 1].scatter(d_B, trim_case.CS_de, label='BIRE Fits')
    axs[3, 1].axhline(0., c='r', label='Baseline Value')
    axs[3, 1].set_title(r"$C_{S,\delta_e}$", fontsize=14)
    axs[3, 1].set_xticks(np.arange(-90, 120, 30))
    axs[3, 1].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.CS_de
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["CS_de"]["Value"] = (A, w, phi, z)
    CSde_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 1].plot(d_Bfine, CSde_fit, c='g', label='Coefficient Fit')
    axs[3, 1].tick_params(axis='y', labelsize= 14)
    axs[3, 1].tick_params(axis='x', labelsize= 14)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return coefficients

# ROLL COEFFICIENTS
def roll_fits():
    coefficients = {"Cl0": {"Value": None,
                            "Type": "Sine"},
                    "Cl_alpha": {"Value": None,
                                 "Type": "Sine"},
                    "Cl_beta": {"Value": None,
                                "Type": "Poly"},
                    "Cl_pbar": {"Value": None,
                                "Type": "Sine"},
                    "Cl_qbar": {"Value": None,
                                "Type": "Sine"},
                    "Cl_rbar": {"Value": None,
                                "Type": "Sine"},
                    "Cl_da": {"Value": None,
                              "Type": "Sine"},
                    "Cl_de": {"Value": None,
                              "Type": "Poly"}}
    d_Bfine = np.linspace(-90, 90, 200)
    fig, axs = plt.subplots(4, 2, sharex='col')
    fig.suptitle('BIRE Rolling Moment Coefficient Fits', fontsize=16)

    axs[0, 0].scatter(d_B, trim_case.Cell0, label='BIRE Fits')
    axs[0, 0].axhline(0., c='r', label='Baseline Value')
    axs[0, 0].set_title(r"$C_{\ell 0}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cell0
    A, w, phi, z = optimize.leastsq(sine_fit, [0.1, 0.1, -0.66, 0.087])[0]
    coefficients["Cl0"]["Value"] = (A, w, phi, z)
    Cell0_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 0].plot(d_Bfine, Cell0_fit, c='g', label='Coefficient Fit')
    axs[0, 0].tick_params(axis='y', labelsize= 14)

    axs[0, 1].scatter(d_B, trim_case.Cell_alpha, label='BIRE Fits')
    axs[0, 1].axhline(0., c='r', label='Baseline Value')
    axs[0, 1].set_title(r"$C_{\ell,\alpha}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cell_alpha
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.1, 1.42, 3.69])[0]
    coefficients["Cl_alpha"]["Value"] = (A, w, phi, z)
    Cellalpha_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 1].plot(d_Bfine, Cellalpha_fit, c='g', label='Coefficient Fit')
    axs[0, 1].tick_params(axis='y', labelsize= 14)

    axs[1, 0].scatter(d_B, trim_case.Cell_beta, label='BIRE Fits')
    axs[1, 0].axhline(Cell_beta, c='r', label='Baseline Value')
    axs[1, 0].set_title(r"$C_{\ell,\beta}$", fontsize=14)
    Cellbeta_fit = np.full(200, np.mean(trim_case.Cell_beta))
    coefficients["Cl_beta"]["Value"] = (np.mean(trim_case.Cell_beta))
    axs[1, 0].plot(d_Bfine, Cellbeta_fit, c='g', label='Coefficient Fit')
    axs[1, 0].tick_params(axis='y', labelsize= 14)

    axs[1, 1].scatter(d_B, trim_case.Cell_pbar, label='BIRE Fits')
    axs[1, 1].axhline(Cell_pbar, c='r', label='Baseline Value')
    axs[1, 1].set_title(r"$C_{\ell,\overline{p}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cell_pbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["Cl_pbar"]["Value"] = (A, w, phi, z)
    Cellpbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 1].plot(d_Bfine, Cellpbar_fit, c='g', label='Coefficient Fit')
    axs[1, 1].tick_params(axis='y', labelsize= 14)

    axs[2, 0].scatter(d_B, trim_case.Cell_qbar, label='BIRE Fits')
    axs[2, 0].axhline(0., c='r', label='Baseline Value')
    axs[2, 0].set_title(r"$C_{\ell,\overline{q}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cell_qbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.05, 0.1, 1.42, 3.69])[0]
    coefficients["Cl_qbar"]["Value"] = (A, w, phi, z)
    Cellqbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 0].plot(d_Bfine, Cellqbar_fit, c='g', label='Coefficient Fit')
    axs[2, 0].tick_params(axis='y', labelsize= 14)

    axs[2, 1].scatter(d_B, trim_case.Cell_rbar, label='BIRE Fits')
    axs[2, 1].axhline(Cell_rbar, c='r', label='Baseline Value')
    axs[2, 1].set_title(r"$C_{\ell,\overline{r}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cell_rbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
    coefficients["Cl_rbar"]["Value"] = (A, w, phi, z)
    Cellrbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 1].plot(d_Bfine, Cellrbar_fit, c='g', label='Coefficient Fit')
    axs[2, 1].tick_params(axis='y', labelsize= 14)

    axs[3, 0].scatter(d_B, trim_case.Cell_da, label='BIRE Fits')
    axs[3, 0].axhline(Cell_da, c='r', label='Baseline Value')
    axs[3, 0].set_title(r"$C_{\ell,\delta_a}$", fontsize=14)
    axs[3, 0].set_xticks(np.arange(-90, 120, 30))
    axs[3, 0].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cell_da
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["Cl_da"]["Value"] = (A, w, phi, z)
    Cellda_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 0].plot(d_Bfine, Cellda_fit, c='g', label='Coefficient Fit')
    axs[3, 0].tick_params(axis='y', labelsize= 14)
    axs[3, 0].tick_params(axis='x', labelsize= 14)

    axs[3, 1].scatter(d_B, trim_case.Cell_de, label='BIRE Fits')
    axs[3, 1].axhline(0., c='r', label='Baseline Value')
    axs[3, 1].set_title(r"$C_{\ell,\delta_e}$", fontsize=14)
    axs[3, 1].set_xticks(np.arange(-90, 120, 30))
    axs[3, 1].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) + x[2]*np.power(d_B, 1) + x[3] - trim_case.Cell_de
    a3, a2, a1, a0 = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["Cl_de"]["Value"] = (a3, a2, a1, a0)
    Cellde_fit = a3*np.power(d_Bfine, 3) + a2*np.power(d_Bfine, 2) + a1*np.power(d_Bfine, 1) + a0
    axs[3, 1].plot(d_Bfine, Cellde_fit, c='g', label='Coefficient Fit')
    axs[3, 1].tick_params(axis='y', labelsize= 14)
    axs[3, 1].tick_params(axis='x', labelsize= 14)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return coefficients

# PITCH COEFFICIENTS
def pitch_fits():
    coefficients = {"Cm0": {"Value": None,
                            "Type": "Sine"},
                    "Cm_alpha": {"Value": None,
                                 "Type": "Sine"},
                    "Cm_beta": {"Value": None,
                                "Type": "Sine"},
                    "Cm_pbar": {"Value": None,
                                "Type": "Sine"},
                    "Cm_qbar": {"Value": None,
                                "Type": "Sine"},
                    "Cm_rbar": {"Value": None,
                                "Type": "Sine"},
                    "Cm_da": {"Value": None,
                              "Type": "Sine"},
                    "Cm_de": {"Value": None,
                              "Type": "Sine"}}
    d_Bfine = np.linspace(-90, 90, 200)
    fig, axs = plt.subplots(4, 2, sharex='col')
    fig.suptitle('BIRE Pitching Moment Coefficient Fits', fontsize=16)

    axs[0, 0].scatter(d_B, trim_case.Cm0, label='BIRE Fits')
    axs[0, 0].axhline(Cm0, c='r', label='Baseline Value')
    axs[0, 0].set_title(r"$C_{m 0}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm0
    A, w, phi, z = optimize.leastsq(sine_fit, [0.3, 0.05, -0.66, 0.087])[0]
    coefficients["Cm0"]["Value"] = (A, w, phi, z)
    Cm0_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 0].plot(d_Bfine, Cm0_fit, c='g', label='Coefficient Fit')
    axs[0, 0].tick_params(axis='y', labelsize= 14)

    axs[0, 1].scatter(d_B, trim_case.Cm_alpha, label='BIRE Fits')
    axs[0, 1].axhline(Cm_alpha, c='r', label='Baseline Value')
    axs[0, 1].set_title(r"$C_{m,\alpha}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_alpha
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.05, 1.42, 3.69])[0]
    coefficients["Cm_alpha"]["Value"] = (A, w, phi, z)
    Cmalpha_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 1].plot(d_Bfine, Cmalpha_fit, c='g', label='Coefficient Fit')
    axs[0, 1].tick_params(axis='y', labelsize= 14)

    axs[1, 0].scatter(d_B, trim_case.Cm_beta, label='BIRE Fits')
    axs[1, 0].axhline(0., c='r', label='Baseline Value')
    axs[1, 0].set_title(r"$C_{m,\beta}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_beta
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.05, 1.42, 3.69])[0]
    coefficients["Cm_beta"]["Value"] = (A, w, phi, z)
    Cmbeta_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 0].plot(d_Bfine, Cmbeta_fit, c='g', label='Coefficient Fit')
    axs[1, 0].tick_params(axis='y', labelsize= 14)

    axs[1, 1].scatter(d_B, trim_case.Cm_pbar, label='BIRE Fits')
    axs[1, 1].axhline(0., c='r', label='Baseline Value')
    axs[1, 1].set_title(r"$C_{m,\overline{p}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_pbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["Cm_pbar"]["Value"] = (A, w, phi, z)
    Cmpbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 1].plot(d_Bfine, Cmpbar_fit, c='g', label='Coefficient Fit')
    axs[1, 1].tick_params(axis='y', labelsize= 14)

    axs[2, 0].scatter(d_B, trim_case.Cm_qbar, label='BIRE Fits')
    axs[2, 0].axhline(Cm_qbar, c='r', label='Baseline Value')
    axs[2, 0].set_title(r"$C_{m,\overline{q}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_qbar
    A, w, phi, z = optimize.leastsq(sine_fit, [-3., 0.04, 1.42, 3.69])[0]
    coefficients["Cm_qbar"]["Value"] = (A, w, phi, z)
    Cmqbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 0].plot(d_Bfine, Cmqbar_fit, c='g', label='Coefficient Fit')
    axs[2, 0].tick_params(axis='y', labelsize= 14)

    axs[2, 1].scatter(d_B, trim_case.Cm_rbar, label='BIRE Fits')
    axs[2, 1].axhline(0., c='r', label='Baseline Value')
    axs[2, 1].set_title(r"$C_{m,\overline{r}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_rbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
    coefficients["Cm_rbar"]["Value"] = (A, w, phi, z)
    Cmrbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 1].plot(d_Bfine, Cmrbar_fit, c='g', label='Coefficient Fit')
    axs[2, 1].tick_params(axis='y', labelsize= 14)

    axs[3, 0].scatter(d_B, trim_case.Cm_da, label='BIRE Fits')
    axs[3, 0].axhline(0., c='r', label='Baseline Value')
    axs[3, 0].set_title(r"$C_{m,\delta_a}$", fontsize=14)
    axs[3, 0].set_xticks(np.arange(-90, 120, 30))
    axs[3, 0].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_da
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["Cm_da"]["Value"] = (A, w, phi, z)
    Cmda_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 0].plot(d_Bfine, Cmda_fit, c='g', label='Coefficient Fit')
    axs[3, 0].tick_params(axis='y', labelsize= 14)
    axs[3, 0].tick_params(axis='x', labelsize= 14)

    axs[3, 1].scatter(d_B, trim_case.Cm_de, label='BIRE Fits')
    axs[3, 1].axhline(Cm_de, c='r', label='Baseline Value')
    axs[3, 1].set_title(r"$C_{m,\delta_e}$", fontsize=14)
    axs[3, 1].set_xticks(np.arange(-90, 120, 30))
    axs[3, 1].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cm_de
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["Cm_de"]["Value"] = (A, w, phi, z)
    Cmde_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 1].plot(d_Bfine, Cmde_fit, c='g', label='Coefficient Fit')
    axs[3, 1].tick_params(axis='y', labelsize= 14)
    axs[3, 1].tick_params(axis='x', labelsize= 14)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return coefficients

# YAW COEFFICIENTS
def yaw_fits():
    coefficients = {"Cn0": {"Value": None,
                            "Type": "Sine"},
                    "Cn_alpha": {"Value": None,
                                 "Type": "Sine"},
                    "Cn_beta": {"Value": None,
                                "Type": "Sine"},
                    "Cn_pbar": {"Value": None,
                                "Type": "Poly"},
                    "Cn_qbar": {"Value": None,
                                "Type": "Sine"},
                    "Cn_rbar": {"Value": None,
                                "Type": "Sine"},
                    "Cn_da": {"Value": None,
                              "Type": "Sine"},
                    "Cn_de": {"Value": None,
                              "Type": "Poly"}}
    d_Bfine = np.linspace(-90, 90, 200)
    fig, axs = plt.subplots(4, 2, sharex='col')
    fig.suptitle('BIRE Yawing Moment Coefficient Fits', fontsize=16)

    axs[0, 0].scatter(d_B, trim_case.Cn0, label='BIRE Fits')
    axs[0, 0].axhline(0., c='r', label='Baseline Value')
    axs[0, 0].set_title(r"$C_{n 0}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cn0
    A, w, phi, z = optimize.leastsq(sine_fit, [0.1, 0.1, -0.66, 0.087])[0]
    coefficients["Cn0"]["Value"] = (A, w, phi, z)
    Cn0_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 0].plot(d_Bfine, Cn0_fit, c='g', label='Coefficient Fit')
    axs[0, 0].tick_params(axis='y', labelsize= 14)

    axs[0, 1].scatter(d_B, trim_case.Cn_alpha, label='BIRE Fits')
    axs[0, 1].axhline(0., c='r', label='Baseline Value')
    axs[0, 1].set_title(r"$C_{n,\alpha}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cn_alpha
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.03, 1.42, 3.69])[0]
    coefficients["Cn_alpha"]["Value"] = (A, w, phi, z)
    Cnalpha_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[0, 1].plot(d_Bfine, Cnalpha_fit, c='g', label='Coefficient Fit')
    axs[0, 1].tick_params(axis='y', labelsize= 14)

    axs[1, 0].scatter(d_B, trim_case.Cn_beta, label='BIRE Fits')
    axs[1, 0].axhline(Cn_beta, c='r', label='Baseline Value')
    axs[1, 0].set_title(r"$C_{n,\beta}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cn_beta
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.03, 1.42, 3.69])[0]
    coefficients["Cn_beta"]["Value"] = (A, w, phi, z)
    Cnbeta_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[1, 0].plot(d_Bfine, Cnbeta_fit, c='g', label='Coefficient Fit')
    axs[1, 0].tick_params(axis='y', labelsize= 14)

    axs[1, 1].scatter(d_B, trim_case.Cn_pbar, label='BIRE Fits')
    axs[1, 1].axhline(Cn_pbar, c='r', label='Baseline Value')
    axs[1, 1].set_title(r"$C_{n,\overline{p}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) + x[2]*np.power(d_B, 1) + x[3] - trim_case.Cn_pbar
    a3, a2, a1, a0 = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["Cn_pbar"]["Value"] = (a3, a2, a1, a0)
    Cnpbar_fit = a3*np.power(d_Bfine, 3) + a2*np.power(d_Bfine, 2) + a1*np.power(d_Bfine, 1) + a0
    axs[1, 1].plot(d_Bfine, Cnpbar_fit, c='g', label='Coefficient Fit')
    axs[1, 1].tick_params(axis='y', labelsize= 14)

    axs[2, 0].scatter(d_B, trim_case.Cn_qbar, label='BIRE Fits')
    axs[2, 0].axhline(0., c='r', label='Baseline Value')
    axs[2, 0].set_title(r"$C_{n,\overline{q}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cn_qbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.02, 1.42, 3.69])[0]
    coefficients["Cn_qbar"]["Value"] = (A, w, phi, z)
    Cnqbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 0].plot(d_Bfine, Cnqbar_fit, c='g', label='Coefficient Fit')
    axs[2, 0].tick_params(axis='y', labelsize= 14)

    axs[2, 1].scatter(d_B, trim_case.Cn_rbar, label='BIRE Fits')
    axs[2, 1].axhline(Cn_rbar, c='r', label='Baseline Value')
    axs[2, 1].set_title(r"$C_{n,\overline{r}}$", fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cn_rbar
    A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
    coefficients["Cn_rbar"]["Value"] = (A, w, phi, z)
    Cnrbar_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[2, 1].plot(d_Bfine, Cnrbar_fit, c='g', label='Coefficient Fit')
    axs[2, 1].tick_params(axis='y', labelsize= 14)

    axs[3, 0].scatter(d_B, trim_case.Cn_da, label='BIRE Fits')
    axs[3, 0].axhline(Cn_da, c='r', label='Baseline Value')
    axs[3, 0].set_title(r"$C_{n,\delta_a}$", fontsize=14)
    axs[3, 0].set_xticks(np.arange(-90, 120, 30))
    axs[3, 0].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - trim_case.Cn_da
    A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
    coefficients["Cn_da"]["Value"] = (A, w, phi, z)
    Cnda_fit = A*np.sin(w*d_Bfine + phi) + z
    axs[3, 0].plot(d_Bfine, Cnda_fit, c='g', label='Coefficient Fit')
    axs[3, 0].tick_params(axis='y', labelsize= 14)
    axs[3, 0].tick_params(axis='x', labelsize= 14)

    axs[3, 1].scatter(d_B, trim_case.Cn_de, label='BIRE Fits')
    axs[3, 1].axhline(0., c='r', label='Baseline Value')
    axs[3, 1].set_title(r"$C_{n,\delta_e}$", fontsize=14)
    axs[3, 1].set_xticks(np.arange(-90, 120, 30))
    axs[3, 1].set_xlabel(r'$\delta_B$ [deg]', fontsize=14)
    sine_fit = lambda x: x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) + x[2]*np.power(d_B, 1) + x[3] - trim_case.Cn_de
    a3, a2, a1, a0 = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
    coefficients["Cn_de"]["Value"] = (a3, a2, a1, a0)
    Cnde_fit = a3*np.power(d_Bfine, 3) + a2*np.power(d_Bfine, 2) + a1*np.power(d_Bfine, 1) + a0
    axs[3, 1].plot(d_Bfine, Cnde_fit, c='g', label='Coefficient Fit')
    axs[3, 1].tick_params(axis='y', labelsize= 14)
    axs[3, 1].tick_params(axis='x', labelsize= 14)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    return coefficients

plt.close('all')
lift_coeffs = lift_fits()
drag_coeffs = drag_fits()
side_coeffs = side_fits()
roll_coeffs = roll_fits()
pitch_coeffs = pitch_fits()
yaw_coeffs = yaw_fits()


