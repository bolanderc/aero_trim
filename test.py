#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aero_trim unit testing
"""

import pytest
import aero_trim
import numpy as np
import os.path


def test_initialization():
    """Tests that data is properly initialized in a TrimCase object."""
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free, units="Metric")
    coeff_test = np.allclose(trim_case.nondim_coeff, 17145.2854874284,
                             atol=1e-12)
    metric_test = np.allclose(trim_case.g, 9.81, atol=1e-12)
    assert coeff_test*metric_test

def test_dimension_inputs():
    """Tests that the proper exception is thrown if the `database_params`
    keyword is specified.
    """
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    num_pts_per_dimension = [5, 4, 5, 5, 5, 5, 5, 8]
    with pytest.raises(Exception):
        trim_case.import_aero_data(TEST_DATA_FILE_NAME, NUM_DIMENSIONS,
                                   num_pts_per_dimension, DIMENSION_LIMS)

def test_database_arguments():
    """Tests that an exception is thrown if the proper arguments are not
    specified when using a database for trimming.
    """
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    with pytest.raises(Exception):
        trim_case.import_aero_data(TEST_DATA_FILE_NAME)

def test_data_in():
    """Tests that .csv data is properly ordered by the `_import_csv` method."""
    coeff_header_titles = ["Cx", "Cy", "Cz", "Cl", "Cm", "Cn"]
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    data_in, data_array = trim_case._import_csv(TEST_DATA_FILE_NAME,
                                                NUM_DIMENSIONS,
                                                NUM_PTS_PER_DIMENSION,
                                                SORT_HEADER_TITLES)
    data_in_array = data_in[coeff_header_titles].to_numpy()
    assert np.allclose(data_in_array[0, 0], 0.189408635274333, atol=10e-12)

def test_data_array():
    """Tests that .csv data is properly resized by the `_import_csv` method."""
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    data_in, data_array = trim_case._import_csv(TEST_DATA_FILE_NAME,
                                                NUM_DIMENSIONS,
                                                NUM_PTS_PER_DIMENSION,
                                                SORT_HEADER_TITLES)
    assert np.allclose(data_array[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       0.189408635274333, atol=10e-12)

def test_save_aero_data():
    """Tests that .csv data is properly assigned to the aerodynamic
    force and moment functions by the `_import_csv` method.
    """
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    data_in, data_array = trim_case._import_csv(TEST_DATA_FILE_NAME,
                                                NUM_DIMENSIONS,
                                                NUM_PTS_PER_DIMENSION,
                                                SORT_HEADER_TITLES)
    database_params = np.zeros((NUM_DIMENSIONS, NUM_PTS_PER_DIMENSION))
    for i in range(NUM_DIMENSIONS):
        database_params[i, :] = np.linspace(DIMENSION_LIMS[i][0],
                                            DIMENSION_LIMS[i][1],
                                            NUM_PTS_PER_DIMENSION)
    database_params = tuple(database_params)
    trim_case._save_aero_data(data_array, database_params)
    test_case = [x[0] for x in DIMENSION_LIMS]
    force_test = np.allclose(trim_case.c_x(test_case), 0.189408635274333,
                             atol=10e-12)
    moment_test = np.allclose(trim_case.c_ell(test_case), 0.050172906838691,
                              atol=10e-12)
    assert force_test*moment_test

def test_b2w_conversions():
    """Tests that force conversions from the body-fixed frame to the wind frame
    are calculated properly.
    """
    alpha = 5.  # deg.
    beta = 10.  # deg.
    c_a = np.cos(np.deg2rad(alpha))
    s_a = np.sin(np.deg2rad(alpha))
    c_b = np.cos(np.deg2rad(beta))
    s_b = np.sin(np.deg2rad(beta))
    body2stab = np.array([[c_a, 0., s_a],
                          [0., 1., 0.],
                          [-s_a, 0., c_a]])
    stab2wind = np.array([[c_b, s_b, 0.],
                          [-s_b, c_b, 0.],
                          [0., 0., 1.]])
    body2wind = np.matmul(stab2wind, body2stab)
    orthogonal_assert = np.linalg.norm(np.matmul(body2wind.T, body2wind) -
                                       np.eye(3)) < 1e-12

    body_forces = np.array([100., 100., 100.])
    Fxb, Fyb, Fzb = body_forces
    L = aero_trim.bf2w_lift(Fxb, Fyb, Fzb, np.deg2rad(alpha), np.deg2rad(beta))
    D = aero_trim.bf2w_drag(Fxb, Fyb, Fzb, np.deg2rad(alpha), np.deg2rad(beta))
    S = aero_trim.bf2w_side(Fxb, Fyb, Fzb, np.deg2rad(alpha), np.deg2rad(beta))
    wind_forces = np.matmul(body2wind, body_forces)
    wind_forces[0] *= -1.
    wind_forces[2] *= -1.
    force_assert = np.linalg.norm(wind_forces - np.array([D, S, L])) < 1e-12
    assert orthogonal_assert*force_assert

def test_database_conversion():
    """Tests that the database conversion for the linear fits is performed
    properly.
    """
    database_file_name = "./misc/TODatabase_body_3all.csv"
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    aero_data = np.loadtxt(database_file_name, delimiter=",", skiprows=1)
    aero_data_check = np.copy(aero_data)
    converted_aero_data = trim_case._data_conversion(aero_data)
    N = len(aero_data[:, 0])
    pbar = np.zeros(N)
    qbar = np.zeros(N)
    rbar = np.zeros(N)
    CD = np.zeros(N)
    CS = np.zeros(N)
    CL = np.zeros(N)
    alpha_rad = np.deg2rad(aero_data_check[:, 0])
    beta_rad = np.deg2rad(aero_data_check[:, 1])
    for i in range(N):
        pbar[i] = aero_data_check[i, 5]*trim_case.lat_ref_len/(2.*trim_case.V)
        qbar[i] = aero_data_check[i, 6]*trim_case.lon_ref_len/(2.*trim_case.V)
        rbar[i] = aero_data_check[i, 7]*trim_case.lat_ref_len/(2.*trim_case.V)
        CD[i] = aero_trim.bf2w_drag(aero_data_check[i, 8], aero_data_check[i, 9],
                                    aero_data_check[i, 10], alpha_rad[i],
                                    beta_rad[i])
        CS[i] = aero_trim.bf2w_side(aero_data_check[i, 8], aero_data_check[i, 9],
                                    aero_data_check[i, 10], alpha_rad[i],
                                    beta_rad[i])
        CL[i] = aero_trim.bf2w_lift(aero_data_check[i, 8], aero_data_check[i, 9],
                                    aero_data_check[i, 10], alpha_rad[i],
                                    beta_rad[i])
    aero_data_check[:, 5] = pbar
    aero_data_check[:, 6] = qbar
    aero_data_check[:, 7] = rbar
    aero_data_check[:, -6] = CD
    aero_data_check[:, -5] = CS
    aero_data_check[:, -4] = CL
    diff_conv = np.abs(converted_aero_data - aero_data_check)
    conversion_check = np.linalg.norm(diff_conv) < 1e-12
    assert conversion_check

def test_linear_model():
    """Tests that the coefficients of the linear model are calculated
    correctly.
    """
    database_file_name = "./misc/TODatabase_body_3all.csv"
    v_free, rho_free = 222.5211, 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    trim_case.import_aero_data(database_file_name, model="linear")
    lift_coeffs_obs = [8.757199079141e-02, 4.031411055571e+00,
                       -2.175295766053e-12, -5.769614431381e-14,
                       3.735093201199e+00, 8.564535730596e-12,
                       2.684688931344e-07, 6.340879225759e-01,
                       -8.310767411173e-08]
    drag_coeffs_obs = [6.872528951111e-02, 2.857818487492e-03,
                       1.051037229970e-01, 2.148078531185e-01,
                       -3.666540252514e-14, 3.671443148435e-02,
                       -4.772459354103e-12, -5.018999387633e-07,
                       -7.673321083839e-03, -1.407066136080e-07]
    lift_assert = [abs(fit - obs) for fit,obs in zip(trim_case.coeffs_lift,
                                                     lift_coeffs_obs)]
    lift_assert = np.linalg.norm(lift_assert) < 1e-12
    drag_assert = [abs(fit - obs) for fit,obs in zip(trim_case.coeffs_drag,
                                                     drag_coeffs_obs)]
    drag_assert = np.linalg.norm(drag_assert) < 1e-12
    assert lift_assert*drag_assert


def test_vel_comp():
    """Tests that the velocity components are calculated properly."""
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    alpha = 5.  # deg.
    beta = 10.  # deg.
    v_bf = trim_case._vel_comp(alpha, beta)
    v_bf_analytical = [98.1172637986292, 17.300720880988255, 8.58414827856259]
    assert np.allclose(v_bf, v_bf_analytical, atol=1e-13)

def test_sct_rot_rates():
    """Tests that the rotation rates are calculated properly for the steady-
    coordinated turn.
    """
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    elevation_angle = 10.  # deg.
    bank_angle = 20.  # deg.
    s_elev = np.sin(np.deg2rad(elevation_angle))
    c_elev = np.cos(np.deg2rad(elevation_angle))
    s_bank = np.sin(np.deg2rad(bank_angle))
    c_bank = np.cos(np.deg2rad(bank_angle))
    sc_angles = [s_elev, c_elev, s_bank, c_bank]
    orientation_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
    alpha = 5  # deg.
    beta = 10.  # deg.
    [u, v, w] = trim_case._vel_comp(alpha, beta)
    rot_rates = trim_case._sct_rot_rates(sc_angles, orientation_transform,
                                         u, w)
    rot_rates_analytical = [-0.020406793394732076, 0.039582906561664424,
                            0.10875314197455521]
    assert np.allclose(rot_rates, rot_rates_analytical, atol=1e-13)

def test_dimensionalize_aero_fm():
    """Tests that the aerodynamic coefficients are dimensionalized properly."""
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    trim_case.import_aero_data(TEST_DATA_FILE_NAME, NUM_DIMENSIONS,
                               NUM_PTS_PER_DIMENSION, DIMENSION_LIMS)
    alpha = 10.
    beta = 0.
    d_e = -20.
    d_a = 0.
    d_r = 0.
    p = 0.
    q = 0.
    r = 0.
    f_xt = 100.
    params = [alpha, beta, d_e, d_a, d_r, p, q, r]
    forces, moments = trim_case._dimensionalize_aero_fm(params, f_xt)
    forces_test = [707.15108331, -8.5465337, -1656.14365612]
    moments_test = [-33.67271836, 54120.6095819, 110.17488516]
    force_diff = [abs(x - y) for x, y in zip(forces, forces_test)]
    force_assert = np.allclose(force_diff, [0., 0., 0], atol=1e-8)
    moment_diff = [abs(x - y) for x, y in zip(moments, moments_test)]
    moment_assert = np.allclose(moment_diff, [0., 0., 0], atol=1e-8)
    assert force_assert*moment_assert

def test_weight_forces():
    """Tests that the 6DOF equation outputs the correct forces due to
    orientation.
    """
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    elevation_angle = 5.
    bank_angle = 5.
    s_elev = np.sin(np.deg2rad(elevation_angle))
    c_elev = np.cos(np.deg2rad(elevation_angle))
    s_bank = np.sin(np.deg2rad(bank_angle))
    c_bank = np.cos(np.deg2rad(bank_angle))
    euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
    f_weight = trim_case._6dof_weight_forces(euler_transform)
    f_weight_test = [-1786.6927263269924, 1779.893821086036,
                     20344.279468375134]
    assert np.allclose(f_weight, f_weight_test, atol=1e-12)

def test_coriolis_forces():
    """Tests that the 6DOF equation outputs the correct forces Coriolis
    effects.
    """
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    elevation_angle = 5.
    bank_angle = 5.
    alpha = 10.
    beta = 0.
    s_elev = np.sin(np.deg2rad(elevation_angle))
    c_elev = np.cos(np.deg2rad(elevation_angle))
    s_bank = np.sin(np.deg2rad(bank_angle))
    c_bank = np.cos(np.deg2rad(bank_angle))
    euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
    sc_angles = [s_elev, c_elev, s_bank, c_bank]
    vel_bf = trim_case._vel_comp(alpha, beta)
    u, v, w = vel_bf
    rot_rates = trim_case._sct_rot_rates(sc_angles, euler_transform, u, w)
    coriolis_forces = trim_case._6dof_coriolis_forces(rot_rates, vel_bf)
    coriolis_forces_test = [-27.0390174160123, -1779.89382108604,
                            153.345887891757]
    assert np.allclose(coriolis_forces, coriolis_forces_test, atol=1e-12)

def test_rotor_moments():
    """Tests that the 6DOF equation outputs the correct moments due to
    spinning rotors.
    """
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    elevation_angle = 5.
    bank_angle = 5.
    alpha = 10.
    beta = 0.
    s_elev = np.sin(np.deg2rad(elevation_angle))
    c_elev = np.cos(np.deg2rad(elevation_angle))
    s_bank = np.sin(np.deg2rad(bank_angle))
    c_bank = np.cos(np.deg2rad(bank_angle))
    euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
    sc_angles = [s_elev, c_elev, s_bank, c_bank]
    vel_bf = trim_case._vel_comp(alpha, beta)
    u, v, w = vel_bf
    rot_rates = trim_case._sct_rot_rates(sc_angles, euler_transform, u, w)
    rotor_moments = trim_case._6dof_rotor_moments(rot_rates)
    rotor_moments_test = [0.0, -4.47291748473101, 0.391329572800854]
    assert np.allclose(rotor_moments, rotor_moments_test, atol=1e-12)

def test_coriolis_moments():
    """Tests that the 6DOF equation outputs the correct moments due to
    Coriolis effects.
    """
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    elevation_angle = 5.
    bank_angle = 5.
    alpha = 10.
    beta = 0.
    s_elev = np.sin(np.deg2rad(elevation_angle))
    c_elev = np.cos(np.deg2rad(elevation_angle))
    s_bank = np.sin(np.deg2rad(bank_angle))
    c_bank = np.cos(np.deg2rad(bank_angle))
    euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
    sc_angles = [s_elev, c_elev, s_bank, c_bank]
    vel_bf = trim_case._vel_comp(alpha, beta)
    u, v, w = vel_bf
    rot_rates = trim_case._sct_rot_rates(sc_angles, euler_transform, u, w)
    coriolis_moments = trim_case._6dof_coriolis_moments(rot_rates)
    coriolis_moments_test = [-0.504072698009135, -2.91760571417836,
                             0.210988320654507]
    assert np.allclose(coriolis_moments, coriolis_moments_test, atol=1e-12)

def test_6dof_fm():
    """Tests that the 6DOF equation outputs the correct forces and moments. """
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    trim_case.import_aero_data(TEST_DATA_FILE_NAME, NUM_DIMENSIONS,
                                NUM_PTS_PER_DIMENSION, DIMENSION_LIMS)
    alpha = 10.
    beta = 0.
    d_e = 0.
    d_a = 0.
    d_r = 10.
    f_xt = 100.
    params = [alpha, beta, d_e, d_a, d_r]
    theta = 5.
    phi = 5.
    orientation = [theta, phi]
    vel_bf = trim_case._vel_comp(alpha, beta)
    forces, moments = trim_case._6dof_fm(params, f_xt, orientation, vel_bf)
    p, q, r = [-0.002455152426217877,
                0.0024458098300053346,
                0.027955734279568805]
    params = list(params[:] + [p, q, r])
    f_aero_test, m_aero_test = trim_case._dimensionalize_aero_fm(params, f_xt)
    f_weight_test = [-1786.6927263269924, 1779.893821086036,
                     20344.279468375134]
    f_coriolis_test = [-27.0390174160123, -1779.89382108604,
                            153.345887891757]
    f_total_test = [x + y + z for x, y, z in zip(f_aero_test,
                                                 f_weight_test,
                                                 f_coriolis_test)]
    force_assert = np.allclose(forces, f_total_test, atol=1e-12)
    m_rotor_test = [0.0, -4.47291748473101, 0.391329572800854]
    m_coriolis_test = [-0.504072698009135, -2.91760571417836,
                       0.210988320654507]
    m_total_test = [x + y + z for x, y, z in zip(m_aero_test,
                                                 m_rotor_test,
                                                 m_coriolis_test)]
    moment_assert = np.allclose(moments, m_total_test, atol=1e-12)
    assert force_assert*moment_assert

def test_calc_elevation():
    """Tests that the elevation angle is properly calculated."""
    trim_case = aero_trim.TrimCase(V_FREE, RHO_FREE)
    climb = 0.
    bank = 0.
    alpha = 0.
    beta = 0.
    elevation = trim_case._calc_elevation_angle(alpha, beta,
                                                [climb, bank], V_FREE)
    elevation_check1 = np.allclose(elevation, 0., atol=1e-12)

    climb = 20.
    alpha = 20.
    elevation = trim_case._calc_elevation_angle(alpha, beta,
                                                [climb, bank], V_FREE)
    elevation_check2 = np.allclose(elevation, alpha + climb, atol=1e-12)

    climb = -5.
    alpha = 5.
    elevation = trim_case._calc_elevation_angle(alpha, beta,
                                                [climb, bank], V_FREE)
    elevation_check3 = np.allclose(elevation, alpha + climb, atol=1e-12)

    climb = 10.
    alpha = 10.
    bank = 90.
    beta = 60.
    elevation = trim_case._calc_elevation_angle(alpha, beta,
                                                [climb, bank], V_FREE)
    assert elevation_check1*elevation_check2*elevation_check3


V_FREE, RHO_FREE = 100., 0.0023084
TEST_DATA_FILE_NAME = os.path.join("misc", "test_database.csv")
DIMENSION_LIMS = [(-15., 15.),
                  (-15., 15.),
                  (-21.5/4., 21.5/4.),
                  (-25., 25.),
                  (-30., 30.),
                  (-1.2, 1.2),
                  (-1.2, 1.2),
                  (-0.3925, 0.3925)]
NUM_DIMENSIONS = 8
NUM_PTS_PER_DIMENSION = 5
SORT_HEADER_TITLES = ["AOA", "Beta", "d_e", "d_a", "d_r", "p", "q", "r"]

