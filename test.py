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

def test_vel_comp():
    """Tests that the velocity components are calculated properly."""
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
    alpha = 5.  # deg.
    beta = 10.  # deg.
    v_bf = trim_case._vel_comp(alpha, beta)
    v_bf_analytical = [98.1172637986292, 17.300720880988255, 8.58414827856259]
    assert np.allclose(v_bf, v_bf_analytical, atol=1e-13)

def test_sct_rot_rates():
    """Tests that the rotation rates are calculated properly for the steady-
    coordinated turn.
    """
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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
    v_free, rho_free = 100., 0.0023084
    trim_case = aero_trim.TrimCase(v_free, rho_free)
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

