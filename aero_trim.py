"""This module is an implementation of a general aerodynamic trim algorithm.

This module allows users to supply an aerodynamic database along with relevant
atmospheric and reference data to trim an aircraft in flight. The trim
algorithm implemented is a general algorithm allowing for the known parameters
to be specified by the user.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import RegularGridInterpolator as rgi
import scipy.optimize as optimize


class AeroTrim:
    """Trims an aircraft given an aerodynamic database.

    This class creates an aerodynamic database from user-provided data which
    can then be used to trim an aircraft. The trim algorithm implemented is
    general in nature, meaning that the user specifies which state information
    they know about the aircraft and which will determined from the trim
    algorithm.

    Attributes
    ----------
    c_x : function
        A linearly-interpolated database containing all of the body-fixed
        aerodynamic force coefficients in the x-direction
        (out the nose of the aircraft) as a function of the parameters
        describing the database.

    c_y : function
        A linearly-interpolated database containing all of the body-fixed
        aerodynamic force coefficients in the y-direction
        (out the right wing of the aircraft) as a function of the parameters
        describing the database.

    c_z : function
        A linearly-interpolated database containing all of the body-fixed
        aerodynamic force coefficients in the z-direction
        (out the belly of the aircraft) as a function of the parameters
        describing the database.

    c_ell : function
        A linearly-interpolated database containing all of the body-fixed
        aerodynamic moment coefficients about the x-axis
        (out the nose of the aircraft) as a function of the parameters
        describing the database.

    c_m : function
        A linearly-interpolated database containing all of the body-fixed
        aerodynamic moment coefficients about the y-axis
        (out the right wing of the aircraft) as a function of the parameters
        describing the database.

    c_n : function
        A linearly-interpolated database containing all of the body-fixed
        aerodynamic moment coefficients about the z-axis
        (out the belly of the aircraft) as a function of the parameters
        describing the database.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.

    Yields
    ------
    int
        Description of the anonymous integer return value.

    Receives
    --------

    Ohter Parameters
    ----------------

    Raises
    ------

    Warns
    -----

    Warnings
    --------

    See Also
    --------
    func_a : Function a with its description.

    Notes
    -----

    References
    ----------

    Examples
    --------

    """
    def __init__(self):
        """Trims an aircraft given an aerodynamic database.

        Deprecation warning

        Extended Summary

        Parameters
        ----------
        x : type
            Description of parameter `x`.

        Attributes
        ----------
        exposure : float
            Exposure in seconds.

        Methods
        -------
        colorspace(c='rgb')
            Represent the photo in the given colorspace.

        Yields
        ------
        int
            Description of the anonymous integer return value.

        Receives
        --------

        Ohter Parameters
        ----------------

        Raises
        ------

        Warns
        -----

        Warnings
        --------

        See Also
        --------
        func_a : Function a with its description.

        Notes
        -----

        References
        ----------

        Examples
        --------

        """

    def import_aero_data(
            self, file_name, num_dimensions, num_pts_per_dimension,
            dimension_lims, **kwargs):
        """Imports aerodynamic data and saves it for use in the trim algorithm.

        Deprecation warning

        Extended Summary

        Parameters
        ----------
        x : type
            Description of parameter `x`.

        Attributes
        ----------
        exposure : float
            Exposure in seconds.

        Methods
        -------
        colorspace(c='rgb')
            Represent the photo in the given colorspace.

        Yields
        ------
        int
            Description of the anonymous integer return value.

        Receives
        --------

        Ohter Parameters
        ----------------

        Raises
        ------

        Warns
        -----

        Warnings
        --------

        See Also
        --------
        func_a : Function a with its description.

        Notes
        -----

        References
        ----------

        Examples
        --------

        """
        save_numpy = kwargs.get("save_numpy", False)
        save_sorted = kwargs.get("save_sorted", False)
        header_titles = kwargs.get("header_titles",
                                   ['AOA', 'Beta', 'd_e', 'd_a',
                                    'd_r', 'p', 'q', 'r'])
        database_params = kwargs.get("database_params", False)
        if not database_params:
            database_params = np.zeros((num_dimensions, num_pts_per_dimension))
            for i in range(len(num_dimensions)):
                database_params[i, :] = np.linspace(dimension_lims[i][0],
                                                    dimension_lims[i][1],
                                                    num_pts_per_dimension)
            database_params = tuple(database_params)
        if file_name[-4:] == '.csv':
            data_in, data_ND_array = self._import_csv(file_name,
                                                      num_dimensions,
                                                      num_pts_per_dimension,
                                                      header_titles)
            if save_sorted:
                data_in.to_csv("./" + file_name[:-4] + "_sorted.csv")
            if save_numpy:
                np.save("./" + file_name[:-4] + ".npy", data_ND_array)
        if file_name[-4:] == '.npy':
            data_ND_array = np.load(file_name)
        self._save_aero_data(data_ND_array, database_params)

    def _import_csv(
            self, file_name, num_dimensions, num_pts_per_dimension,
            header_titles):
        """Imports data from a .csv file to save as the aerodynamic database.

        Deprecation warning

        Extended Summary

        Parameters
        ----------
        x : type
            Description of parameter `x`.

        Attributes
        ----------
        exposure : float
            Exposure in seconds.

        Methods
        -------
        colorspace(c='rgb')
            Represent the photo in the given colorspace.

        Yields
        ------
        int
            Description of the anonymous integer return value.

        Receives
        --------

        Ohter Parameters
        ----------------

        Raises
        ------

        Warns
        -----

        Warnings
        --------

        See Also
        --------
        func_a : Function a with its description.

        Notes
        -----

        References
        ----------

        Examples
        --------

        """
        data_in = pd.read_csv(file_name, delimiter=',')
        data_in.sort_values(by=header_titles, inplace=True)
        data_in_array = data_in[["Cx", "Cy", "Cz","Cl","Cm","Cn"]].values.to_numpy()
        new_data_dims = [num_pts_per_dimension]*num_dimensions + [6]
        data_ND_array = np.resize(data_in_array, tuple(new_data_dims))
        return data_in, data_ND_array

    def _save_aero_data(self, aero_data, database_params):
        """Saves the aerodynamic coefficient data into a database.

        Deprecation warning

        Extended Summary

        Parameters
        ----------
        x : type
            Description of parameter `x`.

        Attributes
        ----------
        exposure : float
            Exposure in seconds.

        Methods
        -------
        colorspace(c='rgb')
            Represent the photo in the given colorspace.

        Yields
        ------
        int
            Description of the anonymous integer return value.

        Receives
        --------

        Ohter Parameters
        ----------------

        Raises
        ------

        Warns
        -----

        Warnings
        --------

        See Also
        --------
        func_a : Function a with its description.

        Notes
        -----

        References
        ----------

        Examples
        --------

        """
        rgi_options = {"bounds_error": False, "fill_value": None}
        self.Cx = rgi(database_params, aero_data[..., 0], **rgi_options)
        self.Cy = rgi(database_params, aero_data[..., 1], **rgi_options)
        self.Cz = rgi(database_params, aero_data[..., 2], **rgi_options)
        self.Cl = rgi(database_params, aero_data[..., 3], **rgi_options)
        self.Cm = rgi(database_params, aero_data[..., 4], **rgi_options)
        self.Cn = rgi(database_params, aero_data[..., 5], **rgi_options)

