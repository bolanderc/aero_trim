"""This module is an implementation of a general aerodynamic trim algorithm.

This module allows users to supply an aerodynamic database along with relevant
atmospheric and reference data to trim an aircraft in flight. The trim
algorithm implemented is a general algorithm allowing for the known parameters
to be specified by the user.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sys
import scipy.optimize as optimize

from scipy.interpolate import RegularGridInterpolator as rgi
import scipy.optimize as optimize


def bf2w_lift(Fxb, Fyb, Fzb, alpha, beta):
    Fzw = -Fxb*np.sin(alpha) + Fzb*np.cos(alpha)
    return -Fzw

def bf2w_side(Fxb, Fyb, Fzb, alpha, beta):
    Fyw = -Fxb*np.cos(alpha)*np.sin(beta) + Fyb*np.cos(beta) - Fzb*np.sin(alpha)*np.sin(beta)
    return Fyw

def bf2w_drag(Fxb, Fyb, Fzb, alpha, beta):
    Fxw = Fxb*np.cos(alpha)*np.cos(beta) + Fyb*np.sin(beta) + Fzb*np.sin(alpha)*np.cos(beta)
    return -Fxw

class TrimCase:
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
    import_aero_data(file_name, num_dimensions, num_pts_per_dimension,
                     dimension_lims, **kwargs)
        Import aerodynamic force and moment coefficients from a .csv or .npy
        file to generate a coefficient database for use in trimming the
        aircraft.

    Notes
    -----
    The trim alogrithm used in this class is based on the rigid-body 6-DOF
    equations of motion, which are presented in Phillips [1].

    References
    ----------
    [1]    Phillips, W. F., “Tigid-Body 6-DOF Equations of Motion,”
           Mechanics of Flight, John Wiley Sons, Inc., 2010, Chap. 7, 2nd ed.,
           p.753.

    """
    def __init__(self, v_free, rho_free, units="English", **kwargs):
        """Trims an aircraft given an aerodynamic database.

        This class creates an aerodynamic database from user-provided data which
        can then be used to trim an aircraft. The trim algorithm implemented is
        general in nature, meaning that the user specifies which state information
        they know about the aircraft and which will determined from the trim
        algorithm.

        Parameters
        ----------
        v_free : float
            The freestream velocity of the aircraft at the trim flight
            condition.

        rho_free : float
            The freestream density at the trim flight condition.

        units : string, default="English"
            The units for the input parameters: either `Metric` or `English`.

        weight : float, default=20500.
            The weight of the aircraft.

        ang_mom : list of floats, default=[160., 0., 0.]
            The angular momentum vector for all spinning rotors relative to
            the body-fixed coordinate system. See Phillips [1].

        intertia : array_like, default=[[9496., 0., -982.],
                                        [0., 55814., 0.],
                                        [-982., 0., 63100.]]
            The inertia matrix of the aircraft in the body-fixed coordinate
            system.

        ref_area : float, default=300.
            The reference area of the aircraft used for non-dimensionalization.

        lat_ref_len : float, default=30.
            The lateral reference length of the aircraft used for
            non-dimensionalization.

        lon_ref_len : float, default=11.32
            The longitudinal reference length of the aircraft used for
            non-dimensionalization.

        Attributes
        ----------
        W : float
            The weight of the aircraft

        g : float
            Gravitational constant.

        ang_mom_mat : array_like
            The matrix prescribing the effect of spinning rotors on the
            angular momentum.

        interia_mat : array_like
            The intertia tensor of the aircraft.

        ref_area : float
            The reference area of the aircraft used for non-dimensionalization.

        lat_ref_len : float
            The lateral reference length of the aircraft used for
            non-dimensionalization.

        lon_ref_len : float
            The longitudinal reference length of the aircraft used for
            non-dimensionalization.

        V : float
            The freestream velocity magnitude of the aircraft at the trim
            flight condition.

        rho : float
            The freestream density at the trim flight condition.

        nondim_coeff : float
            The coefficient used for non-dimensionalizing the aerodynamic
            forces and moments acting on the aircraft.

        trim_solution : array_like
            Contains the parameters that will trim the aircraft at the flight
            condition specified.

        Notes
        -----
        The variable names implemented in this class try to be explicit in
        their nomenclature, except where the nomenclature is generally
        accepted in aerodynamics. Examples include `W` as the gross weight of
        the aircraft and `g` as the gravitational constant. Note also that
        English units are used by default, but metric units may be used
        as long as all of the keyword input parameters are consistent in
        their units. The default values of the keyword inputs are for the F-16
        as given by Stevens [2].

        References
        ----------
        [1]    Phillips, W. F., “Rigid-Body 6-DOF Equations of Motion,”
               Mechanics of Flight, John Wiley Sons, Inc., 2010, Chap. 7, 2nd ed.,
               p. 753.

        [2]    Stevens, B. L., and Lewis, F. L., “F-16 Model,” Aircraft Control
               and Simulation, John Wiley Sons, Inc., 2003, Appendix A,
               2nd ed., pp. 633-641.

        Examples
        --------
        >>> import aero_trim
        >>>
        >>> v_free, rho_free = 222.5211, 0.0023084
        >>> kwargs = {"weight": 20500.,
                      "ang_mom": [100., 0., 0.],
                      "intertia": [[10000., 0., -400],
                            [0., 60000., 0.],
                            [-400, 0., 45000.]],
                      "ref_area": 200.,
                      "lat_ref_len": 10,
                      "lon_ref_len": 10}
        >>> trim_case = aero_trim.TrimCase(v_free, rho_free, units="English",
                                           **kwargs)

        """
        self.W = kwargs.get("weight", 20500.)
        if units.lower() == "metric":
            self.g = 9.81
        else:
            self.g = 32.2
        ang_mom_vec = kwargs.get("ang_mom", [160., 0, 0])
        hx, hy, hz = ang_mom_vec
        self.ang_mom_mat = np.array([[0., -hz, hy],
                                     [hz, 0., -hx],
                                     [-hy, hx, 0]])
        self.hx = hx
        self.hy = hy
        self.hz = hz
        inertia_mat = kwargs.get("intertia",
                                 np.array([[9496., 0., -982.],
                                           [0., 55814., 0.],
                                           [-982., 0., 63100.]]))
        self.Ixx, self.Iyy, self.Izz = np.diag(inertia_mat)
        self.Ixy = -inertia_mat[0, 1]
        self.Ixz = -inertia_mat[0, 2]
        self.Iyz = -inertia_mat[1, 2]
        self.ref_area = kwargs.get("ref_area", 300.)
        self.lat_ref_len = kwargs.get("lat_ref_len", 30.)
        self.lon_ref_len = kwargs.get("lon_ref_len", 11.32)
        self.V = v_free
        self.rho = rho_free
        self.nondim_coeff = 0.5*rho_free*v_free**2*self.ref_area
        self.CW = self.W/self.nondim_coeff
        self.trim_solution = np.zeros(6)

        # Default Linear Model @ T1
        self.CL0 = 0.0876
        self.CL_alpha = 4.0314
        self.CL_qbar = 3.7263
        self.CL_de = 0.6341
        self.CS_beta = -0.5512
        self.CS_pbar = 0.0165
        self.CS_rbar = 0.6411
        self.CS_da = 0.1011
        self.CS_dr = 0.2052
        self.CD0 = 0.0687
        self.CD1 = 0.0029
        self.CD2 = 0.1051
        self.CD3 = 0.2148
        self.CD_qbar = 0.0366
        self.CD_de = -0.0077
        self.Cell_beta = -0.0939
        self.Cell_pbar = -0.4419
        self.Cell_rbar = 0.0975
        self.Cell_da = -0.1223
        self.Cell_dr = 0.0332
        self.Cm0 = -0.0048
        self.Cm_alpha = -0.5889
        self.Cm_qbar = -5.0267
        self.Cm_de = -0.7826
        self.Cn_beta = 0.2958
        self.Cn_pbar = 0.0057
        self.Cn_rbar = -0.3041
        self.Cn_da = -0.0432
        self.Cn_dr = -0.1071
        self.model = "linear"
        self._w2b_linear_coeffs_conversion()



    def import_aero_data(self, file_name, *args, **kwargs):
        """Imports aerodynamic data and saves it for use in the trim algorithm.

        Body-fixed aerodynamic force and moment coefficient data in the form
        of either a .csv or .npy file is converted to linearly interpolated
        functions of forces and moment data that can be called by the trim
        algorithm given aerodynamic angles, control surface deflections, and
        body-fixed rotation rates.

        Parameters
        ----------
        file_name : string
            The file name containing the aerodynamic coefficient data. Only
            .csv and .npy file types are supported at this time.

        num_dimensions : int
            The number of dimensions in the aerodynamic coefficient data not
            including the aerodynamic forces and moments. E.g. data as a
            function of angle of attack, sideslip, elevator, aileron, and
            rudder deflection angles, and the three body-fixed rotation rates
            would have 8 dimensions.

        num_pts_per_dimension : int or list
            The number of data points in each dimension. If not constant in
            each dimension, then this parameter should be a list and the
            `database_params` keyword should be specified.

        dimension_lims : list of tuple
            A list of tuples indicating the minimum and maximum values of
            each parameter dimension.

        save_numpy : bool, default=False
            Saves the sorted array of aerodynamic coefficients as a numpy
            array of dimension [n, n, n, ..., n, 6] where `n` is the value of
            `num_pts_per_dimension`. This is useful when first running the
            import on a .csv file as the .npy import is much faster.

        save_sorted : bool, default=False
            Saves a .csv file containing the sorted aerodynamic coefficient
            data.

        header_titles : list of str
            Indicates the header titles of the .csv file containing the
            aerodynamic coefficient data. Used in sorting the .csv file.
            Defaults to ['AOA', 'Beta', 'd_e', 'd_a', 'd_r', 'p', 'q', 'r'].

        database_params : tuple of arrays
            A tuple of arrays where each array is the range of cases performed
            in each dimension of the aerodynamic database. Should be specified
            if `num_pts_per_dimension` is not constant.

        model : str, default='database'
            One of 'linear', 'morelli', or 'database' depending on which
            aerodynamic model is desired for the aerodynamic coefficients.

        Methods
        -------
        _import_csv(file_name, num_dimensions, num_pts_per_dimension,
            header_titles)
            Imports a .csv file containing aerodynamic force and moment
            coefficient data, sorts the data, and then returns an array of
            dimension [n, n, n, ..., n, 6] where `n` is the value of
            `num_pts_per_dimension`.

        _save_aero_data(aero_data, database_params)
            Saves the aerodynamic database into force and moment function
            class attributes that can be called by the trim algorithm.

        _data_conversion(aero_data)
            Converts body-fixed aerodynamic force coefficients to the wind
            frame for the linear least-squares model. Also converts
            dimensioned rotation rates to dimensionless rates.

        _linear_fits(converted_aero_data)
            Takes converted aerodynamic data and calculates linear least-
            squares coefficients according to the chosen aerodynamic model.

        _w2b_linear_coeffs_conversion()
            Converts linear least-squares coefficients in the wind frame to
            coefficients in the body-fixed frame for the trim algorithm.

        Raises
        ------
        TypeError
            If the `num_pts_per_dimension` parameter is a list and
            `database_params` is not specified.

        See Also
        --------
        func_a : Function a with its description.

        Examples
        --------
        >>> import aero_trim
        >>>
        >>>
        >>> file_name = "test_database.csv"
        >>> v_free, rho_free = 222.5211, 0.0023084
        >>> trim_case = aero_trim.TrimCase(v_free, rho_free, units="English")
        >>> num_dimensions = 8
        >>> num_pts_per_dimension = 5
        >>> dimension_lims = [(-15., 15.),
                              (-15., 15.),
                              (-21.5/4., 21.5/4.),
                              (-25., 25.),
                              (-30., 30.),
                              (-1.2, 1.2),
                              (-1.2, 1.2),
                              (-0.3925, 0.3925)]
        >>> trim_case.import_aero_data(file_name, num_dimensions,
                                       num_pts_per_dimension, dimension_lims)

        >>> import aero_trim
        >>>
        >>>
        >>> file_name = "test_database.csv"
        >>> v_free, rho_free = 222.5211, 0.0023084
        >>> trim_case = aero_trim.TrimCase(v_free, rho_free, units="English")
        >>> trim_case.import_aero_data(file_name, model='linear')

        """
        model = kwargs.get("model", "linear")
        if model == "database":
            save_numpy = kwargs.get("save_numpy", False)
            save_sorted = kwargs.get("save_sorted", False)
            header_titles = kwargs.get("header_titles",
                                       ['AOA', 'Beta', 'd_e', 'd_a',
                                        'd_r', 'p', 'q', 'r'])
            try:
                num_dimensions, num_pts_per_dimension, dimension_lims = (*args,)
            except ValueError:
                raise TypeError('required arguments for database import not '\
                                'specified')
        elif model == "linear":
            self.model = "linear"
            aero_data = np.loadtxt(file_name, delimiter=",", skiprows=1, usecols=range(14))
            converted_aero_data = self._data_conversion(aero_data)
            self._linear_fits(converted_aero_data)
            self._w2b_linear_coeffs_conversion()

        elif model == "bire":
            self.model = "bire"
            d_B = kwargs.get("d_B", np.linspace(-90, 90, 11))
            aero_data = np.loadtxt(file_name, delimiter=",", skiprows=1)
            converted_aero_data = self._data_conversion(aero_data)
            self._bire_linear_fits(converted_aero_data, d_B)
            self._w2b_linear_coeffs_conversion()
            self._bire_fits_dB(d_B)

        elif model == "morelli":
            self.model = "morelli"
            pass
        else:
            self.model = "database"
            # Get rid of this and implement the version in example.py. Perhaps
            # force the user to simply specify it as a list always.
            database_params = kwargs.get("database_params", False)
            if not database_params:
                try:
                    int(num_pts_per_dimension)
                except TypeError:
                    raise TypeError("Please specify `database_params` if " \
                                    "`num_pts_per_dimension` is not an " \
                                    "integer.")
                database_params = np.zeros((num_dimensions, num_pts_per_dimension))
                for i in range(num_dimensions):
                    database_params[i, :] = np.linspace(dimension_lims[i][0],
                                                        dimension_lims[i][1],
                                                        num_pts_per_dimension)
                database_params = tuple(database_params)
            if file_name[-4:] == '.csv':
                data_in, data_nd_array = self._import_csv(file_name,
                                                          num_dimensions,
                                                          num_pts_per_dimension,
                                                          header_titles)
                if save_sorted:
                    data_in.to_csv("./" + file_name[:-4] + "_sorted.csv")
                if save_numpy:
                    np.save("./" + file_name[:-4] + ".npy", data_nd_array)
            if file_name[-4:] == '.npy':
                data_nd_array = np.load(file_name)
            self._save_aero_data(data_nd_array, database_params)


    def _import_csv(
            self, file_name, num_dimensions, num_pts_per_dimension,
            header_titles):
        """Imports data from a .csv file to save as the aerodynamic database.

        Imports, sorts, and resizes aerodynamic force and moment data from a
        .csv file.

        Parameters
        ----------
        file_name : string
            The .csv file name containing the aerodynamic coefficient data.

        num_dimensions : int
            The number of dimensions in the aerodynamic coefficient data not
            including the aerodynamic forces and moments. E.g. data as a
            function of angle of attack, sideslip, elevator, aileron, and
            rudder deflection angles, and the three body-fixed rotation rates
            would have 8 dimensions.

        num_pts_per_dimension : int or list
            The number of data points in each dimension.

        header_titles : list of str
            Indicates the header titles of the .csv file containing the
            aerodynamic coefficient data. Used in sorting the .csv file.
            Defaults to ['AOA', 'Beta', 'd_e', 'd_a', 'd_r', 'p', 'q', 'r'].

        Returns
        ------
        data_in : DataFrame
            Pandas DataFrame class containing the sorted aerodynamic data from
            the .csv file.

        data_nd_array : array_like
            Numpy N-d array of dimension [n, n, n, ..., n, 6] where `n` is the
            value of `num_pts_per_dimension`.

        See Also
        --------
        pandas.read_csv : Read a comma-separated values (csv) file into
            DataFrame.

        pandas.DataFrame.sort_values : Sort by the values along any axis.

        pandas.DataFrame.to_numpy : Convert a DataFrame to a NumPy array.

        numpy.resize : Return a new array with the specified shape.

        """
        data_in = pd.read_csv(file_name, delimiter=',')
        data_in.sort_values(by=header_titles, inplace=True)
        data_in_array = data_in[["Cx", "Cy", "Cz","Cl","Cm","Cn"]].to_numpy()
        if isinstance(num_pts_per_dimension, int):
            new_data_dims = [num_pts_per_dimension]*num_dimensions + [6]
        else:
            new_data_dims = num_pts_per_dimension + [6]
        data_nd_array = np.resize(data_in_array, tuple(new_data_dims))
        return data_in, data_nd_array

    def _save_aero_data(self, aero_data, database_params):
        """Saves the aerodynamic coefficient data into a database.

        Takes aerodynamic coefficient data and linearly-interpolates along each
        dimension to return a function that will provide data given the
        appropriate flight condition in the user-specified dimensions.

        Parameters
        ----------
        aero_data : array_like
            Aerodynamic data of dimension [n, n, n, ..., n, 6] where `n` is the
            value of `num_pts_per_dimension`.

        database_params : tuple of arrays
            A tuple of arrays where each array is the range of cases performed
            in each dimension of the aerodynamic database.

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

        See Also
        --------
        scipy.interpolate.RegularGridInterpolator : Interpolation on a regular
            grid in arbitrary dimensions.

        """
        rgi_options = {"bounds_error": False, "fill_value": None}
        self.c_x = rgi(database_params, aero_data[..., 0], **rgi_options)
        self.c_y = rgi(database_params, aero_data[..., 1], **rgi_options)
        self.c_z = rgi(database_params, aero_data[..., 2], **rgi_options)
        self.c_ell = rgi(database_params, aero_data[..., 3], **rgi_options)
        self.c_m = rgi(database_params, aero_data[..., 4], **rgi_options)
        self.c_n = rgi(database_params, aero_data[..., 5], **rgi_options)

    def _linear_fits(self, aero_data):
        """Performs a least-squares fit to generate coefficients for a linear
        aerodynamic model.

        Takes an aerodynamic database and performs a least-squares fit to
        construct a linear aerodynamic model. This model is of the form given
        in the Notes section.

        Parameters
        ----------
        aero_data : array_like
            Aerodynamic data with the data arranged so that the columns are
            in the order: alpha, beta, d_e, d_a, d_r, pbar, qbar, rbar, C_D,
            C_S, C_L, C_l, C_m, C_n.

        Attributes
        ----------
        coeffs_lift : array_like
            Linear least-squares coefficients for the lift coefficient.

        CL0 : float
            Least-squares coefficient for the lift coefficient at zero
            aerodynamic angles, control surface deflections, and rotation
            rates.

        CL_alpha : float
            Least-squares coefficient for the lift slope.

        CL_qbar : float
            Least-squares coefficient for the change in lift coefficient with
            respect to the dimensionless pitch rate.

        CL_de : float
            Least-squares coefficient for the change in lift coefficient with
            respect to elevator deflection.

        coeffs_pitch : array_like
            Linear least-squares coefficients for the pitching moment
            coefficient.

        Cm0 : float
            Least-squares coefficient for the pitching moment coefficient at
            zero aerodynamic angles, control surface deflections, and
            rotation rates.

        Cm_alpha : float
            Least-squares coefficient for the pitching moment slope.

        Cm_qbar : float
            Least-squares coefficient for the change in pitching moment
            coefficient with respect to the dimensionless pitch rate.

        Cm_de : float
            Least-squares coefficient for the change in pitching moment
            coefficient with respect to elevator deflection.

        coeffs_side : array_like
            Linear least-squares coefficients for the side force coefficient.

        CS_beta : float
            Least-squares coefficient for the change in side force coefficient
            with respect to sideslip angle.

        CS_pbar : float
            Least-squares coefficient for the change in side force coefficient
            with respect to the dimensionless roll rate.

        CS_rbar : float
            Least-squares coefficient for the change in side force coefficient
            with respect to the dimensionless yaw rate.

        CS_da : float
            Least-squares coefficient for the change in side force coefficient
            with respect to aileron deflection.

        CS_dr : float
            Least-squares coefficient for the change in side force coefficient
            with respect to rudder deflection.

        coeffs_roll : array_like
            Linear least-squares coefficients for the rolling moment
            coefficient.

        Cell_beta : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to sideslip angle.

        Cell_pbar : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to the dimensionless roll rate.

        Cell_rbar : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to the dimensionless yaw rate.

        Cell_da : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to aileron deflection.

        Cell_dr : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to rudder deflection.

        coeffs_yaw : array_like
            Linear least-squares coefficients for the yawing moment
            coefficient.

        Cn_beta : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to sideslip angle.

        Cn_pbar : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to the dimensionless roll rate.

        Cn_rbar : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to the dimensionless yaw rate.

        Cn_da : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to aileron deflection.

        Cn_dr : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to rudder deflection.

        coeffs_drag : array_like
            Linear least-squares coefficients for the drag coefficient.

        CD0 : float
            Least-squares coefficient for the drag coefficient at zero
            aerodynamic angles, control surface deflections, and rotation
            rates.

        CD1 : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the lift coefficient.

        CD2 : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the lift coefficient squared.

        CD3 : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the side force coefficient squared.

        CD_qbar : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the dimensionless pitch rate.

        CD_de : float
            Least-squares coefficient for the change in drag coefficient with
            respect to elevator deflection.

        See Also
        --------
        numpy.linalg.lstsq : Return the least-squares solution to a linear
                             matrix equation.

        Notes
        -----
        The linear aerodynamic model is of the form:

            CL = C_L0 + C_L,alpha*alpha + C_L,qbar*qbar + C_L,delta_e*delta_e

            CS = C_S,beta*beta + C_S,pbar*pbar + C_S,rbar*rbar +
                 C_S,delta_a*delta_a + C_S,delta_r*delta_r

            CD = C_D0 + C_D1*CL + C_D2*CL**2 + C_D3*CS**2 + C_D,qbar*qbar +
                 C_D,delta_e*delta_e

            Cl = C_l,beta*beta + C_l,pbar*pbar + C_l,rbar*rbar +
                 C_l,delta_a*delta_a + C_l,delta_r*delta_r

            Cm = C_m0 + C_m,alpha*alpha + C_m,qbar*qbar + C_m,delta_e*delta_e

            Cn = C_n,beta*beta + C_n,pbar*pbar + C_n,rbar*rbar +
                 C_n,delta_a*delta_a + C_n,delta_r*delta_r

        """

        # Re-organize data
        N = len(aero_data[:, 0])
        A = np.zeros((N, 9))
        b_lift = aero_data[:, 10]
        b_side = aero_data[:, 9]
        b_drag = aero_data[:, 8]
        b_roll = aero_data[:, 11]
        b_pitch = aero_data[:, 12]
        b_yaw = aero_data[:, 13]
        alpha = np.deg2rad(aero_data[:, 0])
        beta = np.deg2rad(aero_data[:, 1])
        d_e = np.deg2rad(aero_data[:, 2])
        d_a = np.deg2rad(aero_data[:, 3])
        d_r = np.deg2rad(aero_data[:, 4])
        pbar = aero_data[:, 5]
        qbar = aero_data[:, 6]
        rbar = aero_data[:, 7]

        # Construct A matrix for full linear least-squares problem.
        A[:, 0] = 1.
        A[:, 1] = alpha
        A[:, 2] = beta
        A[:, 3] = pbar
        A[:, 4] = qbar
        A[:, 5] = rbar
        A[:, 6] = d_a
        A[:, 7] = d_e
        A[:, 8] = d_r

        # Solve for linear least-squares lift coefficients.
        lst_sq_res = np.linalg.lstsq(A, b_lift, rcond=None)
        self.coeffs_lift = lst_sq_res[0]

        # Save relevant coefficients for chosen linear aerodynamic model.
        self.CL0 = self.coeffs_lift[0]
        self.CL_alpha = self.coeffs_lift[1]
        self.CL_qbar = self.coeffs_lift[4]
        self.CL_de = self.coeffs_lift[7]

        # Solve for linear least-squares pitch coefficients.
        lst_sq_res = np.linalg.lstsq(A, b_pitch, rcond=None)
        self.coeffs_pitch = lst_sq_res[0]

        # Save relevant coefficients for chosen linear aerodynamic model.
        self.Cm0 = self.coeffs_pitch[0]
        self.Cm_alpha = self.coeffs_pitch[1]
        self.Cm_qbar = self.coeffs_pitch[4]
        self.Cm_de = self.coeffs_pitch[7]

        # Solve for linear least-squares side force coefficients.
        lst_sq_res = np.linalg.lstsq(A, b_side, rcond=None)
        self.coeffs_side = lst_sq_res[0]

        # Save relevant coefficients for chosen linear aerodynamic model.
        self.CS_beta = self.coeffs_side[2]
        self.CS_pbar = self.coeffs_side[3]
        self.CS_rbar = self.coeffs_side[5]
        self.CS_da = self.coeffs_side[6]
        self.CS_dr = self.coeffs_side[8]

        # Solve for linear least-squares roll coefficients.
        lst_sq_res = np.linalg.lstsq(A, b_roll, rcond=None)
        self.coeffs_roll = lst_sq_res[0]

        # Save relevant coefficients for chosen linear aerodynamic model.
        self.Cell_beta = self.coeffs_roll[2]
        self.Cell_pbar = self.coeffs_roll[3]
        self.Cell_rbar = self.coeffs_roll[5]
        self.Cell_da = self.coeffs_roll[6]
        self.Cell_dr = self.coeffs_roll[8]

        # Solve for linear least-squares yaw coefficients.
        lst_sq_res = np.linalg.lstsq(A, b_yaw, rcond=None)
        self.coeffs_yaw = lst_sq_res[0]

        # Save relevant coefficients for chosen linear aerodynamic model.
        self.Cn_beta = self.coeffs_yaw[2]
        self.Cn_pbar = self.coeffs_yaw[3]
        self.Cn_rbar = self.coeffs_yaw[5]
        self.Cn_da = self.coeffs_yaw[6]
        self.Cn_dr = self.coeffs_yaw[8]

        # Construct A matrix for linear least-squares problem specific to drag
        # model.
        A = np.zeros((N, 10))
        A[:, 0] = 1.
        A[:, 1] = b_lift
        A[:, 2] = np.square(b_lift)
        A[:, 3] = np.square(b_side)
        A[:, 4] = pbar
        A[:, 5] = qbar
        A[:, 6] = rbar
        A[:, 7] = d_a
        A[:, 8] = d_e
        A[:, 9] = d_r

        # Solve for linear least-squares drag coefficients.
        lst_sq_res = np.linalg.lstsq(A, b_drag, rcond=None)
        self.coeffs_drag = lst_sq_res[0]

        # Save relevant coefficients for chosen linear aerodynamic model.
        self.CD0 = self.coeffs_drag[0]
        self.CD1 = self.coeffs_drag[1]
        self.CD2 = self.coeffs_drag[2]
        self.CD3 = self.coeffs_drag[3]
        self.CD_qbar = self.coeffs_drag[5]
        self.CD_de = self.coeffs_drag[8]

    def _bire_linear_fits(self, aero_data, d_B):
        """Performs a least-squares fit to generate coefficients for a linear
        aerodynamic model for the BIRE configuration.

        Takes an aerodynamic database and performs a least-squares fit to
        construct a linear aerodynamic model. This model is of the form given
        in the Notes section.

        Parameters
        ----------
        aero_data : array_like
            Aerodynamic data with the data arranged so that the columns are
            in the order: alpha, beta, d_e, d_a, d_B, pbar, qbar, rbar, C_D,
            C_S, C_L, C_l, C_m, C_n.

        Attributes
        ----------
        coeffs_lift : array_like
            Linear least-squares coefficients for the lift coefficient.

        CL0 : float
            Least-squares coefficient for the lift coefficient at zero
            aerodynamic angles, control surface deflections, and rotation
            rates.

        CL_alpha : float
            Least-squares coefficient for the lift slope.

        CL_qbar : float
            Least-squares coefficient for the change in lift coefficient with
            respect to the dimensionless pitch rate.

        CL_de : float
            Least-squares coefficient for the change in lift coefficient with
            respect to elevator deflection.

        coeffs_pitch : array_like
            Linear least-squares coefficients for the pitching moment
            coefficient.

        Cm0 : float
            Least-squares coefficient for the pitching moment coefficient at
            zero aerodynamic angles, control surface deflections, and
            rotation rates.

        Cm_alpha : float
            Least-squares coefficient for the pitching moment slope.

        Cm_qbar : float
            Least-squares coefficient for the change in pitching moment
            coefficient with respect to the dimensionless pitch rate.

        Cm_de : float
            Least-squares coefficient for the change in pitching moment
            coefficient with respect to elevator deflection.

        coeffs_side : array_like
            Linear least-squares coefficients for the side force coefficient.

        CS_beta : float
            Least-squares coefficient for the change in side force coefficient
            with respect to sideslip angle.

        CS_pbar : float
            Least-squares coefficient for the change in side force coefficient
            with respect to the dimensionless roll rate.

        CS_rbar : float
            Least-squares coefficient for the change in side force coefficient
            with respect to the dimensionless yaw rate.

        CS_da : float
            Least-squares coefficient for the change in side force coefficient
            with respect to aileron deflection.

        CS_dr : float
            Least-squares coefficient for the change in side force coefficient
            with respect to rudder deflection.

        coeffs_roll : array_like
            Linear least-squares coefficients for the rolling moment
            coefficient.

        Cell_beta : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to sideslip angle.

        Cell_pbar : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to the dimensionless roll rate.

        Cell_rbar : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to the dimensionless yaw rate.

        Cell_da : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to aileron deflection.

        Cell_dr : float
            Least-squares coefficient for the change in rolling moment
            coefficient with respect to rudder deflection.

        coeffs_yaw : array_like
            Linear least-squares coefficients for the yawing moment
            coefficient.

        Cn_beta : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to sideslip angle.

        Cn_pbar : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to the dimensionless roll rate.

        Cn_rbar : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to the dimensionless yaw rate.

        Cn_da : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to aileron deflection.

        Cn_dr : float
            Least-squares coefficient for the change in yawing moment
            coefficient with respect to rudder deflection.

        coeffs_drag : array_like
            Linear least-squares coefficients for the drag coefficient.

        CD0 : float
            Least-squares coefficient for the drag coefficient at zero
            aerodynamic angles, control surface deflections, and rotation
            rates.

        CD1 : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the lift coefficient.

        CD2 : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the lift coefficient squared.

        CD3 : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the side force coefficient squared.

        CD_qbar : float
            Least-squares coefficient for the change in drag coefficient with
            respect to the dimensionless pitch rate.

        CD_de : float
            Least-squares coefficient for the change in drag coefficient with
            respect to elevator deflection.

        See Also
        --------
        numpy.linalg.lstsq : Return the least-squares solution to a linear
                             matrix equation.

        Notes
        -----
        The linear aerodynamic model is of the form:

            CL = C_L0 + C_L,alpha*alpha + C_L,qbar*qbar + C_L,delta_e*delta_e

            CS = C_S,beta*beta + C_S,pbar*pbar + C_S,rbar*rbar +
                 C_S,delta_a*delta_a + C_S,delta_r*delta_r

            CD = C_D0 + C_D1*CL + C_D2*CL**2 + C_D3*CS**2 + C_D,qbar*qbar +
                 C_D,delta_e*delta_e

            Cl = C_l,beta*beta + C_l,pbar*pbar + C_l,rbar*rbar +
                 C_l,delta_a*delta_a + C_l,delta_r*delta_r

            Cm = C_m0 + C_m,alpha*alpha + C_m,qbar*qbar + C_m,delta_e*delta_e

            Cn = C_n,beta*beta + C_n,pbar*pbar + C_n,rbar*rbar +
                 C_n,delta_a*delta_a + C_n,delta_r*delta_r

        """
        n_dB = len(d_B)
        self.coeffs_lift = np.zeros((n_dB, 8))
        self.CL0 = np.zeros(n_dB)
        self.CL_alpha = np.zeros(n_dB)
        self.CL_beta = np.zeros(n_dB)
        self.CL_pbar = np.zeros(n_dB)
        self.CL_qbar = np.zeros(n_dB)
        self.CL_rbar = np.zeros(n_dB)
        self.CL_da = np.zeros(n_dB)
        self.CL_de = np.zeros(n_dB)

        self.coeffs_side = np.zeros((n_dB, 8))
        self.CS0 = np.zeros(n_dB)
        self.CS_alpha = np.zeros(n_dB)
        self.CS_beta = np.zeros(n_dB)
        self.CS_pbar = np.zeros(n_dB)
        self.CS_qbar = np.zeros(n_dB)
        self.CS_rbar = np.zeros(n_dB)
        self.CS_da = np.zeros(n_dB)
        self.CS_de = np.zeros(n_dB)

        self.coeffs_drag = np.zeros((n_dB, 9))
        self.CD0 = np.zeros(n_dB)
        self.CD1 = np.zeros(n_dB)
        self.CD2 = np.zeros(n_dB)
        self.CD3 = np.zeros(n_dB)
        self.CD_pbar = np.zeros(n_dB)
        self.CD_qbar = np.zeros(n_dB)
        self.CD_rbar = np.zeros(n_dB)
        self.CD_da = np.zeros(n_dB)
        self.CD_de = np.zeros(n_dB)

        self.coeffs_roll = np.zeros((n_dB, 8))
        self.Cell0 = np.zeros(n_dB)
        self.Cell_alpha = np.zeros(n_dB)
        self.Cell_beta = np.zeros(n_dB)
        self.Cell_pbar = np.zeros(n_dB)
        self.Cell_qbar = np.zeros(n_dB)
        self.Cell_rbar = np.zeros(n_dB)
        self.Cell_da = np.zeros(n_dB)
        self.Cell_de = np.zeros(n_dB)

        self.coeffs_pitch = np.zeros((n_dB, 8))
        self.Cm0 = np.zeros(n_dB)
        self.Cm_alpha = np.zeros(n_dB)
        self.Cm_beta = np.zeros(n_dB)
        self.Cm_pbar = np.zeros(n_dB)
        self.Cm_qbar = np.zeros(n_dB)
        self.Cm_rbar = np.zeros(n_dB)
        self.Cm_da = np.zeros(n_dB)
        self.Cm_de = np.zeros(n_dB)

        self.coeffs_yaw = np.zeros((n_dB, 8))
        self.Cn0 = np.zeros(n_dB)
        self.Cn_alpha = np.zeros(n_dB)
        self.Cn_beta = np.zeros(n_dB)
        self.Cn_pbar = np.zeros(n_dB)
        self.Cn_qbar = np.zeros(n_dB)
        self.Cn_rbar = np.zeros(n_dB)
        self.Cn_da = np.zeros(n_dB)
        self.Cn_de = np.zeros(n_dB)

        for i in range(n_dB):
            dBi = d_B[i]
            mask = aero_data[:, 4] == dBi
            aero_data_dBi = np.copy(aero_data[mask, :])

            # Re-organize data
            N = len(aero_data_dBi[:, 0])
            A = np.zeros((N, 8))
            b_lift = aero_data_dBi[:, 10]
            b_side = aero_data_dBi[:, 9]
            b_drag = aero_data_dBi[:, 8]
            b_roll = aero_data_dBi[:, 11]
            b_pitch = aero_data_dBi[:, 12]
            b_yaw = aero_data_dBi[:, 13]
            alpha = np.deg2rad(aero_data_dBi[:, 0])
            beta = np.deg2rad(aero_data_dBi[:, 1])
            d_e = np.deg2rad(aero_data_dBi[:, 2])
            d_a = np.deg2rad(aero_data_dBi[:, 3])
            pbar = aero_data_dBi[:, 5]
            qbar = aero_data_dBi[:, 6]
            rbar = aero_data_dBi[:, 7]

            # Construct A matrix for full linear least-squares problem.
            A[:, 0] = 1.
            A[:, 1] = alpha
            A[:, 2] = beta
            A[:, 3] = pbar
            A[:, 4] = qbar
            A[:, 5] = rbar
            A[:, 6] = d_a
            A[:, 7] = d_e

            # Solve for linear least-squares lift coefficients.
            lst_sq_res = np.linalg.lstsq(A, b_lift, rcond=None)
            self.coeffs_lift[i, :] = lst_sq_res[0]

            # Save relevant coefficients for chosen linear aerodynamic model.
            self.CL0[i] = self.coeffs_lift[i, 0]
            self.CL_alpha[i] = self.coeffs_lift[i, 1]
            self.CL_beta[i] = self.coeffs_lift[i, 2]
            self.CL_pbar[i] = self.coeffs_lift[i, 3]
            self.CL_qbar[i] = self.coeffs_lift[i, 4]
            self.CL_rbar[i] = self.coeffs_lift[i, 5]
            self.CL_da[i] = self.coeffs_lift[i, 6]
            self.CL_de[i] = self.coeffs_lift[i, 7]

            # Solve for linear least-squares pitch coefficients.
            lst_sq_res = np.linalg.lstsq(A, b_pitch, rcond=None)
            self.coeffs_pitch[i, :] = lst_sq_res[0]

            # Save relevant coefficients for chosen linear aerodynamic model.
            self.Cm0[i] = self.coeffs_pitch[i, 0]
            self.Cm_alpha[i] = self.coeffs_pitch[i, 1]
            self.Cm_beta[i] = self.coeffs_pitch[i, 2]
            self.Cm_pbar[i] = self.coeffs_pitch[i, 3]
            self.Cm_qbar[i] = self.coeffs_pitch[i, 4]
            self.Cm_rbar[i] = self.coeffs_pitch[i, 5]
            self.Cm_da[i] = self.coeffs_pitch[i, 6]
            self.Cm_de[i] = self.coeffs_pitch[i, 7]

            # Solve for linear least-squares side force coefficients.
            lst_sq_res = np.linalg.lstsq(A, b_side, rcond=None)
            self.coeffs_side[i, :] = lst_sq_res[0]

            # Save relevant coefficients for chosen linear aerodynamic model.
            self.CS0[i] = self.coeffs_side[i, 0]
            self.CS_alpha[i] = self.coeffs_side[i, 1]
            self.CS_beta[i] = self.coeffs_side[i, 2]
            self.CS_pbar[i] = self.coeffs_side[i, 3]
            self.CS_qbar[i] = self.coeffs_side[i, 4]
            self.CS_rbar[i] = self.coeffs_side[i, 5]
            self.CS_da[i] = self.coeffs_side[i, 6]
            self.CS_de[i] = self.coeffs_side[i, 7]

            # Solve for linear least-squares roll coefficients.
            lst_sq_res = np.linalg.lstsq(A, b_roll, rcond=None)
            self.coeffs_roll[i, :] = lst_sq_res[0]

            # Save relevant coefficients for chosen linear aerodynamic model.
            self.Cell0[i] = self.coeffs_roll[i, 0]
            self.Cell_alpha[i] = self.coeffs_roll[i, 1]
            self.Cell_beta[i] = self.coeffs_roll[i, 2]
            self.Cell_pbar[i] = self.coeffs_roll[i, 3]
            self.Cell_qbar[i] = self.coeffs_roll[i, 4]
            self.Cell_rbar[i] = self.coeffs_roll[i, 5]
            self.Cell_da[i] = self.coeffs_roll[i, 6]
            self.Cell_de[i] = self.coeffs_roll[i, 7]

            # Solve for linear least-squares yaw coefficients.
            lst_sq_res = np.linalg.lstsq(A, b_yaw, rcond=None)
            self.coeffs_yaw[i, :] = lst_sq_res[0]

            # Save relevant coefficients for chosen linear aerodynamic model.
            self.Cn0[i] = self.coeffs_yaw[i, 0]
            self.Cn_alpha[i] = self.coeffs_yaw[i, 1]
            self.Cn_beta[i] = self.coeffs_yaw[i, 2]
            self.Cn_pbar[i] = self.coeffs_yaw[i, 3]
            self.Cn_qbar[i] = self.coeffs_yaw[i, 4]
            self.Cn_rbar[i] = self.coeffs_yaw[i, 5]
            self.Cn_da[i] = self.coeffs_yaw[i, 6]
            self.Cn_de[i] = self.coeffs_yaw[i, 7]

            # Construct A matrix for linear least-squares problem specific to drag
            # model.
            A = np.zeros((N, 9))
            A[:, 0] = 1.
            A[:, 1] = b_lift
            A[:, 2] = np.square(b_lift)
            A[:, 3] = np.square(b_side)
            A[:, 4] = pbar
            A[:, 5] = qbar
            A[:, 6] = rbar
            A[:, 7] = d_a
            A[:, 8] = d_e

            # Solve for linear least-squares drag coefficients.
            lst_sq_res = np.linalg.lstsq(A, b_drag, rcond=None)
            self.coeffs_drag[i, :] = lst_sq_res[0]

            # Save relevant coefficients for chosen linear aerodynamic model.
            self.CD0[i] = self.coeffs_drag[i, 0]
            self.CD1[i] = self.coeffs_drag[i, 1]
            self.CD2[i] = self.coeffs_drag[i, 2]
            self.CD3[i] = self.coeffs_drag[i, 3]
            self.CD_pbar[i] = self.coeffs_drag[i, 4]
            self.CD_qbar[i] = self.coeffs_drag[i, 5]
            self.CD_rbar[i] = self.coeffs_drag[i, 6]
            self.CD_da[i] = self.coeffs_drag[i, 7]
            self.CD_de[i] = self.coeffs_drag[i, 8]

    def _bire_fits_dB(self, d_B):
        d_B = np.deg2rad(d_B)
        # LIFT FITS
        self._bire_lift_fits_dB(d_B)
        # DRAG FITS
        self._bire_drag_fits_dB(d_B)
        # SIDE FITS
        self._bire_side_fits_dB(d_B)
        # ROLL FITS
        self._bire_roll_fits_dB(d_B)
        # PITCH FITS
        self._bire_pitch_fits_dB(d_B)
        # YAW FITS
        self._bire_yaw_fits_dB(d_B)


    def _bire_lift_fits_dB(self, d_B):
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL0
        A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
        self.CL0_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL_alpha
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CLalpha_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL_beta
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CLbeta_dB = self._generate_sine_function(A, w, phi, z)
        self.CLpbar_dB = self._generate_poly_function([np.mean(self.CL_pbar)])
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL_qbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.001, 1.42, 3.69])[0]
        self.CLqbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL_rbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CLrbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL_da
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CLda_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CL_de
        A, w, phi, z = optimize.leastsq(sine_fit, [0.7, 0.005, 1.42, 0.4])[0]
        self.CLde_dB = self._generate_sine_function(A, w, phi, z)

    def _bire_drag_fits_dB(self, d_B):
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD0
        A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.3, -0.66, 0.03])[0]
        self.CD0_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: (x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) +
                              x[2]*np.power(d_B, 1) + x[3] - self.CD1)
        a = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
        self.CD1_dB = self._generate_poly_function(a)
        sine_fit = lambda x: (x[0]*np.power(d_B, 4) + x[1]*np.power(d_B, 3) +
                              x[2]*np.power(d_B, 2) + x[3]*np.power(d_B, 1) +
                              x[4] - self.CD2)
        a = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087, 0.1])[0]
        self.CD2_dB = self._generate_poly_function(a)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD3
        A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.1, -0.66, 0.087])[0]
        self.CD3_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD_pbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.02, -0.66, 0.087])[0]
        self.CDpbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD_qbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.05, -0.66, 0.087])[0]
        self.CDqbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD_rbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.02, -0.66, 0.087])[0]
        self.CDrbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD_da
        A, w, phi, z = optimize.leastsq(sine_fit, [0.06, 0.01, -0.66, 0.087])[0]
        self.CDda_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CD_de
        A, w, phi, z = optimize.leastsq(sine_fit, [0.03, 0.02, -0.66, 0.087])[0]
        self.CDde_dB = self._generate_sine_function(A, w, phi, z)

    def _bire_side_fits_dB(self, d_B):
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS0
        A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
        self.CS0_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_alpha
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CSalpha_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_beta
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.01, 1.42, 3.69])[0]
        self.CSbeta_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_pbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CSpbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_qbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.05, 1.42, 3.69])[0]
        self.CSqbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_rbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
        self.CSrbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_da
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.005, 1.42, 3.69])[0]
        self.CSda_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.CS_de
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.CSde_dB = self._generate_sine_function(A, w, phi, z)

    def _bire_roll_fits_dB(self, d_B):
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cell0
        A, w, phi, z = optimize.leastsq(sine_fit, [0.01, 4., -0.66, 0.])[0]
        self.Cell0_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cell_alpha
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.1, 1.42, 3.69])[0]
        self.Cellalpha_dB = self._generate_sine_function(A, w, phi, z)
        self.Cellbeta_dB = self._generate_poly_function([np.mean(self.Cell_beta)])
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cell_pbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.Cellpbar_dB = self._generate_sine_function(A, w, phi, z)
        self.Cellqbar_dB = self._generate_poly_function([np.mean(self.Cell_qbar)])
        self.Cellrbar_dB = self._generate_poly_function([np.mean(self.Cell_rbar)])
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cell_da
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.Cellda_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: (x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) +
                              x[2]*np.power(d_B, 1) + x[3] - self.Cell_de)
        a = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
        self.Cellde_dB = self._generate_poly_function(a)

    def _bire_pitch_fits_dB(self, d_B):
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm0
        A, w, phi, z = optimize.leastsq(sine_fit, [0.02, 0.1, -0.66, 0.02])[0]
        self.Cm0_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_alpha
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.05, 0.0, 0.02])[0]
        self.Cmalpha_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_beta
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.05, 1.42, 0.])[0]
        self.Cmbeta_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_pbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.Cmpbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_qbar
        A, w, phi, z = optimize.leastsq(sine_fit, [-3., 0.04, 1.42, 3.69])[0]
        self.Cmqbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_rbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
        self.Cmrbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_da
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.Cmda_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cm_de
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.01, 1.42, -0.4])[0]
        self.Cmde_dB = self._generate_sine_function(A, w, phi, z)

    def _bire_yaw_fits_dB(self, d_B):
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cn0
        A, w, phi, z = optimize.leastsq(sine_fit, [0.001, 0.5, -0.66, 0.0])[0]
        self.Cn0_dB = self._generate_poly_function([np.mean(self.Cn0)])
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cn_alpha
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.03, 1.42, 3.69])[0]
        self.Cnalpha_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cn_beta
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.03, 1.42, 3.69])[0]
        self.Cnbeta_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: (x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) +
                              x[2]*np.power(d_B, 1) + x[3] - self.Cn_pbar)
        a = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
        self.Cnpbar_dB = self._generate_poly_function(a)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cn_qbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.02, 1.42, 3.69])[0]
        self.Cnqbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cn_rbar
        A, w, phi, z = optimize.leastsq(sine_fit, [0.5, 0.08, 1.42, 0.25])[0]
        self.Cnrbar_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: x[0]*np.sin(x[1]*d_B+x[2]) + x[3] - self.Cn_da
        A, w, phi, z = optimize.leastsq(sine_fit, [0.15, 0.04, 1.42, 3.69])[0]
        self.Cnda_dB = self._generate_sine_function(A, w, phi, z)
        sine_fit = lambda x: (x[0]*np.power(d_B, 3) + x[1]*np.power(d_B, 2) +
                              x[2]*np.power(d_B, 1) + x[3] - self.Cn_de)
        a = optimize.leastsq(sine_fit, [0.01, 0.04, -0.66, 0.087])[0]
        self.Cnde_dB = self._generate_poly_function(a)



    def _generate_sine_function(self, A, w, phi, z):
        def dummy(x):
            return A*np.sin(w*x + phi) + z
        return dummy

    def _generate_poly_function(self, a):
        def dummy(x):
            summ = a[-1]
            for i in range(len(a) - 1):
                summ += a[i]*np.power(x, len(a) - 1 - i)
            return summ
        return dummy


    def _w2b_linear_coeffs_conversion(self):
        """Converts linear least-squares coefficients to the body-fixed frame.

        Using the linear least-squares force coefficients calculated in the
        method `_linear_fits`, which are calculated in the wind frame, the
        force coefficients are converted to the body-fixed frame and functions
        are created for each of the body-fixed force and moment coefficients.
        The aerodynamic moment coefficients are already in the body-fixed
        frame and therefore need only be assigned to a function. See the Notes
        section for the order of the function parameters.

        Attributes
        ----------
        c_l : function
            Function taking aerodynamic parameters and returning the lift
            coefficient calculated using the linear model.

        c_s : function
            Function taking aerodynamic parameters and returning the side force
            coefficient calculated using the linear model.

        c_d : function
            Function taking aerodynamic parameters and returning the drag
            coefficient calculated using the linear model.

        c_x : function
            Function taking aerodynamic parameters and returning the force
            coefficient in the x-direction using the linear model.

        c_y : function
            Function taking aerodynamic parameters and returning the force
            coefficient in the y-direction using the linear model.

        c_z : function
            Function taking aerodynamic parameters and returning the force
            coefficient in the z-direction using the linear model.

        c_ell : function
            Function taking aerodynamic parameters and returning the rolling
            moment coefficient using the linear model.

        c_m : function
            Function taking aerodynamic parameters and returning the pitching
            moment coefficient using the linear model.

        c_n : function
            Function taking aerodynamic parameters and returning the yawing
            moment coefficient using the linear model.

        Notes
        -----
        The parameters for all coefficients are in the order:
            alpha, beta, d_a, d_e, d_r, p, q, r

        """
        # Note that the inputs `x` are all in radians.
        if self.model == "bire":
            self.c_l = lambda x: (self.CL0_dB(x[4]) +
                                  self.CLalpha_dB(x[4])*x[0] +
                                  self.CLbeta_dB(x[4])*x[1] +
                                  self.CLpbar_dB(x[4])*x[5] +
                                  self.CLqbar_dB(x[4])*x[6] +
                                  self.CLrbar_dB(x[4])*x[7] +
                                  self.CLda_dB(x[4])*x[2] +
                                  self.CLde_dB(x[4])*x[3])
            self.c_s = lambda x: (self.CS0_dB(x[4]) +
                                  self.CSalpha_dB(x[4])*x[0] +
                                  self.CSbeta_dB(x[4])*x[1] +
                                  self.CSpbar_dB(x[4])*x[5] +
                                  self.CSqbar_dB(x[4])*x[6] +
                                  self.CSrbar_dB(x[4])*x[7] +
                                  self.CSda_dB(x[4])*x[2] +
                                  self.CSde_dB(x[4])*x[3])
            self.c_d = lambda x: (self.CD0_dB(x[4]) +
                                  self.CD1_dB(x[4])*self.c_l(x) +
                                  self.CD2_dB(x[4])*self.c_l(x)**2 +
                                  self.CD3_dB(x[4])*self.c_s(x)**2 +
                                  self.CDpbar_dB(x[4])*x[5] +
                                  self.CDqbar_dB(x[4])*x[6] +
                                  self.CDrbar_dB(x[4])*x[7] +
                                  self.CDda_dB(x[4])*x[2] +
                                  self.CDde_dB(x[4])*x[3])
            self.c_x = lambda x: [-(self.c_d(x)*np.cos(x[0])*np.cos(x[1]) +
                                    self.c_s(x)*np.cos(x[0])*np.sin(x[1]) -
                                    self.c_l(x)*np.sin(x[0]))]
            self.c_y = lambda x: [(self.c_s(x)*np.cos(x[1]) -
                                   self.c_d(x)*np.sin(x[1]))]
            self.c_z = lambda x: [-(self.c_d(x)*np.sin(x[0])*np.cos(x[1]) +
                                    self.c_s(x)*np.sin(x[0])*np.sin(x[1]) +
                                    self.c_l(x)*np.cos(x[0]))]
            self.c_ell = lambda x: [(self.Cell0_dB(x[4]) +
                                     self.Cellalpha_dB(x[4])*x[0] +
                                     self.Cellbeta_dB(x[4])*x[1] +
                                     self.Cellpbar_dB(x[4])*x[5] +
                                     self.Cellqbar_dB(x[4])*x[6] +
                                     self.Cellrbar_dB(x[4])*x[7] +
                                     self.Cellda_dB(x[4])*x[2] +
                                     self.Cellde_dB(x[4])*x[3])]
            self.c_m = lambda x: [(self.Cm0_dB(x[4]) +
                                   self.Cmalpha_dB(x[4])*x[0] +
                                   self.Cmbeta_dB(x[4])*x[1] +
                                   self.Cmpbar_dB(x[4])*x[5] +
                                   self.Cmqbar_dB(x[4])*x[6] +
                                   self.Cmrbar_dB(x[4])*x[7] +
                                   self.Cmda_dB(x[4])*x[2] +
                                   self.Cmde_dB(x[4])*x[3])]
            self.c_n = lambda x: [(self.Cn0_dB(x[4]) +
                                   self.Cnalpha_dB(x[4])*x[0] +
                                   self.Cnbeta_dB(x[4])*x[1] +
                                   self.Cnpbar_dB(x[4])*x[5] +
                                   self.Cnqbar_dB(x[4])*x[6] +
                                   self.Cnrbar_dB(x[4])*x[7] +
                                   self.Cnda_dB(x[4])*x[2] +
                                   self.Cnde_dB(x[4])*x[3])]
        else:
            self.c_l = lambda x: (self.CL0 +
                                  self.CL_alpha*x[0] +
                                  self.CL_qbar*x[6] +
                                  self.CL_de*x[3])
            self.c_s = lambda x: (self.CS_beta*x[1] +
                                  self.CS_pbar*x[5] +
                                  self.CS_rbar*x[7] +
                                  self.CS_da*x[2] +
                                  self.CS_dr*x[4])
            self.c_d = lambda x: (self.CD0 + self.CD1*self.c_l(x) +
                                  self.CD2*self.c_l(x)**2 +
                                  self.CD3*self.c_s(x)**2 +
                                  self.CD_qbar*x[6] +
                                  self.CD_de*x[3])
            self.c_x = lambda x: [-(self.c_d(x)*np.cos(x[0])*np.cos(x[1]) +
                                    self.c_s(x)*np.cos(x[0])*np.sin(x[1]) -
                                    self.c_l(x)*np.sin(x[0]))]
            self.c_y = lambda x: [(self.c_s(x)*np.cos(x[1]) -
                                   self.c_d(x)*np.sin(x[1]))]
            self.c_z = lambda x: [-(self.c_d(x)*np.sin(x[0])*np.cos(x[1]) +
                                    self.c_s(x)*np.sin(x[0])*np.sin(x[1]) +
                                    self.c_l(x)*np.cos(x[0]))]
            self.c_ell = lambda x: [(self.Cell_beta*x[1] +
                                     self.Cell_pbar*x[5] +
                                     self.Cell_rbar*x[7] +
                                     self.Cell_da*x[2] +
                                     self.Cell_dr*x[4])]
            self.c_m = lambda x: [(self.Cm0 +
                                   self.Cm_alpha*x[0] +
                                   self.Cm_qbar*x[6] +
                                   self.Cm_de*x[3])]
            self.c_n = lambda x: [(self.Cn_beta*x[1] +
                                   self.Cn_pbar*x[5] +
                                   self.Cn_rbar*x[7] +
                                   self.Cn_da*x[2] +
                                   self.Cn_dr*x[4])]

    def _data_conversion(self, aero_data):
        """Converts aerodynamic data to be used in linear least-squares.

        Aerodynamic data is passed in and the force coefficients are converted
        from the body-fixed frame to the wind frame so that they are
        appropriate for the linear aerodynamic model. In addition, the
        body-fixed rotation rates are non-dimensionalized for the least-squares
        fits.

        See Also
        ----------
        bf2w_drag : Takes the x-, y-, and z-force coefficients in the
                    body-fixed frame as well as the angle of attack and
                    sideslip angle (in radians) and returns the drag
                    coefficient in the wind frame.

        bf2w_side : Takes the x-, y-, and z-force coefficients in the
                    body-fixed frame as well as the angle of attack and
                    sideslip angle (in radians) and returns the side force
                    coefficient in the wind frame.

        bf2w_lift : Takes the x-, y-, and z-force coefficients in the
                    body-fixed frame as well as the angle of attack and
                    sideslip angle (in radians) and returns the lift
                    coefficient in the wind frame.

        Returns
        -------
        aero_data : array_like
            The converted aerodynamic data to be used for the linear least-
            squares fit.

        """
        b2w_params = np.copy(np.array([aero_data[:, 8],
                                       aero_data[:, 9],
                                       aero_data[:, 10],
                                       np.deg2rad(aero_data[:, 0]),
                                       np.deg2rad(aero_data[:, 1])])).T
        aero_data[:, 8] = [bf2w_drag(*args,) for args in b2w_params]
        aero_data[:, 9] = [bf2w_side(*args,) for args in b2w_params]
        aero_data[:, 10] = [bf2w_lift(*args,) for args in b2w_params]
        aero_data[:, 5] *= self.lat_ref_len/(2.*self.V)
        aero_data[:, 6] *= self.lon_ref_len/(2.*self.V)
        aero_data[:, 7] *= self.lat_ref_len/(2.*self.V)
        return aero_data

    def trim(self, climb_angle=0., shss=False, **kwargs):
        """Trims the aircraft.

        With the appropriate aerodynamic database imported and aerodynamic
        model chosen, the aircraft is trimmed at a specific climb angle and
        load factor. There is an option to trim the aircraft in a steady
        coordinated turn (of which steady-level flight is the default) or in
        steady-heading sideslip, which is useful for analyzing crosswind
        landings.

        Attributes
        ----------
        trim_state : array_like
            Contains the final trim state of the aircraft in the following
            order: angle of attack, sideslip (or bank angle), aileron
            deflection, elevator deflection, rudder deflection, and throttle
            setting.

        Methods
        -------
        _trim_shss(trim_params, climb_angle, **kwargs)
            Used to trim the aircraft in steady-heading sideslip given a climb
            angle as well as a sideslip angle or bank angle.

        _trim_sct(trim_params, climb_angle, load_factor)
            Used to trim the aircraft in a steady coordinated turn given a
            climb angle and load factor.


        See Also
        --------
        scipy.optimize.fsolve : Non-linear root finder to find the roots of a
                                given function.

        """
        trim_params = np.zeros(6)
        if shss:
            args = (np.deg2rad(climb_angle))
            self.trim_state = optimize.fsolve(self._trim_shss, trim_params,
                                                args=args)
        else:
            self.climb_angle = np.deg2rad(climb_angle)
            self.trim_iter = 1
            # self.trim_state = optimize.minimize(self._trim_sct, trim_params,
            #                                     method='BFGS',
            #                                     args=args,
            #                                     options={'gtol':1e-12}).x
            self.trim_state = [0.]*9
            self.trim_error = [100.]*9
            self.bank_angle = np.deg2rad(kwargs.get("bank_angle", 0.))
            while np.abs(self.trim_error[0]) >= 1e-12:
                trim_state_new = self._trim_sct(self.trim_state)
                self.trim_error = [new - old for new, old in zip(trim_state_new, self.trim_state)]
                self.trim_state = trim_state_new
                self.trim_iter += 1
            # trim_solution = optimize.fsolve(self._trim_sct_fsolve, np.zeros(6))
            # print(trim_solution[:-1]*180/np.pi, trim_solution[-1])
            # print(self._6dof_fm(trim_solution, self._vel_comp(trim_solution[0], trim_solution[1])))


    def _trim_shss(self, trim_params, climb_angle, **kwargs):
        """Function used to find a trim solution for steady-heading sideslip.

        This function, when used in a non-linear solver, gives the parameters
        needed to trim an aircraft in steady-heading sideslip given a climb
        angle in addition to either a sideslip angle or bank angle. The
        trim parameters are, in order: angle of attack, sideslip (or bank
        angle), aileron deflection, elevator deflection, rudder deflection,
        and throttle setting.


        Parameters
        ----------
        trim_params : array_like
            The trim parameters that will be optimized to return zero forces
            and moments in the 6 DOF governing equations.
        climb_angle : float
            The climb angle at which the aircraft will be trimmed.
        bank_angle : float, optional
            The bank angle at which the steady-heading sideslip case will be
            trimmed.
        sideslip_angle : float, optional
            The sideslip angle at which the steady-heading sideslip case will
            be trimmed.

        Returns
        -------
        total_fm : list
            The total forces and moments acting on the aircraft. These will be


        """
        pass

    def _trim_sct(self, trim_state):
        alpha, beta, d_a, d_e, d_r, p, q, r, tau = trim_state
        pbar = p*self.lat_ref_len/(2.*self.V)
        qbar = q*self.lon_ref_len/(2.*self.V)
        rbar = r*self.lat_ref_len/(2.*self.V)
        # Calculate body-fixed velocities
        vel_bf = self._vel_comp(alpha, beta)
        u, v, w = vel_bf
        # Calculate elevation angle
        self.elevation_angle = self._calc_elevation_angle(vel_bf,
                                                          [self.climb_angle,
                                                           self.bank_angle])
        if self.trim_iter == 1:
            CD = 0.
            CS = 0.
        else:
            CD = self.c_d([alpha, beta, d_a, d_e, d_r, pbar, qbar, rbar])
            CS = self.c_s([alpha, beta, d_a, d_e, d_r, pbar, qbar, rbar])
        # self.bank_angle = self._calc_bank_angle(alpha, beta,
        #                                         vel_bf, CS, CD, self.load_factor)


        # Calculate rotation rates
        s_elev = np.sin(self.elevation_angle)
        c_elev = np.cos(self.elevation_angle)
        s_bank = np.sin(self.bank_angle)
        c_bank = np.cos(self.bank_angle)
        sc_angles = [s_elev, c_elev, s_bank, c_bank]
        euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
        p, q, r = self._sct_rot_rates(sc_angles, euler_transform, u, w)
        self.rot_rates = [p, q, r]
        pbar = p*self.lat_ref_len/(2.*self.V)
        qbar = q*self.lon_ref_len/(2.*self.V)
        rbar = r*self.lat_ref_len/(2.*self.V)
        CL = self.c_l([alpha, beta, d_a, d_e, d_r, pbar, qbar, rbar])
        CS = self.c_s([alpha, beta, d_a, d_e, d_r, pbar, qbar, rbar])
        CD = self.c_d([alpha, beta, d_a, d_e, d_r, pbar, qbar, rbar])
        if self.model == "bire":
            Cm0 = self.Cm0_dB(d_r)
            Cma = self.Cmalpha_dB(d_r)
            Cmb = self.Cmbeta_dB(d_r)
            Cmp = self.Cmpbar_dB(d_r)
            Cmq = self.Cmqbar_dB(d_r)
            Cmr = self.Cmrbar_dB(d_r)
            Cmda = self.Cmda_dB(d_r)
            Cmde = self.Cmde_dB(d_r)
            Cn0 = self.Cn0_dB(d_r)
            Cna = self.Cnalpha_dB(d_r)
            Cnb = self.Cnbeta_dB(d_r)
            Cnp = self.Cnpbar_dB(d_r)
            Cnq = self.Cnqbar_dB(d_r)
            Cnr = self.Cnrbar_dB(d_r)
            Cnda = self.Cnda_dB(d_r)
            Cnde = self.Cnde_dB(d_r)
            CS0 = self.CS0_dB(d_r)
            CSa = self.CSalpha_dB(d_r)
            CSb = self.CSbeta_dB(d_r)
            CSp = self.CSpbar_dB(d_r)
            CSq = self.CSqbar_dB(d_r)
            CSr = self.CSrbar_dB(d_r)
            CSda = self.CSda_dB(d_r)
            CSde = self.CSde_dB(d_r)
            CL0 = self.CL0_dB(d_r)
            CLa = self.CLalpha_dB(d_r)
            CLb = self.CLbeta_dB(d_r)
            CLp = self.CLpbar_dB(d_r)
            CLq = self.CLqbar_dB(d_r)
            CLr = self.CLrbar_dB(d_r)
            CLda = self.CLda_dB(d_r)
            CLde = self.CLde_dB(d_r)
            Cell0 = self.Cell0_dB(d_r)
            Cella = self.Cellalpha_dB(d_r)
            Cellb = self.Cellbeta_dB(d_r)
            Cellp = self.Cellpbar_dB(d_r)
            Cellq = self.Cellqbar_dB(d_r)
            Cellr = self.Cellrbar_dB(d_r)
            Cellda = self.Cellda_dB(d_r)
            Cellde = self.Cellde_dB(d_r)
            d_e = ((self.hx*r -
                    self.hz*p -
                    (self.Izz - self.Ixx)*p*r -
                    self.Ixz*(r**2 - p**2) -
                    self.Ixy*q*r +
                    self.Iyz*p*q)/(self.nondim_coeff*self.lon_ref_len) -
                   Cm0 - Cma*alpha - Cmb*beta - Cmp*pbar - Cmq*qbar -
                   Cmr*rbar - Cmda*d_a)/Cmde
            d_r = ((self.hy*p -
                    self.hx*q -
                    (self.Ixx - self.Iyy)*p*q -
                    self.Ixy*(p**2 - q**2) -
                    self.Iyz*p*r +
                    self.Ixz*q*r)/(self.nondim_coeff*self.lat_ref_len) -
                   Cn0 - Cna*alpha - Cnb*beta - Cnp*pbar - Cnq*qbar -
                   Cnr*rbar - Cnda*d_a - Cnde*d_e + d_r)
            tau = (self.W*s_elev - self.W*(r*v - q*w)/self.g -
                   self.nondim_coeff*(CL*np.sin(alpha) -
                                      CS*np.cos(alpha)*np.sin(beta) -
                                      CD*np.cos(alpha)*np.cos(beta)))
            beta = (CD*np.sin(beta)/np.cos(beta) -
                    CS0 - CSa*alpha - CSp*pbar - CSq*qbar - CSr*rbar -
                    CSda*d_a - CSde*d_e)/CSb
            alpha = ((self.CW*c_bank*c_elev +
                      self.CW*(q*u - p*v)/self.g -
                      CS*np.sin(alpha)*np.sin(beta) -
                      CD*np.sin(alpha)*np.cos(beta))/np.cos(alpha) -
                     CL0 - CLb*beta - CLp*pbar - CLq*qbar - CLr*rbar -
                     CLda*d_a - CLde*d_e)/CLa
            d_a = ((self.hz*q -
                    self.hy*r -
                    (self.Iyy - self.Izz)*q*r -
                    self.Iyz*(q**2 - r**2) -
                    self.Ixz*p*q +
                    self.Ixy*p*r)/(self.nondim_coeff*self.lat_ref_len) -
                   Cell0 - Cella*alpha - Cellb*beta - Cellp*pbar - Cellq*qbar -
                   Cellr*rbar - Cellde*d_e)/Cellda
        else:
            tau = (self.W*s_elev - self.W*(r*v - q*w)/self.g -
                   self.nondim_coeff*(CL*np.sin(alpha) -
                                      CS*np.cos(alpha)*np.sin(beta) -
                                      CD*np.cos(alpha)*np.cos(beta)))
            beta = (CD*np.sin(beta)/np.cos(beta) -
                    self.CS_pbar*pbar -
                    self.CS_rbar*rbar -
                    self.CS_da*d_a -
                    self.CS_dr*d_r)/self.CS_beta
            alpha = ((self.CW*c_bank*c_elev +
                      self.CW*(q*u - p*v)/self.g -
                      CS*np.sin(alpha)*np.sin(beta) -
                      CD*np.sin(alpha)*np.cos(beta))/np.cos(alpha) -
                     self.CL0 -
                     self.CL_qbar*qbar -
                     self.CL_de*d_e)/self.CL_alpha
            d_a = ((self.hz*q -
                    self.hy*r -
                    (self.Iyy - self.Izz)*q*r -
                    self.Iyz*(q**2 - r**2) -
                    self.Ixz*p*q +
                    self.Ixy*p*r)/(self.nondim_coeff*self.lat_ref_len) -
                   self.Cell_beta*beta -
                   self.Cell_pbar*pbar -
                   self.Cell_rbar*rbar -
                   self.Cell_dr*d_r)/self.Cell_da
            d_e = ((self.hx*r -
                    self.hz*p -
                    (self.Izz - self.Ixx)*p*r -
                    self.Ixz*(r**2 - p**2) -
                    self.Ixy*q*r +
                    self.Iyz*p*q)/(self.nondim_coeff*self.lon_ref_len) -
                   self.Cm0 -
                   self.Cm_alpha*alpha -
                   self.Cm_qbar*qbar)/self.Cm_de
            d_r = ((self.hy*p -
                    self.hx*q -
                    (self.Ixx - self.Iyy)*p*q -
                    self.Ixy*(p**2 - q**2) -
                    self.Iyz*p*r +
                    self.Ixz*q*r)/(self.nondim_coeff*self.lat_ref_len) -
                   self.Cn_beta*beta -
                   self.Cn_pbar*pbar -
                   self.Cn_rbar*rbar -
                   self.Cn_da*d_a)/self.Cn_dr
        # Use aero model to find aero angles, throttle setting, and deflections
        CL = self.c_l([alpha, beta, d_a, d_e, d_r, pbar, qbar, rbar])
        load_factor = CL/self.CW
        output = [*vel_bf, np.rad2deg(self.bank_angle),
                  np.rad2deg(self.elevation_angle), *np.rad2deg(self.rot_rates),
                  *np.rad2deg([alpha, beta, d_a, d_e, d_r]), tau, load_factor]
        output_names = ['u [ft/s]', 'v [ft/s]', 'w [ft/s]', '\u03C6 [deg]',
                        '\u03B8 [deg]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]',
                        '\u03B1 [deg]', '\u03B2 [deg]', '\u03B4_a [deg]',
                        '\u03B4_e [deg]', '\u03B4_r [deg]', '\u03C4 [lbs]',
                        'n_a']
        if self.model == "bire":
            output_names[12] = '\u03B4_B [deg]'
        print("Iteration:" + str(self.trim_iter) + "\n")
        for name, value in zip(output_names, output):
            print(f'{name:10} ==> {value:3.8}')
        return [alpha, beta, d_a, d_e, d_r, p, q, r, tau]
        # return np.linalg.norm(total_forces + total_moments)

    def _trim_sct_fsolve(self, params):
        alpha, beta, d_a, d_e, d_B, tau = params
        # Calculate body-fixed velocities
        vel_bf = self._vel_comp(alpha, beta)
        u, v, w = vel_bf
        # Calculate elevation angle
        self.elevation_angle = self._calc_elevation_angle(vel_bf,
                                                          [self.climb_angle,
                                                           self.bank_angle])
        fm = self._6dof_fm(params, vel_bf)
        return fm




    def _6dof_fm(self, params, vel_bf, **kwargs):
        f_xt = params[-1]
        shss_flag = kwargs.get("shss", False)
        u, v, w = vel_bf
        s_elev = np.sin(self.elevation_angle)
        c_elev = np.cos(self.elevation_angle)
        s_bank = np.sin(self.bank_angle)
        c_bank = np.cos(self.bank_angle)
        euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
        if shss_flag:
            rot_rates = [0., 0., 0.]
        else:
            rot_rates = self._sct_rot_rates([s_elev, c_elev, s_bank, c_bank],
                                            euler_transform, u, w)
        pbar = rot_rates[0]*self.lat_ref_len/(2.*self.V)
        qbar = rot_rates[1]*self.lon_ref_len/(2.*self.V)
        rbar = rot_rates[2]*self.lat_ref_len/(2.*self.V)
        params = list(list(params[:]) + [pbar, qbar, rbar])
        aero_forces, aero_moments = self._dimensionalize_aero_fm(params, f_xt)
        weight_forces = self._6dof_weight_forces(euler_transform)
        rot_forces = self._6dof_coriolis_forces(rot_rates, vel_bf)
        total_forces = [x + y + z for x, y, z in zip(aero_forces,
                                                     weight_forces,
                                                     rot_forces)]
        rotor_moments = self._6dof_rotor_moments(rot_rates)
        corr_moments = self._6dof_coriolis_moments(rot_rates)
        total_moments = [x + y + z for x, y, z in zip(aero_moments,
                                                      rotor_moments,
                                                      corr_moments)]
        self.rot_rates = rot_rates
        return list(total_forces) + list(total_moments)

    def _6dof_weight_forces(self, euler_transform):
        weight_forces = [x*self.W for x in euler_transform]
        return weight_forces

    def _6dof_coriolis_forces(self, rot_rates, vel_bf):
        p, q, r = rot_rates
        u, v, w = vel_bf
        mass = self.W/self.g
        coriolis_forces = [mass*x for x in [r*v - q*w, p*w - r*u, q*u - p*v]]
        return coriolis_forces

    def _6dof_rotor_moments(self, rot_rates):
        rotor_moments = list(np.matmul(self.ang_mom_mat, np.array(rot_rates)))
        return rotor_moments

    def _6dof_coriolis_moments(self, rot_rates):
        p, q, r = rot_rates
        cor_moment_1 = ((self.Iyy - self.Izz)*q*r + self.Iyz*(q**2 - r**2) +
                        self.Ixz*p*q - self.Ixy*p*r)
        cor_moment_2 = ((self.Izz - self.Ixx)*p*r + self.Ixz*(r**2 - p**2) +
                        self.Ixy*q*r - self.Iyz*p*q)
        cor_moment_3 = ((self.Ixx - self.Iyy)*p*q + self.Ixy*(p**2 - q**2) +
                        self.Iyz*p*r - self.Ixz*q*r)
        return [cor_moment_1, cor_moment_2, cor_moment_3]


    def _dimensionalize_aero_fm(self, params, f_xt):
        f_x = self.c_x(params)[0]*self.nondim_coeff + f_xt
        f_y = self.c_y(params)[0]*self.nondim_coeff
        f_z = self.c_z(params)[0]*self.nondim_coeff
        m_x = self.c_ell(params)[0]*self.nondim_coeff*self.lat_ref_len
        m_y = self.c_m(params)[0]*self.nondim_coeff*self.lon_ref_len
        m_z = self.c_n(params)[0]*self.nondim_coeff*self.lat_ref_len
        return [f_x, f_y, f_z], [m_x, m_y, m_z]

    def _vel_comp(self, alpha, beta):
        s_alpha = np.sin(alpha)
        c_alpha = np.cos(alpha)
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        wind_to_body = [c_alpha*c_beta, s_beta, s_alpha*c_beta]
        [u, v, w] = [x*self.V for x in wind_to_body]
        return [u, v, w]

    def _sct_rot_rates(self, sc_angles, euler_transform, u, w):
        s_elev, c_elev = sc_angles[:2]
        s_bank, c_bank = sc_angles[2:]
        constant_coeff = self.g*s_bank*c_elev/(u*c_elev*c_bank + w*s_elev)
        p, q, r = [x*constant_coeff for x in euler_transform]
        return [p, q, r]

    def _calc_bank_angle(self, alpha, beta, vel_bf, CS, CD, load_a):
        elevation_angle = self.elevation_angle
        s_alpha = np.sin(alpha)
        c_alpha = np.cos(alpha)
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        s_elev = np.sin(elevation_angle)
        c_elev = np.cos(elevation_angle)
        u, v, w = vel_bf
        A = s_elev**2*c_elev**2*v**2 + (w*s_elev*c_elev - (CS*s_alpha*s_beta + CD*s_alpha*c_beta)*u*c_elev/self.CW - load_a*c_alpha*u*c_elev)**2
        B = 2.*s_elev*c_elev*v*(c_elev**2*u - (CS*s_alpha*s_beta + CD*s_alpha*c_beta)*w*s_elev/self.CW - load_a*c_alpha*w*s_elev)
        C = (c_elev**2*u - (CS*s_alpha*s_beta + CD*s_alpha*c_beta)*w*s_elev/self.CW - load_a*c_alpha*w*s_elev)**2 - (w*s_elev*c_elev - (CS*s_alpha*s_beta + CD*s_alpha*c_beta)*u*c_elev/self.CW - load_a*c_alpha*u*c_elev)**2
        bank_angle_plus = np.arcsin((-B + np.sqrt(B**2 - 4.*A*C))/(2.*A))
        bank_angle_minus = np.arcsin((-B - np.sqrt(B**2 - 4.*A*C))/(2.*A))
        test_plus = np.allclose(load_a*c_alpha,
                                self._test_bank(CS, CD, alpha, beta, bank_angle_plus,
                                                elevation_angle, vel_bf),
                                atol=1e-12)
        test_minus = np.allclose(load_a*c_alpha,
                                 self._test_bank(CS, CD, alpha, beta, bank_angle_minus,
                                                 elevation_angle, vel_bf),
                                 atol=1e-12)
        if test_plus:
            bank_angle = bank_angle_plus
        else:
            assert test_minus
            bank_angle = bank_angle_minus
        return bank_angle

    def _test_bank(self, CS, CD, alpha, beta, bank, elevation, vel_bf):
        u, v, w = vel_bf
        c_bank = np.cos(bank)
        s_bank = np.sin(bank)
        c_elev = np.cos(elevation)
        s_elev = np.sin(elevation)
        s_alpha = np.sin(alpha)
        c_alpha = np.cos(alpha)
        s_beta = np.sin(beta)
        c_beta = np.cos(beta)
        C1 = c_bank*c_elev
        C2 = (CS*s_alpha*s_beta + CD*s_alpha*c_beta)/self.CW
        C3_num = s_bank**2*c_elev**2*u + s_bank*s_elev*c_elev*v
        C3_denom = u*c_elev*c_bank + w*s_elev
        C3 = C3_num/C3_denom
        return C1 - C2 + C2

    def _calc_elevation_angle(self, vel_bf, orientation):
        climb_angle, bank_angle = orientation
        s_climb = np.sin(climb_angle)
        s_bank = np.sin(bank_angle)
        c_bank = np.cos(bank_angle)
        u, v, w = vel_bf
        A = (u**2/self.V**2 + s_bank**2*v**2/self.V**2 +
             2.*s_bank*c_bank*v*w/self.V**2 + c_bank**2*w**2/self.V**2)
        B = -s_climb*u/self.V
        C = (s_climb**2 - s_bank**2*v**2/self.V**2 -
             2.*s_bank*c_bank*v*w/self.V**2 - c_bank**2*w**2/self.V**2)
        elevation_angle_plus = np.arcsin((-B + np.sqrt(B**2 - A*C))/(A))
        elevation_angle_minus = np.arcsin((-B - np.sqrt(B**2 - A*C))/(A))
        orientation_plus = [elevation_angle_plus, bank_angle]
        orientation_minus = [elevation_angle_minus, bank_angle]
        if np.allclose(-s_climb,
                       self._test_elevation(vel_bf, orientation_plus),
                       atol=1e-12):
            elevation_angle = elevation_angle_plus
        else:
            assert np.allclose(-s_climb,
                               self._test_elevation(vel_bf,
                                                    orientation_minus),
                               atol=1e-12)
            elevation_angle = elevation_angle_minus
        return elevation_angle

    def _test_elevation(self, vel_bf, orientation):
        elevation_angle, bank_angle = orientation
        s_elev = np.sin(elevation_angle)
        c_elev = np.cos(elevation_angle)
        s_bank = np.sin(bank_angle)
        c_bank = np.cos(bank_angle)
        u, v, w = vel_bf
        return (-s_elev*u/self.V + s_bank*c_elev*v/self.V +
                c_bank*c_elev*w/self.V)

#Baseline
# trim_case = TrimCase(222., 0.0023084)
# trim_case.import_aero_data("./misc/T1_final_rad.csv", model='linear')
# trim_case.trim(climb_angle=0., bank_angle=0.0)

#BIRE
trim_case = TrimCase(222., 0.0023084)
trim_case.import_aero_data("./misc/TODatabase_BIRE_body344.csv", model='bire', d_B = np.linspace(-90, 90, 44))
trim_case.trim(climb_angle=0., bank_angle=0.0)
