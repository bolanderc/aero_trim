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

from scipy.interpolate import RegularGridInterpolator as rgi
import scipy.optimize as optimize


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
        self.trim_solution = np.zeros(6)


    def import_aero_data(
            self, file_name, num_dimensions, num_pts_per_dimension,
            dimension_lims, **kwargs):
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
        >>> trim_case.import_aero_data(test_data_file_name, num_dimensions,
                                       num_pts_per_dimension, dimension_lims)

        """
        save_numpy = kwargs.get("save_numpy", False)
        save_sorted = kwargs.get("save_sorted", False)
        header_titles = kwargs.get("header_titles",
                                   ['AOA', 'Beta', 'd_e', 'd_a',
                                    'd_r', 'p', 'q', 'r'])
        database_params = kwargs.get("database_params", False)
        if not database_params:
            try:
                int(num_pts_per_dimension)
            except TypeError:
                print("Please specify `database_params` if " +
                      "`num_pts_per_dimension` is not an integer.")
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

    def trim_shss(self, trim_params, **kwargs):
        elevation_angle = kwargs.get("elevation_angle", 0.)
        alpha = trim_params[0]
        if kwargs.get("phi") is None:
            beta = kwargs.get("beta", trim_params[1])
        else:
            bank_angle = kwargs.get("bank_angle", trim_params[1])
        sym_angle = trim_params[2]
        antsim_angle = trim_params[3]
        rudder_angle = trim_params[4]
        f_xt = trim_params[5]
        vel_bf = self._vel_comp(alpha, beta)

    def _6dof_fm(self, params, f_xt, orientation, vel_bf, **kwargs):
        shss_flag = kwargs.get("shss", False)
        elevation_angle, bank_angle = orientation
        u, v, w = vel_bf
        s_elev = np.sin(np.deg2rad(elevation_angle))
        c_elev = np.cos(np.deg2rad(elevation_angle))
        s_bank = np.sin(np.deg2rad(bank_angle))
        c_bank = np.cos(np.deg2rad(bank_angle))
        euler_transform = [-s_elev, s_bank*c_elev, c_bank*c_elev]
        if shss_flag:
            rot_rates = [0., 0., 0.]
        else:
            rot_rates = self._sct_rot_rates([s_elev, c_elev,
                                             s_bank, c_bank],
                                            euler_transform,
                                            u, w)
        params = list(params[:] + rot_rates)
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
        return total_forces, total_moments

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
        s_alpha = np.sin(np.deg2rad(alpha))
        c_alpha = np.cos(np.deg2rad(alpha))
        s_beta = np.sin(np.deg2rad(beta))
        c_beta = np.cos(np.deg2rad(beta))
        wind_to_body = [c_alpha*c_beta, c_alpha*s_beta, s_alpha*c_beta]
        const_coeff = self.V/np.sqrt(1 - s_alpha**2*s_beta**2)
        [u, v, w] = [x*const_coeff for x in wind_to_body]
        return [u, v, w]

    def _sct_rot_rates(self, sc_angles, euler_transform, u, w):
        s_elev, c_elev = sc_angles[:2]
        s_bank, c_bank = sc_angles[2:]
        constant_coeff = self.g*s_bank*c_elev/(u*c_elev*c_bank + w*s_elev)
        p, q, r = [x*constant_coeff for x in euler_transform]
        return [p, q, r]





