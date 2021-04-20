import sys
import numpy as np
import subprocess
from pluto import *

# -------------------------------------------------------------------
# building mesh arrays for phi, theta, r (x, y, z)
# -------------------------------------------------------------------
class Mesh():
    # based on P. Benitez Llambay routine
    """
    Mesh class, keeps all mesh data.
    Input: directory [string] -> place where domain files are
    """

    def __init__(self, directory=""):

        # -----
        # grid
        # -----
        print('RADMC grid:')
        print('number of grid cells in spherical radius = ', self.nrad)
        print('number of grid cells in colatitude       = ', self.ncol)
        print('number of grid cells in azimuth          = ', self.nsec)

        # -----
        # spherical radius
        # -----
        # CB: we build the array of spherical radii used by RADMC3D from the set of spherical radii
        # adopted in the PLUTO simulation

        try:
            # radial interfaces of grid cells
            domain_rad = np.loadtxt(
                fname=directory+"grid.out", skiprows=11, usecols=1, max_rows=self.nrad)
            last = np.loadtxt(
                fname=directory+"grid.out", skiprows=11+self.nrad-1, usecols=2, max_rows=1)
            domain_rad = np.append(domain_rad, last)
        except IOError:
            print('IOError')

        self.redge = domain_rad                              # r-edge
        self.rmed = 2.0*(domain_rad[1:]*domain_rad[1:]*domain_rad[1:] - domain_rad[:-1]*domain_rad[:-1] *
                         domain_rad[:-1]) / (domain_rad[1:]*domain_rad[1:] - domain_rad[:-1]*domain_rad[:-1]) / 3.0  # r-center
        self.dr = self.redge[1:] - self.redge[:-1]

        # -----
        # colatitude
        # -----
        #
        # CB: note that we can't do mirror symmetry in RADMC3D when scattering_mode_max = 2
        # ie anisotropic scattering is assumed. We need to define the grid's colatitude on
        # both sides about the disc midplane (defined where theta = pi/2)
        #
        # thmin is set as pi/2+atan(zmax_over_H*h) with h the gas aspect ratio
        # zmax_over_H = z_max_grid / pressure scale height, value set in params.dat

        try:
            domain_th = np.loadtxt(fname=directory+"grid.out", skiprows=10+1+self.nrad+1,
                                   usecols=1, max_rows=self.ncol)  # radial interfaces of grid cells
            last = np.loadtxt(
                fname=directory+"grid.out", skiprows=10+1+self.nrad+self.ncol, usecols=2, max_rows=1)
            domain_th = np.append(domain_th, last)
        except IOError:
            print('IOError')

        self.tedge = domain_th                    # colatitude of cell faces
        # colatitude of cell centers
        self.tmed = 0.5*(domain_th[:-1] + domain_th[1:])
        self.dth = self.tedge[1:] - self.tedge[:-1]

        # define number of cells in vertical direction for arrays in
        # cylindrical coordinates
        self.nver = 200  # seems large enough
        # define an array for vertical altitude across the midplane
        #zbuf = -self.rmed.max()*np.cos(self.tmed)
        self.zmed = np.outer(self.rmed,np.cos(self.tmed))

        # -----
        # azimuth
        # -----
        self.pedge = np.linspace(0., 2.*np.pi, self.nsec+1)  # phi-edge
        self.pmed = 0.5*(self.pedge[:-1] + self.pedge[1:])  # phi-center
        self.dp = self.pedge[1:] - self.pedge[:-1]


# -------------------------------------------------------------------
# reading fields
# can be density, energy, velocities, etc
# -------------------------------------------------------------------
class Field(Mesh):
    # based on P. Benitez Llambay routine
    """
    Field class, it stores all the mesh, parameters and scalar data
    for a scalar field.
    Input: field [string] -> filename of the field
           staggered='c' [string] -> staggered direction of the field.
                                      Possible values: 'x', 'y', 'xy', 'yx'
           directory='' [string] -> where filename is
           dtype='float64' (numpy dtype) -> 'float64', 'float32',
                                             depends if FARGO_OPT+=-DFLOAT is activated
    """

    def __init__(self, field, parameters, staggered='c', directory='', dtype='float64'):

        D = pload(parameters=parameters)

        nrad = np.shape(D.rho)[2]
        ncol = np.shape(D.rho)[1]
        nsec = np.shape(D.rho)[0]
        nsec = 600

        nrad = int(nrad)
        nsec = int(nsec)
        self.nrad = nrad
        self.nsec = nsec
        # colatitude (ncol is a global variable)
        self.ncol = ncol

        Mesh.__init__(self, directory=parameters['directory'])    # all Mesh attributes inside Field

        # get units via definitions.h file
        command = 'awk " /^#define  UNIT_LENGTH/ " '+directory+'definitions.h'
        # check which version of python we're using
        if sys.version_info[0] < 3:   # python 2.X
            buf = subprocess.check_output(command, shell=True)
        else:                         # python 3.X
            buf = subprocess.getoutput(command)
        num = buf.split()[2]
        ulen = num.split('*')[0][1:]
        self.culength = float(ulen)*1.5e11  # from au to meters
        command = 'awk " /^#define  UNIT_DENSITY/ " '+directory+'definitions.h'
        # check which version of python we're using
        if sys.version_info[0] < 3:   # python 2.X
            buf = subprocess.check_output(command, shell=True)
        else:                         # python 3.X
            buf = subprocess.getoutput(command)
        num = buf.split()[2]
        ulen = num.split('*')[0][1:]
        self.cumass = float(ulen)*2e30  # from Msol to kg
        # unit of temperature = mean molecular weight * 8.0841643e-15 * M / L;
        self.cutemp = 2.35 * 8.0841643e-15 * self.cumass / self.culength

        if parameters['override_units'] == 'Yes':
            self.cumass = new_unit_mass
            self.culength = new_unit_length
            # Deduce new units of time and temperature:
            # T = sqrt( pow(L,3.) / 6.673e-11 / M )
            # U = mmw * 8.0841643e-15 * M / L;
            self.cutime = np.sqrt(self.culength**3 / 6.673e-11 / self.cumass)
            self.cutemp = 2.35 * 8.0841643e-15 * self.cumass / self.culength
            print('### NEW UNITS SPECIFIED: ###')
            print('new unit of length [m] = ', self.culength)
            print('new unit of mass [kg]  = ', self.cumass)
            print('new unit of time [s] = ', self.cutime)
            print('new unit of temperature [K] = ', self.cutemp)

        # now, staggering:
        if staggered.count('r') > 0:
            self.r = self.redge[:-1]  # do not dump last element
        else:
            self.r = self.rmed

        #self.data = D.rho  # scalar data is here.
        self.data = np.zeros((nsec,ncol,nrad))
        for ns in range(nsec):
            self.data[ns] = D.rho
        self.x3 = self.pmed
        self.x3r = self.pedge
        self.dx3 = self.pedge[1:] - self.pedge[:-1]