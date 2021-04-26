import struct
import numpy as np
import sys
import time


# -------------------------------------------------------------------
# reading particles
# -------------------------------------------------------------------
class Particles:
    # G. Picogna
    """
    Particle class, it stores all the parameters and scalar data
    for particles
    Input: ns='' [int] -> output number 
           directory='' [string] -> where filename is
    """

    def __init__(self, ns, directory=''):

        def number_step_str():
            nsstr = str(ns)
            while len(nsstr) < 4:
                nsstr = '0'+nsstr
            return nsstr

        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'

        nstepstr = number_step_str()
        fname = directory+"part_data."+nstepstr+".dbl"
        fdata = open(fname, "rb")
        fmt1 = "<"+"i"
        fmt2 = "<"+"d"

        nb1 = struct.calcsize(fmt1)
        nb2 = struct.calcsize(fmt2)

        nop = struct.unpack(fmt1, fdata.read(nb1))[0]
        particles_step = struct.unpack(fmt1, fdata.read(nb1))[0]
        particles_time = struct.unpack(fmt2, fdata.read(nb2))[0]

        dtype = np.dtype([
            ("pid", np.int32),
            ("pcell_x", np.int32),
            ("pcell_y", np.int32),
            ("pcell_z", np.int32),
            ("pos_x", np.float64),
            ("pos_y", np.float64),
            ("pos_z", np.float64),
            ("vel_x", np.float64),
            ("vel_y", np.float64),
            ("vel_z", np.float64),
            ("tstop", np.float64)
        ])

        particle_dictionary = np.fromfile(fdata, dtype=dtype)
        fdata.close()

        sorter = particle_dictionary['pid'].argsort()
        data_dictionary = particle_dictionary[sorter]

        particles_radius = np.array(
            [0.0001, 0.001, 0.01, 0.1, 0.3, 1., 3., 10., 100., 1000.])

        self.nop = nop
        self.particles_time = particles_time
        self.particles_step = particles_step
        self.pid = data_dictionary['pid']
        self.pcell_x = data_dictionary['pcell_x']
        self.pcell_y = data_dictionary['pcell_y']
        self.pcell_z = data_dictionary['pcell_z']
        self.pos_x = data_dictionary['pos_x']
        self.pos_y = data_dictionary['pos_y']
        self.pos_z = data_dictionary['pos_z']
        self.vel_x = data_dictionary['vel_x']
        self.vel_y = data_dictionary['vel_y']
        self.vel_z = data_dictionary['vel_z']
        self.tstop = data_dictionary['tstop']
        self.radius = [particles_radius[i] for j in range(100000) for i in range(10)]


def populate_dust_bins(density, particle_data, nbin, bins, particles_per_bin_per_cell, particles_per_bin,
                       tstop_per_bin):

    for index_part in range(particle_data.nop):

        # skip particles that are bigger than the selected bin sizes
        if particle_data.radius[index_part] >= bins[nbin]:
            continue

        # radial index of the cell where the particle is
        index_radial = np.where(density.redge > particle_data.pos_x[index_part])[0][0] - 1
        if index_radial < 0 or index_radial >= (density.nrad + 2):
            print('In recalc_density step: radial index = ', index_radial)
            sys.exit()

        # polar index of the cell where the particles is
        index_polar = np.where(density.tedge > particle_data.pos_y[index_part])[0][0] - 1
        if index_polar < 0 or index_polar >= (density.ncol + 2):
            print('In recalc_density step: polar index = ', index_polar)
            sys.exit()

        # find out which bin particle belongs to
        index_bin = np.where(bins > particle_data.radius[index_part])[0][0] - 1

        particles_per_bin[index_bin] += 1
        tstop_per_bin[index_bin] += particle_data.tstop[index_part]

        # azimuthal index of the cell where the particle is
        for index_azimuthal in range(density.nsec):

            index = index_bin * density.nsec * density.ncol * density.nrad + index_azimuthal * density.ncol * \
                    density.nrad + index_polar * density.nrad + index_radial

            particles_per_bin_per_cell[index] += 1

    for ibin in range(nbin):
        if particles_per_bin[ibin] == 0:
            particles_per_bin[ibin] = 1
        tstop_per_bin[ibin] /= particles_per_bin[ibin]
        print(str(particles_per_bin[ibin]) + ' grains between ' +
              str(bins[ibin]) + ' and ' + str(bins[ibin + 1]) + ' meters')
