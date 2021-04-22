import struct
import numpy as np
import sys


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


def populate_dust_bins(density, particle_data, nbin, bins, dust_cube, particles_per_bin, tstop_per_bin):

    particles_per_bin_per_cell = np.zeros((density.nsec * density.ncol * density.nrad * nbin))
    for m in range(particle_data.nop):

        # radial index of the cell where the particle is
        i = int(np.log(particle_data.pos_x[m] / density.redge.min()) / np.log(
            density.redge.max() / density.redge.min()) * density.nrad)
        if i < 0 or i >= (density.nrad + 2):
            print('In recalc_density step: radial index = ', i)
            sys.exit()

        # polar index of the cell where the particles is
        j = density.ncol - 1 - int(np.log((np.pi - particle_data.pos_y[m]) / (np.pi - density.tedge.max())) / np.log(
            (np.pi - density.tedge.min()) / (np.pi - density.tedge.max())) * density.ncol)
        if j < 0 or j >= (density.ncol + 2):
            print('In recalc_density step: polar index = ', j)
            sys.exit()

        # azimuthal index of the cell where the particle is
        # (general expression since grid spacing in azimuth is always arithmetic)
        k = particle_data.pcell_z[m]
        if k < 0 or k >= (density.nsec + 2):
            print('In recalc_density step: azimuthal index = ', k)
            sys.exit()

        # find out which bin particle belongs to
        ibin = 0
        for n in range(nbin):
            if particle_data.radius[m] < bins[n + 1]:
                ibin = n
                break

        # skip particles that are bigger than the selected bin sizes
        if particle_data.radius[m] >= bins[nbin + 1]:
            continue

        k = ibin * density.nsec * density.ncol * density.nrad + k * density.ncol * density.nrad + j * density.nrad + i
        particles_per_bin_per_cell[k] += 1
        particles_per_bin[ibin] += 1
        tstop_per_bin[ibin] += particle_data.tstop[m]

    for ibin in range(nbin):
        if particles_per_bin[ibin] == 0:
            particles_per_bin[ibin] = 1
        tstop_per_bin[ibin] /= particles_per_bin[ibin]
        print(str(particles_per_bin[ibin]) + ' grains between ' +
              str(bins[ibin]) + ' and ' + str(bins[ibin + 1]) + ' meters')

    dust_cube = particles_per_bin_per_cell.reshape((nbin, density.nsec, density.ncol, density.nrad))

    # free RAM memory
    del particles_per_bin_per_cell
