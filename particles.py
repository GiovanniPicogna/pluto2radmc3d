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

        def NStepStr():
            nsstr = str(ns)
            while len(nsstr) < 4:
                nsstr = '0'+nsstr
            return nsstr

        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'

        nstepstr = NStepStr()
        fname = directory+"part_data."+nstepstr+".dbl"
        fdata = open(fname, "rb")
        fmt1 = "<"+"i"
        fmt2 = "<"+"d"

        nb1 = struct.calcsize(fmt1)
        nb2 = struct.calcsize(fmt2)

        NOP = struct.unpack(fmt1, fdata.read(nb1))[0]
        P_Step = struct.unpack(fmt1, fdata.read(nb1))[0]
        P_Time = struct.unpack(fmt2, fdata.read(nb2))[0]

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

        P_DataDict = np.fromfile(fdata, dtype=dtype)
        fdata.close()

        sorter = P_DataDict['pid'].argsort()
        DataDict = P_DataDict[sorter]

        particles_radius = np.array(
            [0.0001, 0.001, 0.01, 0.1, 0.3, 1., 3., 10., 100., 1000.])

        self.NOP = NOP
        self.P_Time = P_Time
        self.P_Step = P_Step
        self.pid = DataDict['pid']
        self.pcell_x = DataDict['pcell_x']
        self.pcell_y = DataDict['pcell_y']
        self.pcell_z = DataDict['pcell_z']
        self.pos_x = DataDict['pos_x']
        self.pos_y = DataDict['pos_y']
        self.pos_z = DataDict['pos_z']
        self.vel_x = DataDict['vel_x']
        self.vel_y = DataDict['vel_y']
        self.vel_z = DataDict['vel_z']
        self.tstop = DataDict['tstop']
        self.radius = [particles_radius[i] for j in range(100000) for i in range(10)]

def populate_dust_bins(density, particle_data, nbin, bins, particles_per_bin_per_cell, particles_per_bin, tstop_per_bin):

    for m in range(particle_data.NOP):

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
        for bin in range(nbin):
            if particle_data.radius[m] < bins[bin + 1]:
                ibin = bin
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