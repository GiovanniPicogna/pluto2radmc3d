import struct
import numpy as np

# -------------------------------------------------------------------
# reading particles
# -------------------------------------------------------------------
class Particles():
    # G. Picogna
    """
    Particle class, it stores all the parameters and scalar data
    for particles
    Input: ns='' [int] -> output number 
           directory='' [string] -> where filename is
    """

    def __init__(self, ns, directory=''):

        def NStepStr(ns):
            nsstr = str(ns)
            while len(nsstr) < 4:
                nsstr = '0'+nsstr
            return nsstr

        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'

        nstepstr = NStepStr(ns)
        fname = directory+"part_data."+nstepstr+".dbl"
        fdata = open(fname, "rb")
        fmt1 = "<"+"i"
        fmt2 = "<"+"d"

        nb1 = struct.calcsize(fmt1)
        nb2 = struct.calcsize(fmt2)

        NOP = struct.unpack(fmt1, fdata.read(nb1))[0]
        NDIM = 3  # hard coded.
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
        self.radius = [particles_radius[i]
                       for j in range(100000) for i in range(10)]