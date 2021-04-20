import numpy as np
try:
    import h5py as h5
    hasH5 = True
except ImportError:
    hasH5 = False

class pload(object):
    # based on A. Mignone routine
    def __init__(self, parameters, datatype=None, level=0, x1range=None, x2range=None, x3range=None):
        """Loads the data.
        **Inputs**:
          datatype -- Datatype (default = 'double')

        **Outputs**:
          pyPLUTO pload object whose keys are arrays of data values.
        """

        self.NStep = parameters['on']
        self.Dt = 1.e-4

        self.n1 = 0
        self.n2 = 0
        self.n3 = 0

        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.dx1 = []
        self.dx2 = []
        self.dx3 = []

        self.x1range = x1range
        self.x2range = x2range
        self.x3range = x3range

        self.NStepStr = str(self.NStep)
        while len(self.NStepStr) < 4:
            self.NStepStr = '0'+self.NStepStr

        if datatype is None:
            datatype = "double"
        self.datatype = datatype

        self.level = level
        self.wdir = parameters['directory']

        Data_dictionary = self.ReadDataFile(self.NStepStr,parameters)
        for keys in Data_dictionary:
            object.__setattr__(self, keys, Data_dictionary.get(keys))

    def ReadTimeInfo(self, timefile):
        """ Read time info from the outfiles.
        **Inputs**:
           timefile -- name of the out file which has timing information. 
        """

        fh5 = h5.File(timefile, 'r')
        self.SimTime = fh5.attrs.get('time')
        self.Dt = 1.e-2
        fh5.close()

    def keys(self, f):
        return [key for key in f.keys()]

    def ReadVarFile(self, varfile, parameters):
        """ Read variable names from the outfiles.
        **Inputs**:
          varfile -- name of the out file which has variable information. 
        """
        fh5 = h5.File(varfile, 'r')
        self.filetype = 'single_file'
        self.endianess = '>'  # not used with AMR, kept for consistency
        self.vars = []
        num = str(parameters['on'])
        for iv in range(len(fh5['Timestep_'+num+'/vars'].keys())):
            self.vars.append(self.keys(fh5['Timestep_'+num+'/vars'])[iv])

        fh5.close()

    def ReadGridFile(self, gridfile):
        """ Read grid values from the grid.out file.
        *Inputs**:    
          gridfile -- name of the grid.out file which has information about the grid. 
        """

        xL = []
        xR = []
        nmax = []
        gfp = open(gridfile, "r")
        for i in gfp.readlines():
            if len(i.split()) == 1:
                try:
                    int(i.split()[0])
                    nmax.append(int(i.split()[0]))
                except:
                    pass

            if len(i.split()) == 3:
                try:
                    int(i.split()[0])
                    xL.append(float(i.split()[1]))
                    xR.append(float(i.split()[2]))
                except:
                    if (i.split()[1] == 'GEOMETRY:'):
                        self.geometry = i.split()[2]
                    pass

        self.n1, self.n2, self.n3 = nmax
        n1 = self.n1
        n1p2 = self.n1 + self.n2
        n1p2p3 = self.n1 + self.n2 + self.n3
        self.x1 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1)])
        self.dx1 = np.asarray([(xR[i]-xL[i]) for i in range(n1)])
        self.x2 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1, n1p2)])
        self.dx2 = np.asarray([(xR[i]-xL[i]) for i in range(n1, n1p2)])
        self.x3 = np.asarray([0.5*(xL[i]+xR[i]) for i in range(n1p2, n1p2p3)])
        self.dx3 = np.asarray([(xR[i]-xL[i]) for i in range(n1p2, n1p2p3)])

        # Stores the total number of points in '_tot' variable in case only
        # a portion of the domain is loaded. Redefine the x and dx arrays
        # to match the requested ranges
        self.n1_tot = self.n1
        self.n2_tot = self.n2
        self.n3_tot = self.n3
        if (self.x1range != None):
            self.n1_tot = self.n1
            self.irange = range(
                abs(self.x1-self.x1range[0]).argmin(), abs(self.x1-self.x1range[1]).argmin()+1)
            self.n1 = len(self.irange)
            self.x1 = self.x1[self.irange]
            self.dx1 = self.dx1[self.irange]
        else:
            self.irange = range(self.n1)
        if (self.x2range != None):
            self.n2_tot = self.n2
            self.jrange = range(
                abs(self.x2-self.x2range[0]).argmin(), abs(self.x2-self.x2range[1]).argmin()+1)
            self.n2 = len(self.jrange)
            self.x2 = self.x2[self.jrange]
            self.dx2 = self.dx2[self.jrange]
        else:
            self.jrange = range(self.n2)
        if (self.x3range != None):
            self.n3_tot = self.n3
            self.krange = range(
                abs(self.x3-self.x3range[0]).argmin(), abs(self.x3-self.x3range[1]).argmin()+1)
            self.n3 = len(self.krange)
            self.x3 = self.x3[self.krange]
            self.dx3 = self.dx3[self.krange]
        else:
            self.krange = range(self.n3)
        self.Slice = (self.x1range != None) or (
            self.x2range != None) or (self.x3range != None)

        # Create the xr arrays containing the edges positions
        # Useful for pcolormesh which should use those
        self.x1r = np.zeros(len(self.x1)+1)
        self.x1r[1:] = self.x1 + self.dx1/2.0
        self.x1r[0] = self.x1r[1]-self.dx1[0]
        self.x2r = np.zeros(len(self.x2)+1)
        self.x2r[1:] = self.x2 + self.dx2/2.0
        self.x2r[0] = self.x2r[1]-self.dx2[0]
        self.x3r = np.zeros(len(self.x3)+1)
        self.x3r[1:] = self.x3 + self.dx3/2.0
        self.x3r[0] = self.x3r[1]-self.dx3[0]

        prodn = self.n1*self.n2*self.n3
        if prodn == self.n1:
            self.nshp = (self.n1)
        elif prodn == self.n1*self.n2:
            self.nshp = (self.n2, self.n1)
        else:
            self.nshp = (self.n3, self.n2, self.n1)

    def getGrid(self, fp, dim):
        x = fp['node_coords'][dim][:]
        x = x.astype('float64')
        return x

    def getGridCell(self, fp, dim):
        x = fp['cell_coords'][dim][:]
        x = x.astype('float64')
        return x

    def getVar(self, fp, step, var):
        returnData = (fp["Timestep_"+str(step)+"/vars"][var][:])
        return returnData

    def DataScanHDF5(self, fp, myvars, parameters):
        """ Scans HDF5 data files in PLUTO. 

        **Inputs**:

          fp     -- Data file pointer\n
          myvars -- Names of the variables to read\n

        **Output**:

          Dictionary consisting of variable names as keys and its values. 

        """

        # Read the grid information
        dim = np.size(fp['cell_coords'].keys())

        x1r = self.getGrid(fp, 'X')
        x1 = self.getGridCell(fp, 'X')
        nx = x1.shape[2]
        dx1 = x1r[1:]-x1r[:-1]
        x2r = 0
        x2 = 0
        dx2 = 0
        ny = 0
        x3r = 0
        x3 = 0
        dx3 = 0
        nz = 0
        dt = 0
        if(dim > 1):
            x2r = self.getGrid(fp, 'Y')
            x2 = self.getGridCell(fp, 'Y')
            ny = x2.shape[1]
            dx2 = x2r[1:]-x2r[:-1]
        if(dim > 2):
            x3r = self.getGrid(fp, 'Z')
            x3 = self.getGridCell(fp, 'Z')
            nz = x3.shape[0]
            dx3 = x3r[1:]-x3r[:-1]

        NewGridDict = dict([('n1', nx), ('n2', ny), ('n3', nz),
                            ('x1', x1), ('x2', x2), ('x3', x3),
                            ('x1r', x1r), ('x2r', x2r), ('x3r', x3r),
                            ('dx1', dx1), ('dx2', dx2), ('dx3', dx3),
                            ('Dt', dt)])

        # Variables table
        nvar = len(myvars)
        vars = np.zeros((nx, ny, nz, nvar))

        h5vardict = {}
        for iv in range(nvar):
            h5vardict[myvars[iv]] = self.getVar(fp, parameters['on'], myvars[iv])
            #h5vardict[myvars[iv]] = vars[:,:,:,iv].squeeze()

        OutDict = dict(NewGridDict)
        OutDict.update(h5vardict)
        return OutDict

    def ReadSingleFile(self, datafilename, myvars, parameters, n1, n2, n3, endian, dtype, ddict):
        """Reads a single data file, data.****.dtype.

        **Inputs**: 

          datafilename -- Data file name\n
          myvars -- List of variable names to be read\n
          n1 -- No. of points in X1 direction\n
          n2 -- No. of points in X2 direction\n
          n3 -- No. of points in X3 direction\n
          endian -- Endianess of the data\n
          dtype -- datatype\n
          ddict -- Dictionary containing Grid and Time Information
          which is updated

        **Output**:

          Updated Dictionary consisting of variable names as keys and its values.
        """

        fp = h5.File(datafilename, 'r')

        print("Reading Data file : %s" % datafilename)

        h5d = self.DataScanHDF5(fp, myvars, parameters)
        ddict.update(h5d)

        fp.close()

    def ReadDataFile(self, num, parameters):
        """Reads the data file generated from PLUTO code.

        **Inputs**:

        num -- Data file number in form of an Integer.

        **Outputs**:

        Dictionary that contains all information about Grid, Time and 
        variables.

        """
        gridfile = self.wdir+"grid.out"
        dtype = 'd'
        dataext = '.dbl.h5'
        nstr = num
        varfile = self.wdir+"data."+nstr+dataext

        self.ReadVarFile(varfile, parameters)
        self.ReadGridFile(gridfile)
        self.ReadTimeInfo(varfile)
        nstr = num
        if self.endianess == 'big':
            endian = ">"
        else:
            endian = "<"

        D = [('NStep', self.NStep), ('SimTime', self.SimTime), ('Dt', self.Dt),
             ('n1', self.n1), ('n2', self.n2), ('n3', self.n3),
             ('x1', self.x1), ('x2', self.x2), ('x3', self.x3),
             ('dx1', self.dx1), ('dx2', self.dx2), ('dx3', self.dx3),
             ('endianess', self.endianess), ('datatype', self.datatype),
             ('filetype', self.filetype)]
        ddict = dict(D)

        datafilename = self.wdir+"data."+nstr+dataext
        self.ReadSingleFile(datafilename, self.vars, parameters, self.n1, self.n2,
                            self.n3, endian, dtype, ddict)

        return ddict