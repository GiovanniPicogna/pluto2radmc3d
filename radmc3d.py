# ---------------------
# define RT model class
# ---------------------
class RTmodel():
    def __init__(self, distance=140, label='', npix=256, Lambda=800,
                 incl=30.0, posang=0.0, phi=0.0,
                 line='12co', imolspec=1, iline=3, linenlam=80, vkms=0, widthkms=4,):

        # disk parameters
        self.distance = distance * pc
        self.label = label
        # RT pars
        self.Lambda = Lambda
        self.line = gasspecies
        self.npix = npix
        self.incl = incl
        self.posang = posang
        self.phi = float(phi)
        # line emission pars
        self.imolspec = imolspec
        self.iline = iline
        self.vkms = vkms
        self.widthkms = widthkms
        self.linenlam = linenlam

# -------------------------
# functions calling RADMC3D
# -------------------------
def write_radmc3d_script(parameters):

    # RT in the dust continuum
    if parameters['RTdust_or_gas'] == 'dust':
        command = 'radmc3d image lambda '+str(parameters['wavelength']*1e3)+' npix '+str(
            parameters['nbpixels'])+' incl '+str(parameters['inclination'])+' posang '+str(
                parameters['posangle']+90.0)+' phi '+str(parameters['phiangle'])
        if parameters['plot_tau'] == 'Yes':
            command = 'radmc3d image tracetau lambda '+str(parameters['wavelength']*1e3)+' npix '+str(
                parameters['nbpixels'])+' incl '+str(parameters['inclination'])+' posang '+str(
                    parameters['posangle']+90.0)+' phi '+str(parameters['phiangle'])
        if parameters['polarized_scat'] == 'Yes':
            command = command+' stokes'

    # RT in gas lines
    if parameters['RTdust_or_gas'] == 'gas':
        if parameters['widthkms'] == 0.0:
            command = 'radmc3d image iline '+str(parameters['iline'])+' vkms '+str(parameters['vkms'])+' npix '+str(
                parameters['nbpixels'])+' incl '+str(parameters['inclination'])+' posang '+str(
                    parameters['posangle']+90.0)+' phi '+str(parameters['phiangle'])
        else:
            command = 'radmc3d image iline '+str(iline)+' widthkms '+str(widthkms)+' linenlam '+str(linenlam)+' npix '+str(
                nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)
            if plot_tau == 'Yes':
                command = 'radmc3d image tracetau iline '+str(iline)+' widthkms '+str(widthkms)+' linenlam '+str(
                    linenlam)+' npix '+str(nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)
                #command='radmc3d tausurf 1.0 iline '+str(iline)+' widthkms '+str(widthkms)+' linenlam '+str(linenlam)+' npix '+str(nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)

    # optional: second-order ray tracing
    if parameters['secondorder'] == 'Yes':
        command = command+' secondorder'

    # write execution script
    print(command)
    SCRIPT = open('script_radmc', 'w')
    if parameters['Tdust_eq_Thydro'] == 'No':
        SCRIPT.write('radmc3d mctherm; '+command)
    else:
        SCRIPT.write(command)
    SCRIPT.close()
    os.system('chmod a+x script_radmc')


# -------------------------
# Convert result of RADMC3D calculation into fits file
# -------------------------
def exportfits(M,parameters):
    # name of .fits file where data is output
    # Note that M.label will disentangle dust and gas line calculations
    if parameters['plot_tau'] == 'No':
        if parameters['brightness_temp'] == 'Yes':
            image = 'imageTb_'
        else:
            image = 'image_'
        if parameters['RTdust_or_gas'] == 'gas':
            # no need for lambda in file's name for gas RT calculations...
            outfile = image+str(M.label)+'_i'+str(parameters['inclination']) + \
                '_phi'+str(M.phi)+'_PA'+str(parameters['posangle'])
        if parameters['RTdust_or_gas'] == 'dust':
            outfile = image+str(M.label)+'_lbda'+str(parameters['wavelength'])+'_i' + \
                str(parameters['inclination'])+'_phi'+str(M.phi)+'_PA'+str(parameters['posangle'])
    else:
        if parameters['RTdust_or_gas'] == 'gas':
            # no need for lambda in file's name for gas RT calculations...
            outfile = 'tau_'+str(M.label)+'_i'+str(parameters['inclination']) + \
                '_phi'+str(M.phi)+'_PA'+str(parameters['posangle'])
        if parameters['RTdust_or_gas'] == 'dust':
            outfile = 'tau_'+str(M.label)+'_lbda'+str(parameters['wavelength'])+'_i' + \
                str(parameters['inclination'])+'_phi'+str(M.phi)+'_PA'+str(parameters['posangle'])
    if parameters['secondorder'] == 'Yes':
        outfile = outfile+'_so'

    if parameters['add_noise'] == 'Yes':
        outfileall = outfile+'_wn'+str(parameters['noise_dev_std'])+'_all.fits'
    else:
        outfileall = outfile+'_all.fits'
    outfile = outfile+'.fits'

    LOG = open('fluxlog.txt', 'a')
    LOG.write(outfile+"\n")

    # input file produced by radmc3D
    infile = 'image.out'
    # read header info:
    f = open(infile, 'r')
    iformat = int(f.readline())
    # nb of pixels
    im_nx, im_ny = tuple(np.array(f.readline().split(), dtype=int))
    # nb of wavelengths, can be different from one for multi-color images of gas emission
    nlam = int(f.readline())
    # pixel size in each direction in cm
    pixsize_x, pixsize_y = tuple(np.array(f.readline().split(), dtype=float))
    # read wavelength in microns
    lbda = np.zeros(nlam)
    for i in range(nlam):
        lbda[i] = float(f.readline())
    # convert lbda in velocity (km/s)
    if nlam > 1:
        lbda0 = lbda[nlam//2]  # implicitly gas RT
        vel_range = c * 1e-5 * (lbda-lbda0)/lbda0  # in km/s (c is in cm/s!)
        print('range of wavelengths = ', lbda)
        print('range of velocities = ', vel_range)
        dv = vel_range[1]-vel_range[0]
    else:
        if parameters['RTdust_or_gas'] == 'gas':
            lbda0 = lbda[0]
        else:
            lbda0 = parameters['wavelength']*1e3  # in microns
    f.readline()               # empty line

    # calculate physical scales
    distance = M.distance          # distance is in cm here
    pixsize_x_deg = 180.0*pixsize_x / distance / pi
    pixsize_y_deg = 180.0*pixsize_y / distance / pi

    # surface of a pixel in radian squared
    pixsurf_ster = pixsize_x_deg*pixsize_y_deg * (pi/180.)**2

    # 1 Jansky converted in cgs x pixel surface in cm^2 (same
    # conversion whether RT in dust or gas lines)
    fluxfactor = 1.e23 * pixsurf_ster
    print('fluxfactor = ', fluxfactor)

    # beam area in pixel^2
    mycdelt = pixsize_x/(M.distance/pc)/au
    beam = (np.pi/(4.*np.log(2.)))*parameters['bmaj']*parameters['bmin']/(mycdelt**2.)
    print('beam = ', beam)
    # stdev lengths in pixel
    stdev_x = (parameters['bmaj']/(2.*np.sqrt(2.*np.log(2.)))) / mycdelt
    stdev_y = (parameters['bmin']/(2.*np.sqrt(2.*np.log(2.)))) / mycdelt

    # ---------------
    # load image data
    # ---------------
    if parameters['polarized_scat'] == 'No':
        if parameters['nlam'] == 1:  # dust continuum RT calculations (implicitly)
            images = np.loadtxt(infile, skiprows=6)
            im = images.reshape(im_ny, im_nx)
            naxis = 2
        if parameters['nlam'] > 1:   # gas RT calculations (implicitly)
            intensity_in_each_channel = np.zeros((parameters['nlam'], im_ny, im_nx))
            moment0 = np.zeros((im_ny, im_nx))
            moment1 = np.zeros((im_ny, im_nx))
            naxis = 2
            images = np.loadtxt(infile, skiprows=5+nlam)
            for i in range(parameters['nlam']):
                im_v = images[i*im_ny*im_nx:(i+1)*im_ny*im_nx]
                im = im_v.reshape(im_ny, im_nx)
                # sometimes the intensity has a value at the origin
                # that is unrealistically large, in particular when
                # the velocity relative to the systemic one is
                # large. We put the specific intensity to zero at the
                # origin, as it should be in our disc model!
                im[im_ny//2, im_nx//2] = 0.0
                # ---
                if (parameters['add_noise'] == 'Yes'):
                    # noise standard deviation in Jy per pixel
                    noise_dev_std_Jy_per_pixel = (
                        parameters['noise_dev_std']/fluxfactor) / np.sqrt(0.5*beam)  # 1D
                    if i == 0:
                        print('noise_dev_std_Jy_per_pixel = ',
                              noise_dev_std_Jy_per_pixel)
                    # noise array
                    noise_array = np.random.normal(
                        0.0, noise_dev_std_Jy_per_pixel, size=im_ny*im_nx)
                    noise_array = noise_array.reshape(im_ny, im_nx)
                    im += noise_array
                # ---
                # keep track of beam-convolved specific intensity in each channel map
                if parameters['plot_tau'] == 'No':
                    # avoid division by zeros w/o noise...
                    smooth = Gauss_filter(
                        im, stdev_x, stdev_y, bpaangle, Plot=False) + 1e-20
                    intensity_in_each_channel[i, :, :] = smooth
                else:
                    # in this case im contains the optical depth (non convolved)
                    intensity_in_each_channel[i, :, :] = im
                # ---
                if parameters['moment_order'] == 0:
                    moment0 += im*dv  # non convolved (done afterwards)
                if parameters['moment_order'] == 1:
                    moment0 += smooth*dv
                    '''
                    # apply 3-sigma clipping to smooth array if noise
                    # is included for the computation of moment 1 map:
                    if add_noise == 'Yes':
                        for k in range(im_ny):
                            for l in range(im_nx):
                                if np.abs(smooth[k,l]) < 2.0*noise_dev_std_Jy_per_pixel:
                                    smooth[k,l] = 0.0
                    '''
                    moment1 += smooth*vel_range[i]*dv
                if parameters['moment_order'] > 1:
                    sys.exit(
                        'moment map of order > 1 not implemented yet, I must exit!')
            # end loop over wavelengths
            if parameters['moment_order'] == 0:
                im = moment0   # non-convolved quantity
            if parameters['moment_order'] == 1:
                # ratio of two beam-convolved quantities (convolution not redone afterwards)
                im = moment1/moment0
                if parameters['add_noise'] == 'Yes':
                    '''
                    buf = moment0*fluxfactor*beam   # moment 0 in Jy/beam
                    print('min(buf) = ',buf.min())
                    print('max(buf) = ',buf.max())
                    print('noise_dev_std = ', noise_dev_std)
                    for k in range(im_ny):
                        for l in range(im_nx):
                            if np.abs(buf[k,l]) < 7.0*noise_dev_std:
                                im[k,l] = float("NaN")#0.0
                    '''

    if parameters['RTdust_or_gas'] == 'dust' and parameters['polarized_scat'] == 'Yes':
        naxis = 3
        images = np.zeros((5*im_ny*im_nx))
        im = images.reshape(5, im_ny, im_nx)
        for j in range(im_ny):
            for i in range(im_nx):
                line = f.readline()
                dat = line.split()
                im[0, j, i] = float(dat[0])  # I
                im[1, j, i] = float(dat[1])  # Q
                im[2, j, i] = float(dat[2])  # U
                im[3, j, i] = float(dat[3])  # V
                im[4, j, i] = math.sqrt(
                    float(dat[1])**2.0+float(dat[2])**2.0)  # P
                if (j == im_ny-1) and (i == im_nx-1):
                    f.readline()     # empty line

    # close image file
    f.close()

    # Fits header
    hdu = fits.PrimaryHDU()
    hdu.header['BITPIX'] = -32
    hdu.header['NAXIS'] = 2  # naxis
    hdu.header['NAXIS1'] = im_nx
    hdu.header['NAXIS2'] = im_ny
    hdu.header['EPOCH'] = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    hdu.header['CRVAL1'] = float(0.0)
    hdu.header['CRVAL2'] = float(0.0)
    hdu.header['CDELT1'] = float(-1.*pixsize_x_deg)
    hdu.header['CDELT2'] = float(pixsize_y_deg)
    hdu.header['LBDAMIC'] = float(lbda0)
    hdu.header['CUNIT1'] = 'deg     '
    hdu.header['CUNIT2'] = 'deg     '
    hdu.header['CRPIX1'] = float((im_nx+1.)/2.)
    hdu.header['CRPIX2'] = float((im_ny+1.)/2.)
    hdu.header['BUNIT'] = 'JY/PIXEL'
    hdu.header['BTYPE'] = 'Intensity'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0
    del hdu.header['EXTEND']

    # keep track of all parameters in params.dat file
    # for i in range(len(lines_params)):
    #    hdu.header[var[i]] = par[i]
    LOG.write('pixsize '+str(pixsize_x_deg*3600.)+"\n")

    # conversion of the intensity from erg/s/cm^2/Hz/steradian to Jy/pix
    if parameters['plot_tau'] == 'No' and parameters['moment_order'] != 1:
        im = im*fluxfactor

    hdu.data = im.astype('float32')

    if parameters['plot_tau'] == 'No':
        print("Total flux [Jy] = "+str(np.sum(hdu.data)))
        LOG.write('flux '+str(np.sum(hdu.data))+"\n")
    LOG.close()

    hdu.writeto(outfile, output_verify='fix', overwrite=True)

    # save entire intensity channels in another fits file
    if parameters['polarized_scat'] == 'No' and parameters['nlam'] > 1:
        hdu2 = fits.PrimaryHDU()
        hdu2.header['BITPIX'] = -32
        hdu2.header['NAXIS'] = 3  # naxis
        hdu2.header['NAXIS1'] = im_nx
        hdu2.header['NAXIS2'] = im_ny
        hdu2.header['NAXIS3'] = nlam
        hdu2.header['EPOCH'] = 2000.0
        hdu2.header['EQUINOX'] = 2000.0
        hdu2.header['LONPOLE'] = 180.0
        hdu2.header['CTYPE1'] = 'RA---SIN'
        hdu2.header['CTYPE2'] = 'DEC--SIN'
        hdu2.header['CTYPE3'] = 'VRAD'
        hdu2.header['CRVAL1'] = float(0.0)
        hdu2.header['CRVAL2'] = float(0.0)
        hdu2.header['CRVAL3'] = float(-widthkms)
        hdu2.header['CDELT1'] = float(-1.*pixsize_x_deg)
        hdu2.header['CDELT2'] = float(pixsize_y_deg)
        hdu2.header['CDELT3'] = float(dv)  # in km/s
        hdu2.header['LBDAMIC'] = float(lbda0)
        hdu2.header['CUNIT1'] = 'deg     '
        hdu2.header['CUNIT2'] = 'deg     '
        hdu2.header['CUNIT3'] = 'km/s     '
        hdu2.header['CRPIX1'] = float((im_nx+1.)/2.)
        hdu2.header['CRPIX2'] = float((im_ny+1.)/2.)
        hdu2.header['CRPIX3'] = float(1.0)
        hdu2.header['BUNIT'] = 'JY/beam'
        hdu2.header['BTYPE'] = 'Intensity'
        hdu2.header['BSCALE'] = float(1.0)
        hdu2.header['BZERO'] = float(0.0)
        del hdu2.header['EXTEND']
        hdu2.data = intensity_in_each_channel.astype('float32')
        hdu2.writeto(outfileall, output_verify='fix', overwrite=True)
        # ----------

    return outfile

# --------------------
# writing radmc3d.inp
# --------------------
def write_radmc3dinp(incl_dust=1,
                     incl_lines=0,
                     lines_mode=1,
                     nphot=1000000,
                     nphot_scat=1000000,
                     nphot_spec=1000000,
                     nphot_mono=1000000,
                     istar_sphere=0,
                     scattering_mode_max=0,
                     tgas_eq_tdust=1,
                     modified_random_walk=0,
                     itempdecoup=1,
                     setthreads=2,
                     rto_style=3):

    print('writing radmc3d.inp')

    RADMCINP = open('radmc3d.inp', 'w')
    inplines = ["incl_dust = "+str(int(incl_dust))+"\n",
                "incl_lines = "+str(int(incl_lines))+"\n",
                "lines_mode = "+str(int(lines_mode))+"\n",
                "nphot = "+str(int(nphot))+"\n",
                "nphot_scat = "+str(int(nphot_scat))+"\n",
                "nphot_spec = "+str(int(nphot_spec))+"\n",
                "nphot_mono = "+str(int(nphot_mono))+"\n",
                "istar_sphere = "+str(int(istar_sphere))+"\n",
                "scattering_mode_max = "+str(int(scattering_mode_max))+"\n",
                "tgas_eq_tdust = "+str(int(tgas_eq_tdust))+"\n",
                "modified_random_walk = "+str(int(modified_random_walk))+"\n",
                "itempdecoup = "+str(int(itempdecoup))+"\n",
                "setthreads="+str(int(setthreads))+"\n",
                "rto_style="+str(int(rto_style))+"\n"]

    RADMCINP.writelines(inplines)
    RADMCINP.close()

# --------------------
# writing lines.inp
# --------------------
def write_lines(specie, lines_mode):

    print("writing lines.inp")
    path_lines = 'lines.inp'

    lines = open(path_lines, 'w')

    lines.write('2 \n')              # <=== Put this to 2
    # Nr of molecular or atomic species to be modeled
    lines.write('1 \n')
    # LTE calculations
    if lines_mode == 1:
        lines.write('%s    leiden    0    0    0' %
                    specie)    # incl x, incl y, incl z
    else:
        # non-LTE calculations
        lines.write('%s    leiden    0    0    1\n' %
                    specie)    # incl x, incl y, incl z
        lines.write('h2')
    lines.close()

    # Get molecular data file
    molecular_file = 'molecule_'+str(specie)+'.inp'
    if os.path.isfile(molecular_file) == False:
        print('--------- Downloading molecular data file ----------')
        datafile = str(specie)
        if specie == 'hco+':
            datafile = 'hco+@xpol'
        if specie == 'so':
            datafile = 'so@lique'
        if specie == 'cs':
            datafile = 'cs@lique'
        command = 'curl -O https://home.strw.leidenuniv.nl/~moldata/datafiles/'+datafile+'.dat'
        os.system(command)
        command = 'mv '+datafile+'.dat molecule_'+str(specie)+'.inp'
        os.system(command)

# -----------------------
# writing out star parameters
# -----------------------
def write_stars(Rstar=1, Tstar=6000):
    wmin = 0.1
    wmax = 10000.0
    Nw = 150
    Pw = (wmax/wmin)**(1.0/(Nw-1))

    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in range(1, Nw):
        waves[i] = wmin*Pw**i

    print('writing stars.inp')

    path = 'stars.inp'
    wave = open(path, 'w')

    wave.write('\t 2\n')
    wave.write('1 \t'+str(Nw)+'\n')
    wave.write(str(Rstar*R_Sun)+'\t'+str(M_Sun)+'\t 0 \t 0 \t 0 \n')
    for i in range(Nw):
        wave.write('\t'+str(waves[i])+'\n')
    wave.write('\t -'+str(Tstar)+'\n')
    wave.close()

# ---------------------------------------
# write spatial grid in file amr_grid.inp
# ---------------------------------------
def write_AMRgrid(F, R_Scaling=1, Plot=False):

    print("writing spatial grid")
    path_grid = 'amr_grid.inp'

    grid = open(path_grid, 'w')

    grid.write('1 \n')              # iformat/ format number = 1
    grid.write('0 \n')              # Grid style (regular = 0)
    grid.write('101 \n')            # coordsystem: 100 < spherical < 200
    grid.write('0 \n')              # gridinfo
    grid.write('1 \t 1 \t 1 \n')    # incl x, incl y, incl z

    # spherical radius, colatitude, azimuth
    grid.write(str(F.nrad) + '\t' + str(F.ncol)+'\t' + str(F.nsec)+'\n')

    # nrad+1 dimension as we need to enter the coordinates of the cells edges
    for i in range(F.nrad + 1):
        # with unit conversion in cm
        grid.write(str(F.redge[i]*F.culength*1e2)+'\t')
    grid.write('\n')

    # colatitude
    for i in range(F.ncol + 1):
        grid.write(str(F.tedge[i])+'\t')
    grid.write('\n')

    # azimuth
    for i in range(F.nsec + 1):
        grid.write(str(F.pedge[i])+'\t')
    grid.write('\n')

    grid.close()

# -----------------------
# writing out wavelength
# -----------------------
def write_wavelength():
    wmin = 0.1
    wmax = 10000.0
    Nw = 150
    Pw = (wmax/wmin)**(1.0/(Nw-1))

    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in range(1, Nw):
        waves[i] = wmin*Pw**i

    print('writing wavelength_micron.inp')

    path = 'wavelength_micron.inp'
    wave = open(path, 'w')
    wave.write(str(Nw)+'\n')
    for i in range(Nw):
        wave.write(str(waves[i])+'\n')
    wave.close()