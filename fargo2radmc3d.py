# ===================================================================
#                        PLUTO to RADMC3D
# code written by Clement Baruteau (CB), Sebastian Perez (SP) and Marcelo Barraza (MB)
# with substantial contributions from Simon Casassus (SC) and Gaylor Wafflard-Fernandez (GWF)
# ===================================================================
#
# present program can run with either Python 2.X or Python 3.X.
#
# Setup PLUTO outputs for input into RADMC3D (v0.41, Dullemond et
# al). Python based.
#
# ===================================================================

import os.path

# -----------------------------
# Required librairies
# -----------------------------
from astropy.io import fits
from pylab import *
from matplotlib import ticker
from field import *
from image import *
from makedustopac import *
from opacities import *
from particles import *
from radmc3d import *
from read_parameters import *


def main():

    # =========================
    # Read RT parameter file
    # =========================

    parameters = dict()
    read_parameters_file(parameters)
    check_parameters_consistency(parameters)
    print_parameters(parameters)

    # gas density field:
    density = Field(field='rho', parameters=parameters)

    # number of grid cells in the radial and azimuthal directions
    nrad = density.nrad
    ncol = density.ncol
    nsec = density.nsec

    # volume of each grid cell (code units)
    calculate_volume(density)

    volume = np.zeros((nsec, ncol, nrad))
    for i in range(nsec):
        for j in range(ncol):
            for k in range(nrad):
                volume[i, j, k] = density.rmed[k]**2 * np.sin(density.tmed[j]) * density.dr[k] * density.dth[j] * density.dp[i]

    # Mass of gas in units of the star's mass
    Mgas = np.sum(density.data*volume)
    print('Mgas / Mstar= '+str(Mgas)+' and Mgas [kg] = '+str(Mgas*density.cumass))

    # Allocate arrays
    nbin = parameters['nbin']
    bins = np.asarray([0.00002, 0.0002, 0.002, 0.02, 0.2, 0.4, 2., 4, 20, 200, 2000])
    dust_cube = np.zeros((nbin,nsec,ncol,nrad))
    particles_per_bin = np.zeros(nbin)
    tstop_per_bin = np.zeros(nbin)

    # =========================
    # Compute dust mass volume density for each size bin
    # =========================
    if parameters['RTdust_or_gas'] == 'dust' and parameters['recalc_density'] == 'Yes' and parameters['polarized_scat'] == 'No':
        print('--------- computing dust mass volume density ----------')

        particle_data = Particles(ns=parameters['particle_file'], directory=parameters['dir'])
        populate_dust_bins(density, particle_data, nbin, bins, dust_cube, particles_per_bin, tstop_per_bin)

        frac = np.zeros(nbin)
        buf = 0.0
        # finally compute dust surface density for each size bin
        for ibin in range(nbin):
            # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
            frac[ibin] = (pow(bins[ibin+1], (4.0-parameters['pindex'])) - pow(bins[ibin], (4.0-parameters['pindex']))) / \
                (pow(parameters['amax'], (4.0-parameters['pindex'])) - pow(parameters['amin'], (4.0-parameters['pindex'])))
            # total mass of dust particles in current size bin 'ibin'
            M_i_dust = parameters['ratio'] * Mgas * frac[ibin]
            buf += M_i_dust
            print('Dust mass [in units of Mstar] in species ',
                ibin, ' = ', M_i_dust)
            # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
            dust_cube[ibin, :, :, :] *= M_i_dust / volume / particles_per_bin[ibin]
            # conversion in g/cm^2
            # dimensions: nbin, nrad, nsec
            dust_cube[ibin, :, :, :] *= (density.cumass*1e3)/((density.culength*1e2)**2.)

        # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
        if parameters['bin_small_dust'] == 'Yes':
            frac[0] *= 5e3
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas")
            print("Mass fraction of bin 0 changed to: ", str(frac[0]))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # radial index corresponding to 0.3"
            imin = np.argmin(np.abs(density.rmed-1.4))
            # radial index corresponding to 0.6"
            imax = np.argmin(np.abs(density.rmed-2.8))
            dust_cube[0, imin:imax, :] = density.data[imin:imax, :] * parameters['ratio'] * frac[0] * \
                (density.cumass*1e3)/((density.culength*1e2)
                                ** 2.)  # dimensions: nbin, nrad, nsec

        print('Total dust mass [g] = ', np.sum(
            dust_cube[:, :, :, :]*volume*(density.culength*1e2)**2.))
        print('Total dust mass [Mgas] = ', np.sum(
            dust_cube[:, :, :, :]*volume*(density.culength*1e2)**2.)/(Mgas*density.cumass*1e3))
        print('Total dust mass [Mstar] = ', np.sum(
            dust_cube[:, :, :, :]*volume*(density.culength*1e2)**2.)/(density.cumass*1e3))

        # Total dust surface density
        dust_surface_density = np.sum(dust_cube, axis=0)
        print(
            'Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())

        DUSTOUT = open('dust_density.inp', 'w')
        DUSTOUT.write('1 \n')  # iformat
        DUSTOUT.write(str(nrad * nsec * ncol) + ' \n')  # n cells
        DUSTOUT.write(str(int(nbin)) + ' \n')  # nbin size bins

        # array (ncol, nbin, nrad, nsec)
        rhodustcube = np.zeros((nbin, nsec, ncol, nrad))

        # dust aspect ratio as function of ibin and r (or actually, R, cylindrical radius)
        hd = np.zeros((nbin, nrad))

        # gus aspect ratio
        hgas = np.zeros(nrad)
        for irad in range(nrad):
            hgas[irad] = (density.rmed[irad] * np.cos(density.tmed[:]) * density.data[0, :, irad]).sum(axis=0) / (
            density.data[0, :, irad]).sum(axis=0)

        for ibin in range(nbin):
            if parameters['polarized_scat'] == 'No':
                for irad in range(nrad):
                    hd[ibin, irad] = hgas[irad] / np.sqrt(
                        1.0 + tstop_per_bin[ibin] / parameters['alphaviscosity'] * (1.0 + 2.0 * tstop_per_bin[ibin]) /
                        (1.0 + tstop_per_bin[ibin]))
            else:
                print(
                    "Set of initial conditions not implemented for pluto yet. Only parameters['polarized_scat'] == 'No'")
                sys.exit('I must exit!')

        # work out exponential and normalization factors exp(-z^2 / 2H_d^2)
        # with z = r cos(theta) and H_d = h_d x R = h_d x r sin(theta)
        # r = spherical radius, R = cylindrical radius
        rho_dust_cube = dust_cube
        rho_dust_cube = np.nan_to_num(rho_dust_cube)

        # for plotting purposes
        axirhodustcube = np.sum(rho_dust_cube, axis=3) / nsec  # ncol, nbin, nrad

        # Renormalize dust's mass volume density such that the sum over the 3D grid's volume of
        # the dust's mass volume density x the volume of each grid cell does give us the right
        # total dust mass, which equals ratio x Mgas.
        rhofield = np.sum(rho_dust_cube, axis=0)  # sum over dust bins

        Cedge, Aedge, Redge = np.meshgrid(
            density.tedge, density.pedge, density.redge)  # ncol+1, nrad+1, Nsec+1

        r2 = Redge * Redge
        jacob = r2[:-1, :-1, :-1] * np.sin(Cedge[:-1, :-1, :-1])
        dphi = Aedge[1:, :-1, :-1] - Aedge[:-1, :-1, :-1]  # same as 2pi/nsec
        dr = Redge[:-1, :-1, 1:] - Redge[:-1, :-1, :-1]  # same as Rsup-Rinf
        dtheta = Cedge[:-1, 1:, :-1] - Cedge[:-1, :-1, :-1]
        # volume of a cell in cm^3
        vol = jacob * dr * dphi * dtheta * \
              ((density.culength * 1e2) ** 3)  # ncol, nrad, Nsec

        total_mass = np.sum(rhofield * vol)

        normalization_factor = parameters['ratio'] * Mgas * (density.cumass * 1e3) / total_mass
        rho_dust_cube = rho_dust_cube * normalization_factor
        print('total dust mass after vertical expansion [g] = ', np.sum(np.sum(
            rho_dust_cube, axis=0) * vol), ' as normalization factor = ', normalization_factor)

        # write mass volume densities for all size bins
        for ibin in range(nbin):
            print('dust species in bin', ibin, 'out of ', nbin - 1)
            for k in range(nsec):
                for j in range(ncol):
                    for i in range(nrad):
                        DUSTOUT.write(str(rho_dust_cube[ibin, k, j, i]) + ' \n')

        # print max of dust's mass volume density at each colatitude
        for j in range(ncol):
            print('max(rho_dustcube) [g cm-3] for colatitude index j = ',
                  j, ' = ', rho_dust_cube[:, :, j, :].max())

        DUSTOUT.close()

        # free RAM memory
        del rho_dust_cube, dust_cube
    elif parameters['RTdust_or_gas'] == 'gas':
        print("Set of initial conditions not implemented for pluto yet. Only parameters['RTdust_or_gas'] == 'dust'")
        sys.exit('I must exit!')
    elif parameters['polarized_scat'] == 'Yes':
        print("Set of initial conditions not implemented for pluto yet. Only parameters['polarized_scat'] == 'No'")
        sys.exit('I must exit!')
    else:
        print('--------- I did not compute dust densities (recalc_density = No in params.dat file) ----------')

    # =========================
    # Compute dust opacities
    # =========================
    if parameters['RTdust_or_gas'] == 'dust' and parameters['recalc_opac'] == 'Yes':
        print('--------- computing dust opacities ----------')

        # Calculation of opacities uses the python scripts makedustopac.py and bhmie.py
        # which were written by C. Dullemond, based on the original code by Bohren & Huffman.

        logawidth = 0.05          # Smear out the grain size by 5% in both directions
        na = 20            # Use 10 grain size samples per bin size
        chop = 1.            # Remove forward scattering within an angle of 5 degrees
        # Extrapolate optical constants beyond its wavelength grid, if necessary
        extrapol = True
        verbose = False         # If True, then write out status information
        ntheta = 181           # Number of scattering angle sampling points
        # link to optical constants file
        optconstfile = os.path.expanduser(parameters['opacity_dir']) + '/' + parameters['species'] + '.lnk'

        # The material density in gram / cm^3
        graindens = 2.0  # default density in g / cc
        if parameters['species'] == 'mix_2species_porous' or parameters['species'] == 'mix_2species_porous_ice' or parameters['species'] == 'mix_2species_porous_ice70':
            graindens = 0.1  # g / cc
        if parameters['species'] == 'mix_2species' or parameters['species'] == 'mix_2species_60silicates_40ice':
            graindens = 1.7  # g / cc
        if parameters['species'] == 'mix_2species_ice70':
            graindens = 1.26  # g / cc
        if parameters['species'] == 'mix_2species_60silicates_40carbons':
            graindens = 2.7  # g / cc

        # Set up a wavelength grid (in cm) upon which we want to compute the opacities
        # 1 micron -> 1 cm
        lamcm = 10.0**np.linspace(0, 4, 200)*1e-4

        # Set up an angular grid for which we want to compute the scattering matrix Z
        theta = np.linspace(0., 180., ntheta)

        for ibin in range(int(nbin)):
            # median grain size in cm in current bin size:
            agraincm = 10.0**(0.5*(np.log10(1e2 *
                            bins[ibin]) + np.log10(1e2*bins[ibin+1])))

            print('====================')
            print('bin ', ibin+1, '/', nbin)
            print('grain size [cm]: ', agraincm,
                ' with grain density [g/cc] = ', graindens)
            print('====================')
            pathout = parameters['species']+str(ibin)
            opac = compute_opac_mie(optconstfile, graindens, agraincm, lamcm, theta=theta,
                                    extrapolate=extrapol, logawidth=logawidth, na=na,
                                    chopforward=chop, verbose=verbose)
            if (parameters['scat_mode'] >= 3):
                print("Writing dust opacities in dustkapscatmat* files")
                write_radmc3d_scatmat_file(opac, pathout)
            else:
                print("Writing dust opacities in dustkappa* files")
                write_radmc3d_kappa_file(opac, pathout)
    else:
        print('------- taking dustkap* opacity files in current directory (recalc_opac = No in params.dat file) ------ ')

    # Write dustopac.inp file even if we don't (re)calculate dust opacities
    if parameters['RTdust_or_gas'] == 'dust':
        write_dustopac(species=parameters['species'], scat_mode=parameters['scat_mode'], nbin=parameters['nbin'], )
        if parameters['plot_opac'] == 'Yes':
            print('--------- plotting dust opacities ----------')
            plot_opacities(species=parameters['species'], amin=parameters['amin'], amax=parameters['amax'],
                        nbin=parameters['nbin'], lbda1=parameters['wavelength']*1e3)

    # write radmc3d script, particularly useful in case radmc3d mctherm /
    # ray_tracing are run on a different platform
    write_radmc3d_script(parameters)

    # =========================
    # Call to RADMC3D thermal solution and ray tracing
    # =========================
    if (parameters['recalc_radmc'] == 'Yes' or parameters['recalc_rawfits'] == 'Yes'):
        # Write other parameter files required by RADMC3D
        print('--------- printing auxiliary files ----------')

        # need to check why we need to output wavelength...
        if parameters['recalc_rawfits'] == 'No':
            write_wavelength()
            write_stars(Rstar=parameters['rstar'], Tstar=parameters['teff'])
            # Write 3D spherical grid for RT computational calculation
            write_AMRgrid(density, Plot=False)

            # rto_style = 3 means that RADMC3D will write binary output files
            # setthreads corresponds to the number of threads (cores) over which radmc3d runs
            write_radmc3dinp(incl_dust=parameters['incl_dust'], incl_lines=parameters['incl_lines'], lines_mode=parameters['lines_mode'], nphot_scat=parameters['nb_photons_scat'], 
                            nphot=parameters['nb_photons'], rto_style=3, tgas_eq_tdust=parameters['tgas_eq_tdust'], modified_random_walk=1, scattering_mode_max=parameters['scat_mode'],
                            setthreads=parameters['nbcores'])

        # Add 90 degrees to position angle so that RADMC3D's definition of
        # position angle be consistent with observed position
        # angle, which is what we enter in the params.dat file
        M = RTmodel(distance=parameters['distance'], Lambda=parameters['wavelength']*1e3, label=parameters['label'], line=parameters['gasspecies'], iline=parameters['iline'],
                    vkms=parameters['vkms'], widthkms=parameters['widthkms'], npix=parameters['nbpixels'], phi=parameters['phiangle'], incl=parameters['inclination'], 
                    posang=parameters['posangle']+90.0)

        # Set dust / gas temperature if Tdust_eq_Thydro == 'Yes'
        if parameters['recalc_rawfits'] == 'No' and parameters['Tdust_eq_Thydro'] == 'Yes' and parameters['RTdust_or_gas'] == 'dust' and parameters['recalc_temperature'] == 'Yes':
            print('--------- Writing temperature file (no mctherm) ----------')
            os.system('rm -f dust_temperature.bdat')        # avoid confusion!...
            TEMPOUT = open('dust_temperature.dat', 'w')
            TEMPOUT.write('1 \n')                           # iformat
            TEMPOUT.write(str(nrad*nsec*ncol)+' \n')        # n cells
            TEMPOUT.write(str(int(nbin))+' \n')             # nbin size bins

            gas_temp = np.zeros((ncol, nrad, nsec))
            thydro = parameters['aspectratio']*parameters['aspectratio']*density.cutemp * \
                density.rmed**(-1.0+2.0*parameters['flaringindex'])
            for k in range(nsec):
                for j in range(ncol):
                    gas_temp[j, :, k] = thydro

            # write dust temperature for all size bins
            for ibin in range(nbin):
                print('writing temperature of dust species in bin',
                    ibin, 'out of ', nbin-1)
                for k in range(nsec):
                    for j in range(ncol):
                        for i in range(nrad):
                            TEMPOUT.write(str(gas_temp[j, i, k])+' \n')
            TEMPOUT.close()
            del gas_temp

        # Now run RADMC3D
        if parameters['recalc_rawfits'] == 'No':
            print('--------- Now executing RADMC3D ----------')
            os.system('./script_radmc')

        print('--------- exporting results in fits format ----------')
        outfile = exportfits(M,parameters)

        if parameters['plot_temperature'] == 'Yes':
            # Plot midplane and surface temperature profiles
            if parameters['RTdust_or_gas'] == 'dust':
                Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
                Temp = Temp[4:]
                Temp = Temp.reshape(nbin, nsec, ncol, nrad)
                # Keep temperature of the largest dust species
                Temp = Temp[-1, :, :, :]
            else:
                print("Set of initial conditions not implemented for pluto yet. Only parameters['RTdust_or_gas'] == 'dust'")
                sys.exit('I must exit!')

            # Temperature in the midplane (ncol/2 given that the grid extends on both sides about the midplane)
            # not really in the midplane because theta=pi/2 is an edge colatitude...
            Tm = Temp[:, ncol//2, :]
            # Temperature at one surface
            Ts = Temp[:, 0, :]
            # Azimuthally-averaged radial profiles
            axiTm = np.sum(Tm, axis=0)/nsec
            axiTs = np.sum(Ts, axis=0)/nsec
            fig = plt.figure(figsize=(4., 3.))
            ax = fig.gca()
            S = density.rmed*density.culength/1.5e11  # radius in a.u.
            # gas temperature in hydro simulation in Kelvin (assuming T in R^-1/2, no matter
            # the value of the gas flaring index in the simulation)
            Tm_model = parameters['aspectratio']*parameters['aspectratio'] * \
                density.cutemp*density.rmed**(-1.0+2.0*parameters['flaringindex'])
            ax.plot(S, axiTm, 'bo', markersize=1., label='midplane')
            ax.plot(S, Tm_model, 'b--', markersize=1., label='midplane hydro')
            ax.plot(S, axiTs, 'rs', markersize=1., label='surface')
            ax.set_xlabel(r'$R ({\rm au})$', fontsize=12)
            ax.set_ylabel(r'$T ({\rm K})$', fontsize=12)
            # ax.set_xlim(20.0, 100.0) # cuidadin!
            ax.set_xlim(S.min(), S.max())
            # ax.set_ylim(10.0, 150.0)  # cuidadin!
            ax.set_ylim(Tm.min(), Ts.max())
            ax.tick_params(axis='both', direction='in', top='on', right='on')
            ax.tick_params(axis='both', which='minor',
                        top='on', right='on', direction='in')
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.legend(frameon=False)
            fig.add_subplot(ax)
            filenameT = 'T_R_'+parameters['label']+'.pdf'
            fig.savefig(filenameT, dpi=180, bbox_inches='tight')
            fig.clf()
            # Save radial profiles in an ascii file
            filenameT2 = 'T_R_'+parameters['label']+'.dat'
            TEMPOUT = open(filenameT2, 'w')
            TEMPOUT.write(
                '# radius [au] \t T_midplane_radmc3d \t T_surface_radmc3d \t T_midplane_hydro\n')
            for i in range(nrad):
                TEMPOUT.write('%f \t %f \t %f \t %f\n' %
                            (S[i], axiTm[i], axiTs[i], Tm_model[i]))
            TEMPOUT.close()
            # free RAM memory
            del Temp
    else:
        print('------- I did not run RADMC3D, using existing .fits file for convolution ')
        print('------- (recalc_radmc = No in params.dat file) and final image ------ ')

        if parameters['RTdust_or_gas'] == 'dust':
            outfile = 'image_'+str(parameters['label'])+'_lbda'+str(parameters['wavelength'])+'_i'+str(parameters['inclination'])+'_phi'+str(parameters['phiangle'])+'_PA'+str(parameters['posangle'])
        else:
            print("Set of initial conditions not implemented for pluto yet. Only parameters['RTdust_or_gas'] == 'dust'")
            sys.exit('I must exit!')

        if parameters['secondorder'] == 'Yes':
            outfile = outfile+'_so'
        if parameters['dustdens_eq_gasdens'] == 'Yes':
            outfile = outfile+'_ddeqgd'
        if parameters['bin_small_dust'] == 'Yes':
            outfile = outfile+'_bin0'

        outfile = outfile+'.fits'


    # =========================
    # Convolve raw flux with beam and produce final image
    # =========================
    if parameters['recalc_fluxmap'] == 'Yes':
        print('--------- Convolving and writing final image ----------')

        f = fits.open('./'+outfile)

        # remove .fits extension
        outfile = os.path.splitext(outfile)[0]

        # add bmaj information
        outfile = outfile + '_bmaj'+str(parameters['bmaj'])

        outfile = outfile+'.fits'

        hdr = f[0].header
        # pixel size converted from degrees to arcseconds
        cdelt = np.abs(hdr['CDELT1']*3600.)

        # get wavelength and convert it from microns to mm
        lbda0 = hdr['LBDAMIC']*1e-3

        # a) case with no polarized scattering: fits file directly contains raw intensity field
        if parameters['polarized_scat'] == 'No':
            nx = hdr['NAXIS1']
            ny = hdr['NAXIS2']
            raw_intensity = f[0].data
            if parameters['recalc_radmc'] == 'No' and parameters['plot_tau'] == 'No':
                # sum over pixels
                print("Total flux [Jy] = "+str(np.sum(raw_intensity)))
            # check beam is correctly handled by inserting a source point at the
            # origin of the raw intensity image
            if parameters['check_beam'] == 'Yes':
                raw_intensity[:, :] = 0.0
                raw_intensity[nx//2-1, ny//2-1] = 1.0
            # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
            if parameters['add_noise'] == 'Yes' and parameters['RTdust_or_gas'] == 'dust' and parameters['plot_tau'] == 'No':
                # beam area in pixel^2
                beam = (np.pi/(4.*np.log(2.)))*parameters['bmaj']*parameters['bmin']/(cdelt**2.)
                # noise standard deviation in Jy per pixel (I've checked the expression below works well)
                noise_dev_std_Jy_per_pixel = parameters['noise_dev_std'] / \
                    np.sqrt(0.5*beam)  # 1D
                # noise array
                noise_array = np.random.normal(
                    0.0, noise_dev_std_Jy_per_pixel, size=parameters['nbpixels']*parameters['nbpixels'])
                noise_array = noise_array.reshape(parameters['nbpixels'], parameters['nbpixels'])
                raw_intensity += noise_array
            if parameters['brightness_temp'] == 'Yes':
                # beware that all units are in cgs! We need to convert
                # 'intensity' from Jy/pixel to cgs units!
                # pixel size in each direction in cm
                pixsize_x = cdelt*parameters['distance']*au
                pixsize_y = pixsize_x
                # solid angle subtended by pixel size
                pixsurf_ster = pixsize_x*pixsize_y/parameters['distance']/parameters['distance']/pc/pc
                # convert intensity from Jy/pixel to erg/s/cm2/Hz/sr
                intensity_buf = raw_intensity/1e23/pixsurf_ster
                # beware that lbda0 is in mm right now, we need to have it in cm in the expression below
                raw_intensity = (h*c/kB/(lbda0*1e-1))/np.log(1. +
                                                            2.*h*c/intensity_buf/pow((lbda0*1e-1), 3.))
                #raw_intensity = np.nan_to_num(raw_intensity)
        else:
            print("Set of initial conditions not implemented for pluto yet. Only parameters['polarized_scat'] == 'No'")
            sys.exit('I must exit!')

        # ------------
        # smooth image
        # ------------
        # beam area in pixel^2
        beam = (np.pi/(4.*np.log(2.)))*parameters['bmaj']*parameters['bmin']/(cdelt**2.)
        # stdev lengths in pixel
        stdev_x = (parameters['bmaj']/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
        stdev_y = (parameters['bmin']/(2.*np.sqrt(2.*np.log(2.)))) / cdelt

        # a) case with no polarized scattering
        if parameters['polarized_scat'] == 'No' and parameters['plot_tau'] == 'No':
            # Call to Gauss_filter function
            if parameters['moment_order'] != 1:
                smooth = Gauss_filter(raw_intensity, stdev_x,
                                    stdev_y, parameters['bpaangle'], Plot=False)
            else:
                smooth = raw_intensity

            # convert image from Jy/pixel to mJy/beam or microJy/beam
            # could be refined...
            if parameters['brightness_temp'] == 'Yes':
                convolved_intensity = smooth
            if parameters['brightness_temp'] == 'No':
                convolved_intensity = smooth * 1e3 * beam   # mJy/beam

            strflux = 'Flux of continuum emission (mJy/beam)'
            if parameters['gasspecies'] == 'co':
                strgas = r'$^{12}$CO'
            elif parameters['gasspecies'] == '13co':
                strgas = r'$^{13}$CO'
            elif parameters['gasspecies'] == 'c17o':
                strgas = r'C$^{17}$O'
            elif parameters['gasspecies'] == 'c18o':
                strgas = r'C$^{18}$O'
            elif parameters['gasspecies'] == 'hco+':
                strgas = r'HCO+'
            elif parameters['gasspecies'] == 'so':
                strgas = r'SO'
            else:
                strgas = parameters['gasspecies']
            if parameters['gasspecies'] != 'so':
                strgas += r' ($%d \rightarrow %d$)' % (parameters['iline'], parameters['iline']-1)
            if parameters['gasspecies'] == 'so' and parameters['iline'] == 14:
                strgas += r' ($5_6 \rightarrow 4_5$)'

            if parameters['brightness_temp'] == 'Yes':
                if parameters['RTdust_or_gas'] == 'dust':
                    strflux = r'Brightness temperature (K)'
            else:
                if convolved_intensity.max() < 1.0:
                    convolved_intensity = smooth * 1e6 * beam   # microJy/beam
                    strflux = r'Flux of continuum emission ($\mu$Jy/beam)'

        if parameters['plot_tau'] == 'Yes':
            convolved_intensity = raw_intensity
            strflux = r'Absorption optical depth $\tau'

        # -------------------------------------
        # SP: save convolved flux map solution to fits
        # -------------------------------------
        hdu = fits.PrimaryHDU()
        hdu.header['BITPIX'] = -32
        hdu.header['NAXIS'] = 2  # 2
        hdu.header['NAXIS1'] = parameters['nbpixels']
        hdu.header['NAXIS2'] = parameters['nbpixels']
        hdu.header['EPOCH'] = 2000.0
        hdu.header['EQUINOX'] = 2000.0
        hdu.header['LONPOLE'] = 180.0
        hdu.header['CTYPE1'] = 'RA---SIN'
        hdu.header['CTYPE2'] = 'DEC--SIN'
        hdu.header['CRVAL1'] = float(0.0)
        hdu.header['CRVAL2'] = float(0.0)
        hdu.header['CDELT1'] = hdr['CDELT1']
        hdu.header['CDELT2'] = hdr['CDELT2']
        hdu.header['LBDAMIC'] = hdr['LBDAMIC']
        hdu.header['CUNIT1'] = 'deg     '
        hdu.header['CUNIT2'] = 'deg     '
        hdu.header['CRPIX1'] = float((parameters['nbpixels']+1.)/2.)
        hdu.header['CRPIX2'] = float((parameters['nbpixels']+1.)/2.)
        if strflux == 'Flux of continuum emission (mJy/beam)':
            hdu.header['BUNIT'] = 'milliJY/BEAM'
        if strflux == r'Flux of continuum emission ($\mu$Jy/beam)':
            hdu.header['BUNIT'] = 'microJY/BEAM'
        if strflux == '':
            hdu.header['BUNIT'] = ''
        hdu.header['BTYPE'] = 'FLUX DENSITY'
        hdu.header['BSCALE'] = 1
        hdu.header['BZERO'] = 0
        del hdu.header['EXTEND']
        # keep track of all parameters in params.dat file
        # for i in range(len(lines_params)):
        #    hdu.header[var[i]] = par[i]
        hdu.data = convolved_intensity
        inbasename = os.path.basename('./'+outfile)
        if parameters['add_noise'] == 'Yes':
            substr = '_wn'+str(parameters['noise_dev_std'])+'_JyBeam.fits'
            jybeamfileout = re.sub('.fits', substr, inbasename)
        else:
            jybeamfileout = re.sub('.fits', '_JyBeam.fits', inbasename)
        hdu.writeto(jybeamfileout, overwrite=True)

        # --------------------
        # plotting image panel
        # --------------------
        matplotlib.rcParams.update({'font.size': 20})
        matplotlib.rc('font', family='Arial')
        fontcolor = 'white'

        # name of pdf file for final image
        fileout = re.sub('.fits', '.pdf', jybeamfileout)
        fig = plt.figure(figsize=(8., 8.))
        plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
        ax = plt.gca()

        # Set x-axis orientation, x- and y-ranges
        # Convention is that RA offset increases leftwards (ie,
        # east is to the left), while Dec offset increases from
        # bottom to top (ie, north is the top)
        if (nx % 2) == 0:
            dpix = 0.5
        else:
            dpix = 0.0
        dpix = 0.0
        a0 = cdelt*(nx//2.-dpix)   # >0
        a1 = -cdelt*(nx//2.+dpix)  # <0
        d0 = -cdelt*(nx//2.-dpix)  # <0
        d1 = cdelt*(nx//2.+dpix)   # >0
        # da positive definite
        if parameters['minmaxaxis'] < abs(a0):
            da = parameters['minmaxaxis']
        else:
            da = np.max(abs(a0), abs(a1))
        mina = da
        maxa = -da
        xlambda = mina - 0.166*da
        ax.set_ylim(-da, da)
        ax.set_xlim(da, -da)      # x (=R.A.) increases leftward
        dmin = -da
        dmax = da

        # x- and y-ticks and labels
        ax.tick_params(top='on', right='on', length=5, width=1.0, direction='out')
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.set_xticks(ax.get_yticks())    # set same ticks in x and y in cartesian
        # ax.set_yticks(ax.get_xticks())    # set same ticks in x and y in cartesian
        ax.set_xlabel('RA offset [arcsec]')
        ax.set_ylabel('Dec offset [arcsec]')

        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'
        min = convolved_intensity.min()
        max = convolved_intensity.max()
        CM = ax.imshow(convolved_intensity, origin='lower', cmap=parameters['mycolormap'],
                    interpolation='bilinear', extent=[a0, a1, d0, d1], vmin=min, vmax=max, aspect='auto')

        # Add wavelength in top-left corner
        strlambda = '$\lambda$='+str(round(lbda0, 2))+'mm'  # round to 2 decimals
        if lbda0 < 0.01:
            strlambda = '$\lambda$='+str(round(lbda0*1e3, 2))+'$\mu$m'
        ax.text(xlambda, dmax-0.166*da, strlambda, fontsize=20,
                color='white', weight='bold', horizontalalignment='left')

        # Add + sign at the origin
        ax.plot(0.0, 0.0, '+', color='white', markersize=10)
        '''
        if check_beam == 'Yes':
            ax.contour(convolved_intensity,levels=[0.5*convolved_intensity.max()],color='black', linestyles='-',origin='lower',extent=[a0,a1,d0,d1])
        '''

        # plot beam
        if parameters['plot_tau'] == 'No':
            from matplotlib.patches import Ellipse
            e = Ellipse(xy=[xlambda, dmin+0.166*da], width=parameters['bmin'],
                        height=parameters['bmaj'], angle=parameters['bpaangle']+90.0)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('white')
            e.set_alpha(1.0)
            ax.add_artist(e)

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb = plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')
        cax.xaxis.set_major_locator(plt.MaxNLocator(6))
        # title on top
        cax.xaxis.set_label_position('top')
        cax.set_xlabel(strflux)
        cax.xaxis.labelpad = 8

        plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
        plt.clf()

        # =====================
        # Compute deprojection and polar expansion (SP)
        # =====================
        if parameters['deproj_polar'] == 'Yes':
            currentdir = os.getcwd()
            alpha_min = 0.          # deg, PA of offset from the star
            Delta_min = 0.          # arcsec, amplitude of offset from the star
            RA = 0.0  # if input image is a prediction, star should be at the center
            DEC = 0.0  # note that this deprojection routine works in WCS coordinates
            cosi = np.cos(parameters['inclination_input']*np.pi/180.)

            print('deprojection around PA [deg] = ', parameters['posangle'])
            print('and inclination [deg] = ', parameters['inclination_input'])

            # makes a new directory "deproj_polar_dir" and calculates a number
            # of products: copy of the input image [_fullim], centered at
            # (RA,DEC) [_centered], deprojection by cos(i) [_stretched], polar
            # image [_polar], etc. Also, a _radial_profile which is the
            # average radial intensity.
            exec_polar_expansions(jybeamfileout, 'deproj_polar_dir', parameters['posangle'], cosi, RA=RA, DEC=DEC,
                                alpha_min=alpha_min, Delta_min=Delta_min,
                                XCheckInv=False, DoRadialProfile=False,
                                DoAzimuthalProfile=False, PlotRadialProfile=False,
                                zoomfactor=1.)

            # Save polar fits in current directory
            fileout = re.sub('.pdf', '_polar.fits', fileout)
            command = 'cp deproj_polar_dir/'+fileout+' .'
            os.system(command)

            filein = re.sub('.pdf', '_polar.fits', fileout)
            # Read fits file with deprojected field in polar coordinates
            f = fits.open(filein)
            convolved_intensity = f[0].data    # uJy/beam

            # azimuthal shift such that PA=0 corresponds to y-axis pointing upwards, and
            # increases counter-clockwise from that axis
            if parameters['xaxisflip'] == 'Yes':
                jshift = int(parameters['nbpixels']/2)
            else:
                jshift = int(parameters['nbpixels']/2)
            convolved_intensity = np.roll(
                convolved_intensity, shift=-jshift, axis=1)

            # -------------------------------
            # plot image in polar coordinates
            # -------------------------------
            fileout = re.sub('.fits', '.pdf', filein)
            fig = plt.figure(figsize=(8., 8.))
            plt.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.09)
            ax = plt.gca()

            # Set x- and y-ranges
            ax.set_xlim(-180, 180)          # PA relative to Clump 1's
            if (parameters['minmaxaxis'] < np.max(abs(a0), abs(a1))):
                ymax = parameters['minmaxaxis']
            else:
                ymax = np.max(abs(a0), abs(a1))
            ax.set_ylim(0, ymax)      # Deprojected radius in arcsec

            if ((nx % 2) == 0):
                dpix = 0.5
            else:
                dpix = 0.0
            a0 = cdelt*(nx//2.-dpix)   # >0

            ax.tick_params(top='on', right='on', length=5,
                        width=1.0, direction='out')
            ax.set_xticks((-180, -120, -60, 0, 60, 120, 180))
            # ax.set_yticks((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7))
            ax.set_xlabel('Position Angle [deg]')
            ax.set_ylabel('Radius [arcsec]')

            # imshow does a bilinear interpolation. You can switch it off by putting
            # interpolation='none'
            min = convolved_intensity.min()  # not exactly same as 0
            max = convolved_intensity.max()
            CM = ax.imshow(convolved_intensity, origin='lower', cmap=parameters['mycolormap'], interpolation='bilinear', extent=[
                        -180, 180, 0, np.max(abs(a0), abs(a1))], vmin=min, vmax=max, aspect='auto')   # (left, right, bottom, top)

            # Add wavelength in bottom-left corner
            strlambda = '$\lambda$=' + \
                str(round(lbda0, 2))+'mm'  # round to 2 decimals
            if lbda0 < 0.01:
                strlambda = '$\lambda$='+str(round(lbda0*1e3, 2))+'$\mu$m'
            ax.text(-160, 0.95*ymax, strlambda, fontsize=20, color='white',
                    weight='bold', horizontalalignment='left', verticalalignment='top')

            # plot color-bar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="2.5%", pad=0.12)
            cb = plt.colorbar(CM, cax=cax, orientation='horizontal')
            cax.xaxis.tick_top()
            cax.xaxis.set_tick_params(labelsize=20, direction='out')
            cax.xaxis.set_major_locator(plt.MaxNLocator(6))
            # title on top
            cax.xaxis.set_label_position('top')
            cax.set_xlabel(strflux)
            cax.xaxis.labelpad = 8

            plt.savefig('./'+fileout, dpi=160)
            plt.clf()

            if parameters['axi_intensity'] == 'Yes':
                average_convolved_intensity = np.zeros(parameters['nbpixels'])
                for j in range(parameters['nbpixels']):
                    for i in range(parameters['nbpixels']):
                        average_convolved_intensity[j] += convolved_intensity[j][i]/parameters['nbpixels']

                # radius in arcseconds
                rkarr = np.linspace(
                    0, density.rmed[-1]*density.culength/1.5e11/parameters['distance'], parameters['nbpixels'])

                nb_noise = 0
                if parameters['add_noise'] == 'Yes':
                    nb_noise = 1

                file = open('axiconv%d.dat' % (nb_noise), 'w')
                for kk in range(parameters['nbpixels']):
                    file.write('%s\t%s\t%s\n' % (str(rkarr[kk]), str(
                        np.mean(convolved_intensity[kk])), str(np.std(convolved_intensity[kk]))))
                file.close()

                fig = plt.figure(figsize=(8., 3.))
                ax = plt.gca()
                plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.25)

                ax.plot(rkarr*parameters['distance'], average_convolved_intensity, color='k')
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                # Deprojected radius in arcsec
                ax.set_xlim(0, parameters['minmaxaxis']*parameters['distance'])
                ax.tick_params('both', labelsize=18)
                ax.set_xlabel('Orbital radius [au]', fontsize=18)
                # ax.set_ylabel(r'Axisymmetric convolved intensity [$\mu$Jy/beam]', family='monospace', fontsize=18)
                ax.set_ylabel(r'$I_C$ [$\mu$Jy/beam]', fontsize=18)

                plt.savefig('./'+'axi'+fileout, dpi=160)
                plt.clf()

            os.system('rm -rf deproj_polar_dir')
            os.chdir(currentdir)

    print('--------- done! ----------')


if __name__ == "__main__":
    main()
