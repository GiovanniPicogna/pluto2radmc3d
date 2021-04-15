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

# -----------------------------
# Requires librairies
# -----------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
import math
import copy
import sys
import os
import array
import subprocess
from astropy.io import fits
import matplotlib
import re
from astropy.convolution import convolve, convolve_fft
from matplotlib.colors import LinearSegmentedColormap
import os.path
from scipy import ndimage
import scipy.interpolate
from copy import deepcopy
from astropy.wcs import WCS
import scipy as sp
from scipy.ndimage import map_coordinates
from pylab import *
import matplotlib.colors as colors
from makedustopac import *
from matplotlib.mlab import *
import struct
import read_parameters
import field
try:
    import h5py as h5
    hasH5 = True
except ImportError:
    hasH5 = False

def main():

    parameters = dict()
    read_parameters.read_parameters_file(parameters)
    read_parameters.check_parameters_consistency(parameters)
    read_parameters.print_parameters(parameters)

    # gas surface density field:
    gas = field.Field(field='rho', parameters=parameters, directory=parameters['dir'])

    # 2D computational grid: R = grid cylindrical radius in code units, T = grid azimuth in radians
    R = gas.redge
    T = gas.pedge

    # number of grid cells in the radial and azimuthal directions
    nrad = gas.nrad
    ncol = gas.ncol
    nsec = gas.nsec

    # extra useful quantities (code units)
    Rinf = gas.redge[0:len(gas.redge)-1]
    Rsup = gas.redge[1:len(gas.redge)]

    # volume of each grid cell (code units)
    volume = np.zeros((nsec,ncol,nrad))
    for i in range(nsec):
        for j in range(ncol):
            for k in range(nrad):
                volume[i,j,k] = gas.rmed[k]**2 * np.sin(gas.tmed[j]) * gas.dr[k] * gas.dth[j] * gas.dp[i]

    # Mass of gas in units of the star's mass
    Mgas = np.sum(gas.data*volume)
    print('Mgas / Mstar= '+str(Mgas)+' and Mgas [kg] = '+str(Mgas*gas.cumass))

    # Allocate arrays
    dust = np.zeros((nsec*ncol*nrad*nbin))
    bins = np.asarray([0.00002, 0.0002, 0.002, 0.02, 0.2, 0.4, 2., 4, 20, 200, 2000])
    nparticles = np.zeros(nbin)     # number of particles per bin size
    # average Stokes number of particles per bin size
    avgstokes = np.zeros(nbin)

    # Color map
    mycolormap = 'nipy_spectral'
    if moment_order == 1:
        mycolormap = 'RdBu_r'

    '''
    # check out memory available on your architecture
    mem_gib = float(psutil.virtual_memory().total/1024**3)
    mem_array_gib = nrad*nsec*ncol*nbin*8.0/(1024.**3)
    if (mem_array_gib/mem_gib > 0.5):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Beware that the memory requested for allocating the dust mass volume density or the temperature arrays')
        print('is very close to the amount of memory available on your architecture...')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    '''

    # =========================
    # 2. compute dust surface density for each size bin
    # =========================
    print('--------- computing dust surface density ----------')

    if (RTdust_or_gas == 'dust' and recalc_density == 'Yes' and polarized_scat == 'No' and fargo3d == 'No'):

        # read information on the dust particles
        #(rad, azi, vr, vt, Stokes, a) = np.loadtxt(dir+'/dustsystat'+str(on)+'.dat', unpack=True)
        ns = int(particle_file)
        pdata = Particles(ns=ns, directory=dir)
        rad = pdata.pos_x
        azi = pdata.pos_y
        vr = pdata.vel_x
        vt = pdata.vel_y
        Stokes = pdata.tstop
        a = pdata.radius

        # Populate dust bins
        for m in range(len(rad)):   # sum over dust particles
            r = rad[m]
            t = azi[m]
            # radial index of the cell where the particle is
            i = int(np.log(r/gas.redge.min())/np.log(gas.redge.max()/gas.redge.min()) * nrad)
            if (i < 0 or i >= nrad+2):
                sys.exit('pb with i = ', i, ' in recalc_density step: I must exit!')

            # polar index of the cell where the particles is
            j = ncol - 1 - int(np.log((np.pi-t)/(np.pi-gas.tedge.max()))/np.log((np.pi-gas.tedge.min())/(np.pi-gas.tedge.max())) * ncol)
            #print(j,gas.tmed[j],t)
            #print(gas.tmed[j],pdata.pos_y[m])
            if (j < 0 or j >= ncol+2):
                sys.exit('pb with j = ', j, ' in recalc_density step: I must exit!')
    
            # azimuthal index of the cell where the particle is
            # (general expression since grid spacing in azimuth is always arithmetic)
            k = pdata.pcell_z[m]
            if (k < 0 or k >= nsec+2):
                sys.exit('pb with k = ', k, ' in recalc_density step: I must exit!')

            # particle size
            pcsize = a[m]
            #print(pcsize)
            # find out which bin particle belongs to
            if(pcsize < 0.0002):
                ibin = 0
            elif(pcsize < 0.002):
                ibin = 1
            elif(pcsize < 0.02):
                ibin = 2
            elif(pcsize < 0.2):
                ibin = 3
            elif(pcsize < 0.4):
                ibin = 4
            elif(pcsize < 2):
                ibin = 5
            elif(pcsize < 4):
                ibin = 6
            elif(pcsize < 20):
                ibin = 7
            elif(pcsize < 200):
                ibin = 8
            else:
                ibin = 9

            if(ibin < 3):
                k = ibin*nsec*ncol*nrad + k*ncol*nrad + j*nrad + i
                dust[k] += 1
                nparticles[ibin] += 1
                avgstokes[ibin] += Stokes[m]

        for ibin in range(nbin):
            if nparticles[ibin] == 0:
                nparticles[ibin] = 1
            avgstokes[ibin] /= nparticles[ibin]
            print(str(nparticles[ibin])+' grains between ' +
                str(bins[ibin])+' and '+str(bins[ibin+1])+' meters')

        # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
        dustcube = dust.reshape(nbin, nsec, ncol, nrad)

        frac = np.zeros(nbin)
        buf = 0.0
        # finally compute dust surface density for each size bin
        for ibin in range(nbin):
            # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
            frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin] **
                        (4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
            # total mass of dust particles in current size bin 'ibin'
            M_i_dust = ratio * Mgas * frac[ibin]
            buf += M_i_dust
            print('Dust mass [in units of Mstar] in species ',
                ibin, ' = ', M_i_dust)
            # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
            dustcube[ibin, :, :, :] *= M_i_dust / volume / nparticles[ibin]
            # conversion in g/cm^2
            # dimensions: nbin, nrad, nsec
            dustcube[ibin, :, :, :] *= (gas.cumass*1e3)/((gas.culength*1e2)**2.)

        # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
        if bin_small_dust == 'Yes':
            frac[0] *= 5e3
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas")
            print("Mass fraction of bin 0 changed to: ", str(frac[0]))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # radial index corresponding to 0.3"
            imin = np.argmin(np.abs(gas.rmed-1.4))
            # radial index corresponding to 0.6"
            imax = np.argmin(np.abs(gas.rmed-2.8))
            dustcube[0, imin:imax, :] = gas.data[imin:imax, :] * ratio * frac[0] * \
                (gas.cumass*1e3)/((gas.culength*1e2)
                                ** 2.)  # dimensions: nbin, nrad, nsec

        print('Total dust mass [g] = ', np.sum(
            dustcube[:, :, :, :]*volume*(gas.culength*1e2)**2.))
        print('Total dust mass [Mgas] = ', np.sum(
            dustcube[:, :, :, :]*volume*(gas.culength*1e2)**2.)/(Mgas*gas.cumass*1e3))
        print('Total dust mass [Mstar] = ', np.sum(
            dustcube[:, :, :, :]*volume*(gas.culength*1e2)**2.)/(gas.cumass*1e3))

        # Total dust surface density
        dust_surface_density = np.sum(dustcube, axis=0)
        print(
            'Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())

    else:
        print("Set of initial conditions not implemented for pluto yet.")
        print("Use the following: RTdust_or_gas == 'dust' and recalc_density == 'Yes' and polarized_scat == 'No' and fargo3d == 'No'")
        sys.exit('I must exit!')

    # =========================
    # 3. Compute dust mass volume density for each size bin
    #    (vertical expansion assuming hydrostatic equilibrium)
    # =========================
    if RTdust_or_gas == 'dust' and recalc_density == 'Yes':
        print('--------- computing dust mass volume density ----------')

        DUSTOUT = open('dust_density.inp', 'w')
        DUSTOUT.write('1 \n')                           # iformat
        DUSTOUT.write(str(nrad*nsec*ncol)+' \n')        # n cells
        DUSTOUT.write(str(int(nbin))+' \n')             # nbin size bins

        # array (ncol, nbin, nrad, nsec)
        rhodustcube = np.zeros((nbin, nsec, ncol, nrad))

        # dust aspect ratio as function of ibin and r (or actually, R, cylindrical radius)
        hd = np.zeros((nbin, nrad))

        # gus aspect ratio
        hgas = np.zeros(nrad)
        for irad in range(nrad):
            hgas[irad] = (gas.rmed[irad]*np.cos(gas.tmed[:])*gas.data[0,:,irad]).sum(axis=0)/(gas.data[0,:,irad]).sum(axis=0)

        for ibin in range(nbin):
            if polarized_scat == 'No':
                # avg stokes number for that bin
                St = avgstokes[ibin]
            for irad in range(nrad):
                hd[ibin,irad] = hgas[irad]/np.sqrt(1.0 + St/alphaviscosity*(1.0 + 2.0*St)/(1.0 + St))

        # dust aspect ratio as function of ibin, r and phi (2D array for each size bin)
        hd2D = np.zeros((nbin, nsec, nrad))
        for th in range(nsec):
            hd2D[:, th, :] = hd    # nbin, nrad, nsec

        # grid radius function of ibin, r and phi (2D array for each size bin)
        r2D = np.zeros((nbin, nsec, nrad))
        for ibin in range(nbin):
            for th in range(nsec):
                r2D[ibin, th, :] = gas.rmed

        # work out exponential and normalization factors exp(-z^2 / 2H_d^2)
        # with z = r cos(theta) and H_d = h_d x R = h_d x r sin(theta)
        # r = spherical radius, R = cylindrical radius
        rhodustcube = dustcube
        rhodustcube = np.nan_to_num(rhodustcube)
        #for j in range(ncol):
            # ncol, nbin, nrad, nsec
        #    rhodustcube[j, :, :, :] = dustcube #* \
                #np.exp(-0.5*(np.cos(gas.tmed[j]) / hd2D)**2.0)
            # quantity is now in g / cm^3
        #    rhodustcube[j, :, :, :] /= (np.sqrt(2.*pi)
        #                                * r2D * hd2D * gas.culength*1e2)

        # for plotting purposes
        axirhodustcube = np.sum(rhodustcube, axis=3)/nsec  # ncol, nbin, nrad

        # Renormalize dust's mass volume density such that the sum over the 3D grid's volume of
        # the dust's mass volume density x the volume of each grid cell does give us the right
        # total dust mass, which equals ratio x Mgas.
        rhofield = np.sum(rhodustcube, axis=0)  # sum over dust bins

        Cedge, Aedge, Redge = np.meshgrid(
            gas.tedge, gas.pedge, gas.redge)   # ncol+1, nrad+1, Nsec+1

        r2 = Redge*Redge
        jacob = r2[:-1, :-1, :-1] * np.sin(Cedge[:-1, :-1, :-1])
        dphi = Aedge[1:, :-1, :-1] - Aedge[:-1, :-1, :-1]     # same as 2pi/nsec
        dr = Redge[:-1, :-1, 1:] - Redge[:-1, :-1, :-1]     # same as Rsup-Rinf
        dtheta = Cedge[:-1, 1:, :-1] - Cedge[:-1, :-1, :-1]
        # volume of a cell in cm^3
        vol = jacob * dr * dphi * dtheta * \
            ((gas.culength*1e2)**3)       # ncol, nrad, Nsec

        total_mass = np.sum(rhofield*vol)

        normalization_factor = ratio * Mgas * (gas.cumass*1e3) / total_mass
        rhodustcube = rhodustcube*normalization_factor
        print('total dust mass after vertical expansion [g] = ', np.sum(np.sum(
            rhodustcube, axis=0)*vol), ' as normalization factor = ', normalization_factor)

        # write mass volume densities for all size bins
        for ibin in range(nbin):
            print('dust species in bin', ibin, 'out of ', nbin-1)
            for k in range(nsec):
                for j in range(ncol):
                    for i in range(nrad):
                        DUSTOUT.write(str(rhodustcube[ibin, k, j, i])+' \n')

        # print max of dust's mass volume density at each colatitude
        for j in range(ncol):
            print('max(rho_dustcube) [g cm-3] for colatitude index j = ',
                j, ' = ', rhodustcube[:, :, j, :].max())

        DUSTOUT.close()

        # free RAM memory
        del rhodustcube, dustcube, dust, hd2D, r2D


    else:
        print('--------- I did not compute dust densities (recalc_density = No in params.dat file) ----------')


    # =========================
    # 4. Compute dust opacities
    # =========================
    if RTdust_or_gas == 'dust' and recalc_opac == 'Yes':
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
        optconstfile = os.path.expanduser(opacity_dir)+'/'+species+'.lnk'

        # The material density in gram / cm^3
        graindens = 2.0  # default density in g / cc
        if (species == 'mix_2species_porous' or species == 'mix_2species_porous_ice' or species == 'mix_2species_porous_ice70'):
            graindens = 0.1  # g / cc
        if (species == 'mix_2species' or species == 'mix_2species_60silicates_40ice'):
            graindens = 1.7  # g / cc
        if species == 'mix_2species_ice70':
            graindens = 1.26  # g / cc
        if species == 'mix_2species_60silicates_40carbons':
            graindens = 2.7  # g / cc

        # Set up a wavelength grid (in cm) upon which we want to compute the opacities
        # 1 micron -> 1 cm
        lamcm = 10.0**np.linspace(0, 4, 200)*1e-4

        # Set up an angular grid for which we want to compute the scattering matrix Z
        theta = np.linspace(0., 180., ntheta)

        for ibin in range(int(nbin)):
            # median grain size in cm in current bin size:
            if fargo3d == 'No':
                agraincm = 10.0**(0.5*(np.log10(1e2 *
                                bins[ibin]) + np.log10(1e2*bins[ibin+1])))
            else:
                agraincm = 1e2*dust_size[ibin]
            print('====================')
            print('bin ', ibin+1, '/', nbin)
            print('grain size [cm]: ', agraincm,
                ' with grain density [g/cc] = ', graindens)
            print('====================')
            pathout = species+str(ibin)
            opac = compute_opac_mie(optconstfile, graindens, agraincm, lamcm, theta=theta,
                                    extrapolate=extrapol, logawidth=logawidth, na=na,
                                    chopforward=chop, verbose=verbose)
            if (scat_mode >= 3):
                print("Writing dust opacities in dustkapscatmat* files")
                write_radmc3d_scatmat_file(opac, pathout)
            else:
                print("Writing dust opacities in dustkappa* files")
                write_radmc3d_kappa_file(opac, pathout)
    else:
        print('------- taking dustkap* opacity files in current directory (recalc_opac = No in params.dat file) ------ ')

    # Write dustopac.inp file even if we don't (re)calculate dust opacities
    if RTdust_or_gas == 'dust':
        write_dustopac(species, nbin)
        if plot_opac == 'Yes':
            print('--------- plotting dust opacities ----------')
            plot_opacities(species=species, amin=amin, amax=amax,
                        nbin=nbin, lbda1=wavelength*1e3)

    # write radmc3d script, particularly useful in case radmc3d mctherm /
    # ray_tracing are run on a different platform
    write_radmc3d_script()


    # =======================================
    # 5. Compute gas density and temperature (RT in gas lines)
    # =======================================
    if RTdust_or_gas == 'gas' and recalc_density == 'Yes':
        print('--------- computing gas mass volume density ----------')

        # nrad, nsec, quantity is in g / cm^2
        gascube = gas.data*(gas.cumass*1e3)/((gas.culength*1e2)**2.)

        # Artificially make a cavity devoid of gas (test)
        if cavity_gas == 'Yes':
            imin = np.argmin(np.abs(gas.rmed-0.9))
            axi_cav_gas = 0.0
            for j in range(nsec):
                axi_cav_gas += gascube[imin, j]
            axi_cav_gas /= nsec
            for i in range(nrad):
                if i < imin:
                    for j in range(nsec):
                        gascube[i, j] = axi_cav_gas * \
                            (gas.rmed[i]/gas.rmed[imin])**(2.0)

        GASOUT = open('numberdens_%s.inp' % gasspecies, 'w')
        GASOUT.write('1 \n')                           # iformat
        GASOUT.write(str(nrad*nsec*ncol)+' \n')        # n cells

        if lines_mode > 1:
            GASOUTH2 = open('numberdens_h2.inp', 'w')
            GASOUTH2.write('1 \n')                           # iformat
            GASOUTH2.write(str(nrad*nsec*ncol)+' \n')        # n cells

        # array (ncol, nrad, nsec)
        rhogascubeh2 = np.zeros((ncol, nrad, nsec))      # H2
        rhogascubeh2_cyl = np.zeros((gas.nver, nrad, nsec))
        rhogascube = np.zeros((ncol, nrad, nsec))      # gas species (eg, CO...)
        rhogascube_cyl = np.zeros((gas.nver, nrad, nsec))

        # gas aspect ratio as function of r (or actually, R, cylindrical radius)
        # gas aspect ratio (gas.rmed[i] = R in code units)
        hgas = aspectratio * (gas.rmed)**(flaringindex)

        # Set dust / gas temperature if Tdust_eq_Thydro == 'Yes' (currently default case)
        if Tdust_eq_Thydro == 'Yes':
            print('--------- Writing temperature.inp file (no mctherm) ----------')
            TEMPOUT = open('gas_temperature.inp', 'w')
            TEMPOUT.write('1 \n')                           # iformat
            TEMPOUT.write(str(nrad*nsec*ncol)+' \n')        # n cells
            gas_temp = np.zeros((ncol, nrad, nsec))
            gas_temp_cyl = np.zeros((gas.nver, nrad, nsec))
            thydro = aspectratio*aspectratio*gas.cutemp * \
                gas.rmed**(-1.0+2.0*flaringindex)
            for k in range(nsec):
                for j in range(gas.nver):
                    # only function of R (cylindrical radius)
                    gas_temp_cyl[j, :, k] = thydro
            # Now, sweep through the spherical grid
            for j in range(ncol):
                for i in range(nrad):
                    R = gas.rmed[i]*np.sin(gas.tmed[j])  # cylindrical radius
                    z = gas.rmed[i]*np.cos(gas.tmed[j])  # vertical altitude
                    icyl = np.argmin(np.abs(gas.rmed-R))
                    if R < gas.rmed[icyl] and icyl > 0:
                        icyl -= 1
                    jcyl = np.argmin(np.abs(gas.zmed-z))
                    if z < gas.zmed[jcyl] and jcyl > 0:
                        jcyl -= 1
                    if (icyl < nrad-1 and jcyl < gas.nver-1):
                        gas_temp[j, i, :] = (gas_temp_cyl[jcyl, icyl, :]*(gas.rmed[icyl+1]-R)*(gas.zmed[jcyl+1]-z) + gas_temp_cyl[jcyl+1, icyl, :]*(gas.rmed[icyl+1]-R)*(z-gas.zmed[jcyl]) + gas_temp_cyl[jcyl, icyl+1, :]*(
                            R-gas.rmed[icyl])*(gas.zmed[jcyl+1]-z) + gas_temp_cyl[jcyl+1, icyl+1, :]*(R-gas.rmed[icyl])*(z-gas.zmed[jcyl])) / ((gas.rmed[icyl+1]-gas.rmed[icyl]) * (gas.zmed[jcyl+1]-gas.zmed[jcyl]))
                    else:
                        # simple nearest-grid point interpolation...
                        gas_temp[j, i, :] = gas_temp_cyl[jcyl, icyl, :]
                    '''
                    icyl = np.argmin(np.abs(gas.rmed-R))
                    jcyl = np.argmin(np.abs(gas.zmed-z))
                    gas_temp[j,i,:] = gas_temp_cyl[jcyl,icyl,:]  # simple nearest-grid point interpolation...
                    '''
            for k in range(nsec):
                for j in range(ncol):
                    for i in range(nrad):
                        TEMPOUT.write(str(gas_temp[j, i, k])+' \n')
            TEMPOUT.close()

        # gas aspect ratio and grid radius as function of r and phi (2D array for each size bin)
        hg2D = np.zeros((nrad, nsec))
        r2D = np.zeros((nrad, nsec))
        for th in range(nsec):
            hg2D[:, th] = hgas     # nrad, nsec
            r2D[:, th] = gas.rmed  # nrad, nsec

        # work out vertical expansion. First, for the array in cylindrical coordinates
        for j in range(gas.nver):
            # nver, nrad, nsec
            rhogascubeh2_cyl[j, :, :] = gascube * \
                np.exp(-0.5*(gas.zmed[j]/hg2D/r2D)**2.0)
            # quantity is now in cm^-3
            rhogascubeh2_cyl[j, :, :] /= (np.sqrt(2.*pi)
                                        * r2D * hg2D * gas.culength*1e2 * 2.3*mp)

        # multiply by constant abundance ratio
        rhogascube_cyl = rhogascubeh2_cyl*abundance

        # Simple model for photodissociation: drop
        # number density by 5 orders of magnitude if 1/2
        # erfc(z/sqrt(2)H) x Sigma_gas / mu m_p < 1e21 cm^-2
        if (photodissociation == 'Yes' or freezeout == 'Yes'):
            for k in range(nsec):
                for j in range(gas.nver):
                    for i in range(nrad):
                        # all relevant quantities below are in in cgs
                        chip = 0.5 * \
                            math.erfc(gas.zmed[j]/np.sqrt(2.0)/hgas[i] /
                                    gas.rmed[i]) * gascube[i, k] / (2.3*mp)
                        chim = 0.5 * \
                            math.erfc(-gas.zmed[j]/np.sqrt(2.0)/hgas[i] /
                                    gas.rmed[i]) * gascube[i, k] / (2.3*mp)
                        if (photodissociation == 'Yes' and (chip < 1e21 or chim < 1e21)):
                            rhogascube_cyl[j, i, k] *= 1e-5
                    # Simple modelling of freezeout: drop CO abundance
                        if freezeout == 'Yes' and gas_temp_cyl[j, i, k] < 19.0:
                            rhogascube_cyl[j, i, k] *= 1e-5

        # then, sweep through the spherical grid
        for j in range(ncol):
            for i in range(nrad):
                R = gas.rmed[i]*np.sin(gas.tmed[j])  # cylindrical radius
                z = gas.rmed[i]*np.cos(gas.tmed[j])  # vertical altitude
                icyl = np.argmin(np.abs(gas.rmed-R))
                if R < gas.rmed[icyl] and icyl > 0:
                    icyl -= 1
                jcyl = np.argmin(np.abs(gas.zmed-z))
                if z < gas.zmed[jcyl] and jcyl > 0:
                    jcyl -= 1
                if (icyl < nrad-1 and jcyl < gas.nver-1):
                    rhogascube[j, i, :] = (rhogascube_cyl[jcyl, icyl, :]*(gas.rmed[icyl+1]-R)*(gas.zmed[jcyl+1]-z) + rhogascube_cyl[jcyl+1, icyl, :]*(gas.rmed[icyl+1]-R)*(z-gas.zmed[jcyl]) + rhogascube_cyl[jcyl, icyl+1, :]*(
                        R-gas.rmed[icyl])*(gas.zmed[jcyl+1]-z) + rhogascube_cyl[jcyl+1, icyl+1, :]*(R-gas.rmed[icyl])*(z-gas.zmed[jcyl])) / ((gas.rmed[icyl+1]-gas.rmed[icyl]) * (gas.zmed[jcyl+1]-gas.zmed[jcyl]))
                    rhogascubeh2[j, i, :] = (rhogascubeh2_cyl[jcyl, icyl, :]*(gas.rmed[icyl+1]-R)*(gas.zmed[jcyl+1]-z) + rhogascubeh2_cyl[jcyl+1, icyl, :]*(gas.rmed[icyl+1]-R)*(z-gas.zmed[jcyl]) + rhogascubeh2_cyl[jcyl, icyl+1, :]*(
                        R-gas.rmed[icyl])*(gas.zmed[jcyl+1]-z) + rhogascubeh2_cyl[jcyl+1, icyl+1, :]*(R-gas.rmed[icyl])*(z-gas.zmed[jcyl])) / ((gas.rmed[icyl+1]-gas.rmed[icyl]) * (gas.zmed[jcyl+1]-gas.zmed[jcyl]))
                else:
                    # simple nearest-grid point interpolation...
                    rhogascube[j, i, :] = rhogascube_cyl[jcyl, icyl, :]
                    rhogascubeh2[j, i, :] = rhogascubeh2_cyl[jcyl, icyl, :]

        print('--------- Writing numberdens.inp file ----------')
        for k in range(nsec):
            for j in range(ncol):
                for i in range(nrad):
                    GASOUT.write(str(rhogascube[j, i, k])+' \n')
                    if lines_mode > 1:
                        GASOUTH2.write(str(rhogascubeh2[j, i, k])+' \n')
        GASOUT.close()
        if lines_mode > 1:
            GASOUTH2.close()

        # plot azimuthally-averaged density vs. radius and colatitude
        if plot_density == 'Yes':
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.ticker as ticker
            from matplotlib.ticker import (
                MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
            matplotlib.rcParams.update({'font.size': 20})
            matplotlib.rc('font', family='Arial')
            fontcolor = 'white'
            print('--------- plotting density(R,theta) ----------')
            axidensh2 = np.zeros((ncol, nrad))
            axidens = np.zeros((ncol, nrad))
            for j in range(ncol):
                for i in range(nrad):
                    for k in range(nsec):
                        axidens[j, i] += rhogascube[j, i, k]
                        axidensh2[j, i] += rhogascubeh2[j, i, k]
                    axidens[j, i] /= (nsec+0.0)
                    axidensh2[j, i] /= (nsec+0.0)
            X = gas.rmed*gas.culength/1.5e11  # in au
            Y = gas.tmed
            fig = plt.figure(figsize=(8., 8.))
            plt.subplots_adjust(left=0.12, right=0.94, top=0.88, bottom=0.11)
            ax = plt.gca()
            ax.set_ylabel('latitude [rad]')
            ax.set_xlabel('radius [code units]')
            ax.tick_params(top='on', right='on', length=5,
                        width=1.0, direction='out')
            ax.tick_params(axis='x', which='minor', top=True)
            ax.tick_params(axis='y', which='minor', right=True)
            mynorm = matplotlib.colors.LogNorm()
            vmin = axidens.min()
            vmax = axidens.max()
            CF = ax.pcolormesh(X, Y, axidens, cmap=mycolormap,
                            vmin=vmin, vmax=vmax, norm=mynorm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="2.5%", pad=0.12)
            cb = plt.colorbar(CF, cax=cax, orientation='horizontal')
            cax.xaxis.tick_top()
            cax.xaxis.set_tick_params(labelsize=20, direction='out')
            cax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
            # title on top
            cax.xaxis.set_label_position('top')
            if gasspecies == 'co':
                strgas = r'$^{12}$CO'
            elif gasspecies == '13co':
                strgas = r'$^{13}$CO'
            elif gasspecies == 'c17o':
                strgas = r'C$^{17}$O'
            elif gasspecies == 'c18o':
                strgas = r'C$^{18}$O'
            else:
                strgas = gasspecies
            cax.set_xlabel(strgas+' abundance '+r'[cm$^{-3}$]')
            cax.xaxis.labelpad = 8
            fileout = 'density.pdf'
            plt.savefig('./'+fileout, dpi=160)
            # plot h2 number density if non-LTE transfer
            if lines_mode > 1:
                fig = plt.figure(figsize=(8., 8.))
                plt.subplots_adjust(left=0.12, right=0.94, top=0.88, bottom=0.11)
                ax = plt.gca()
                CF = ax.pcolormesh(X, Y, axidensh2, cmap=mycolormap,
                                vmin=axidensh2.min(), vmax=axidensh2.max(), norm=mynorm)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("top", size="2.5%", pad=0.12)
                cb = plt.colorbar(CF, cax=cax, orientation='horizontal')
                cax.xaxis.tick_top()
                cax.xaxis.set_tick_params(labelsize=20, direction='out')
                cax.xaxis.set_major_locator(
                    ticker.LogLocator(base=10.0, numticks=10))
                cax.xaxis.set_label_position('top')
                cax.set_xlabel('H2 abundance '+r'[cm$^{-3}$]')
                cax.xaxis.labelpad = 8
                fileout = 'densityh2.pdf'
                plt.savefig('./'+fileout, dpi=160)

        # print max of gas mass volume density at each colatitude
        for j in range(ncol):
            print('max(rho_dustcube) for gas species at colatitude index j = ',
                j, ' = ', rhogascube[j, :, :].max())

        # free RAM memory
        del rhogascube, rhogascubeh2, rhogascube_cyl, rhogascubeh2_cyl

        # Microturbulent line broadening
        MTURB = open('microturbulence.inp', 'w')
        MTURB.write('1 \n')                           # iformat
        MTURB.write(str(nrad*nsec*ncol)+' \n')        # n cells
        microturb = np.zeros((ncol, nrad, nsec))

        if turbvel == 'cavity':
            for k in range(nsec):
                for j in range(ncol):
                    for i in range(nrad):
                        if gas.rmed[i] < 1.0:
                            myalpha = 3e-2   # inside cavity
                        else:
                            myalpha = 3e-4   # outside cavity
                        # v_turb ~ sqrt(alpha) x isothermal sound speed
                        microturb[j, i, k] = np.sqrt(
                            myalpha * kB * gas_temp[j, i, k] / 2.3 / mp)
                        MTURB.write(str(microturb[j, i, k])+' \n')
            print('min and max of microturbulent velocity in cm/s = ',
                microturb.min(), microturb.max())
        else:
            for k in range(nsec):
                for j in range(ncol):
                    for i in range(nrad):
                        # ncol, nrad, nsec in cm/s
                        microturb[j, i, k] = turbvel*1.0e2
                        MTURB.write(str(microturb[j, i, k])+' \n')
        MTURB.close()

        # gas velocity field
        if fargo3d == 'Yes':
            vtheta3D = Field(field='gasvz'+str(on)+'.dat',
                            directory=dir).data  # code units
            vtheta3D *= (gas.culength*1e2)/(gas.cutime)  # cm/s
            vrad3D = Field(field='gasvy'+str(on)+'.dat',
                        directory=dir).data  # code units
            vrad3D *= (gas.culength*1e2)/(gas.cutime)  # cm/s
            vphi3D = Field(field='gasvx'+str(on)+'.dat',
                        directory=dir).data  # code units
            f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(
                dir+"/planet0.dat", unpack=True)
            omegaframe = omega[on]
            print('OMEGAFRAME = ', omegaframe)
            for theta in range(ncol):
                for phi in range(nsec):
                    vphi3D[theta, :, phi] += gas.rmed*omegaframe
            vphi3D *= (gas.culength*1e2)/(gas.cutime)  # cm/s
        else:
            vrad3D_cyl = np.zeros((gas.nver, nrad, nsec))   # zeros!
            vphi3D_cyl = np.zeros((gas.nver, nrad, nsec))   # zeros!
            vtheta3D = np.zeros((ncol, nrad, nsec))   # zeros!
            vrad3D = np.zeros((ncol, nrad, nsec))   # zeros!
            vphi3D = np.zeros((ncol, nrad, nsec))   # zeros!
            vrad2D = Field(field='gasvrad'+str(on)+'.dat',
                        directory=dir).data  # code units
            vrad2D *= (gas.culength*1e2)/(gas.cutime)  # cm/s
            vphi2D = Field(field='gasvtheta'+str(on)+'.dat',
                        directory=dir).data  # code units
            f1, xpla, ypla, f4, f5, f6, f7, date, omega, f10, f11 = np.loadtxt(
                dir+"/planet0.dat", unpack=True)
            omegaframe = omega[on]
            print('OMEGAFRAME = ', omegaframe)
            for phi in range(nsec):
                vphi2D[:, phi] += gas.rmed*omegaframe
            vphi2D *= (gas.culength*1e2)/(gas.cutime)  # cm/s
            # Vertical expansion for vrad and vphi (vtheta being assumed zero)
            for z in range(gas.nver):
                vrad3D_cyl[z, :, :] = vrad2D
                vphi3D_cyl[z, :, :] = vphi2D
            # Now, sweep through the spherical grid
            for j in range(ncol):
                for i in range(nrad):
                    R = gas.rmed[i]*np.sin(gas.tmed[j])  # cylindrical radius
                    z = gas.rmed[i]*np.cos(gas.tmed[j])  # vertical altitude
                    icyl = np.argmin(np.abs(gas.rmed-R))
                    if R < gas.rmed[icyl] and icyl > 0:
                        icyl -= 1
                    jcyl = np.argmin(np.abs(gas.zmed-z))
                    if z < gas.zmed[jcyl] and jcyl > 0:
                        jcyl -= 1
                    if (icyl < nrad-1 and jcyl < gas.nver-1):
                        vrad3D[j, i, :] = (vrad3D_cyl[jcyl, icyl, :]*(gas.rmed[icyl+1]-R)*(gas.zmed[jcyl+1]-z) + vrad3D_cyl[jcyl+1, icyl, :]*(gas.rmed[icyl+1]-R)*(z-gas.zmed[jcyl]) + vrad3D_cyl[jcyl, icyl+1, :]*(
                            R-gas.rmed[icyl])*(gas.zmed[jcyl+1]-z) + vrad3D_cyl[jcyl+1, icyl+1, :]*(R-gas.rmed[icyl])*(z-gas.zmed[jcyl])) / ((gas.rmed[icyl+1]-gas.rmed[icyl]) * (gas.zmed[jcyl+1]-gas.zmed[jcyl]))
                        vphi3D[j, i, :] = (vphi3D_cyl[jcyl, icyl, :]*(gas.rmed[icyl+1]-R)*(gas.zmed[jcyl+1]-z) + vphi3D_cyl[jcyl+1, icyl, :]*(gas.rmed[icyl+1]-R)*(z-gas.zmed[jcyl]) + vphi3D_cyl[jcyl, icyl+1, :]*(
                            R-gas.rmed[icyl])*(gas.zmed[jcyl+1]-z) + vphi3D_cyl[jcyl+1, icyl+1, :]*(R-gas.rmed[icyl])*(z-gas.zmed[jcyl])) / ((gas.rmed[icyl+1]-gas.rmed[icyl]) * (gas.zmed[jcyl+1]-gas.zmed[jcyl]))
                    else:
                        # simple nearest-grid point interpolation...
                        vrad3D[j, i, :] = vrad3D_cyl[jcyl, icyl, :]
                        vphi3D[j, i, :] = vphi3D_cyl[jcyl, icyl, :]

        GASVEL = open('gas_velocity.inp', 'w')
        GASVEL.write('1 \n')                           # iformat
        GASVEL.write(str(nrad*nsec*ncol)+' \n')        # n cells
        for k in range(nsec):
            for j in range(ncol):
                for i in range(nrad):
                    GASVEL.write(
                        str(vrad3D[j, i, k])+' '+str(vtheta3D[j, i, k])+' '+str(vphi3D[j, i, k])+' \n')
        GASVEL.close()

        del vrad3D, vphi3D, vtheta3D, vrad3D_cyl, vphi3D_cyl


    # =========================
    # 6. Call to RADMC3D thermal solution and ray tracing
    # =========================
    if (recalc_radmc == 'Yes' or recalc_rawfits == 'Yes'):
        # Write other parameter files required by RADMC3D
        print('--------- printing auxiliary files ----------')

        # need to check why we need to output wavelength...
        if recalc_rawfits == 'No':
            write_wavelength()
            write_stars(Rstar=rstar, Tstar=teff)
            # Write 3D spherical grid for RT computational calculation
            write_AMRgrid(gas, Plot=False)
            if RTdust_or_gas == 'gas':
                write_lines(str(gasspecies), lines_mode)
            # rto_style = 3 means that RADMC3D will write binary output files
            # setthreads corresponds to the number of threads (cores) over which radmc3d runs
            write_radmc3dinp(incl_dust=incl_dust, incl_lines=incl_lines, lines_mode=lines_mode, nphot_scat=nb_photons_scat, nphot=nb_photons,
                            rto_style=3, tgas_eq_tdust=tgas_eq_tdust, modified_random_walk=1, scattering_mode_max=scat_mode, setthreads=nbcores)

        # Add 90 degrees to position angle so that RADMC3D's definition of
        # position angle be consistent with observed position
        # angle, which is what we enter in the params.dat file
        M = RTmodel(distance=distance, Lambda=wavelength*1e3, label=label, line=gasspecies, iline=iline,
                    vkms=vkms, widthkms=widthkms, npix=nbpixels, phi=phiangle, incl=inclination, posang=posangle+90.0)

        # Set dust / gas temperature if Tdust_eq_Thydro == 'Yes'
        if recalc_rawfits == 'No' and Tdust_eq_Thydro == 'Yes' and RTdust_or_gas == 'dust' and recalc_temperature == 'Yes':
            print('--------- Writing temperature file (no mctherm) ----------')
            os.system('rm -f dust_temperature.bdat')        # avoid confusion!...
            TEMPOUT = open('dust_temperature.dat', 'w')
            TEMPOUT.write('1 \n')                           # iformat
            TEMPOUT.write(str(nrad*nsec*ncol)+' \n')        # n cells
            TEMPOUT.write(str(int(nbin))+' \n')             # nbin size bins

            gas_temp = np.zeros((ncol, nrad, nsec))
            thydro = aspectratio*aspectratio*gas.cutemp * \
                gas.rmed**(-1.0+2.0*flaringindex)
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
            if RTdust_or_gas == 'gas' and Tdust_eq_Thydro == 'Yes':
                del gas_temp_cyl

        # Now run RADMC3D
        if recalc_rawfits == 'No':
            print('--------- Now executing RADMC3D ----------')
            os.system('./script_radmc')

        print('--------- exporting results in fits format ----------')
        outfile = exportfits(M)

        if plot_temperature == 'Yes':
            # Plot midplane and surface temperature profiles
            if RTdust_or_gas == 'dust':
                Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
                Temp = Temp[4:]
                Temp = Temp.reshape(nbin, nsec, ncol, nrad)
                # Keep temperature of the largest dust species
                Temp = Temp[-1, :, :, :]
            else:
                f = open('gas_temperature.inp', 'r')
                Temp = np.genfromtxt(f)
                Temp = Temp[2:]
                Temp = Temp.reshape(nsec, ncol, nrad)
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
            S = gas.rmed*gas.culength/1.5e11  # radius in a.u.
            # gas temperature in hydro simulation in Kelvin (assuming T in R^-1/2, no matter
            # the value of the gas flaring index in the simulation)
            Tm_model = aspectratio*aspectratio * \
                gas.cutemp*gas.rmed**(-1.0+2.0*flaringindex)
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
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.legend(frameon=False)
            fig.add_subplot(ax)
            filenameT = 'T_R_'+label+'.pdf'
            fig.savefig(filenameT, dpi=180, bbox_inches='tight')
            fig.clf()
            # Save radial profiles in an ascii file
            filenameT2 = 'T_R_'+label+'.dat'
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

        if RTdust_or_gas == 'gas':
            # no need for lambda in file's name for gas RT calculations...
            outfile = 'image_'+str(label)+'_i'+str(inclination) + \
                '_phi'+str(phiangle)+'_PA'+str(posangle)
        if RTdust_or_gas == 'dust':
            outfile = 'image_'+str(label)+'_lbda'+str(wavelength)+'_i' + \
                str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)

        if secondorder == 'Yes':
            outfile = outfile+'_so'
        if dustdens_eq_gasdens == 'Yes':
            outfile = outfile+'_ddeqgd'
        if bin_small_dust == 'Yes':
            outfile = outfile+'_bin0'

        outfile = outfile+'.fits'


    # =========================
    # 7. Convolve raw flux with beam and produce final image
    # =========================
    if recalc_fluxmap == 'Yes':
        print('--------- Convolving and writing final image ----------')

        f = fits.open('./'+outfile)

        # remove .fits extension
        outfile = os.path.splitext(outfile)[0]

        # add moment order information if gas
        if RTdust_or_gas == 'gas' and widthkms > 0.0:
            outfile = outfile + '_moment'+str(moment_order)
        # add bmaj information
        outfile = outfile + '_bmaj'+str(bmaj)

        outfile = outfile+'.fits'

        hdr = f[0].header
        # pixel size converted from degrees to arcseconds
        cdelt = np.abs(hdr['CDELT1']*3600.)

        # get wavelength and convert it from microns to mm
        lbda0 = hdr['LBDAMIC']*1e-3

        # a) case with no polarized scattering: fits file directly contains raw intensity field
        if polarized_scat == 'No':
            nx = hdr['NAXIS1']
            ny = hdr['NAXIS2']
            raw_intensity = f[0].data
            if (recalc_radmc == 'No' and plot_tau == 'No'):
                # sum over pixels
                print("Total flux [Jy] = "+str(np.sum(raw_intensity)))
            # check beam is correctly handled by inserting a source point at the
            # origin of the raw intensity image
            if check_beam == 'Yes':
                raw_intensity[:, :] = 0.0
                raw_intensity[nx//2-1, ny//2-1] = 1.0
            # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
            if (add_noise == 'Yes' and RTdust_or_gas == 'dust' and plot_tau == 'No'):
                # beam area in pixel^2
                beam = (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
                # noise standard deviation in Jy per pixel (I've checked the expression below works well)
                noise_dev_std_Jy_per_pixel = noise_dev_std / \
                    np.sqrt(0.5*beam)  # 1D
                # noise array
                noise_array = np.random.normal(
                    0.0, noise_dev_std_Jy_per_pixel, size=nbpixels*nbpixels)
                noise_array = noise_array.reshape(nbpixels, nbpixels)
                raw_intensity += noise_array
            if brightness_temp == 'Yes':
                # beware that all units are in cgs! We need to convert
                # 'intensity' from Jy/pixel to cgs units!
                # pixel size in each direction in cm
                pixsize_x = cdelt*distance*au
                pixsize_y = pixsize_x
                # solid angle subtended by pixel size
                pixsurf_ster = pixsize_x*pixsize_y/distance/distance/pc/pc
                # convert intensity from Jy/pixel to erg/s/cm2/Hz/sr
                intensity_buf = raw_intensity/1e23/pixsurf_ster
                # beware that lbda0 is in mm right now, we need to have it in cm in the expression below
                raw_intensity = (h*c/kB/(lbda0*1e-1))/np.log(1. +
                                                            2.*h*c/intensity_buf/pow((lbda0*1e-1), 3.))
                #raw_intensity = np.nan_to_num(raw_intensity)

        # b) case with polarized scattering: fits file contains raw Stokes vectors
        if RTdust_or_gas == 'dust' and polarized_scat == 'Yes':
            cube = f[0].data
            Q = cube[1, :, :]
            U = cube[2, :, :]
            #I = cube[0,:,:]
            #P = cube[4,:,:]
            (nx, ny) = Q.shape
            # define theta angle for calculation of Q_phi below (Avenhaus+ 14)
            x = np.arange(1, nx+1)
            y = np.arange(1, ny+1)
            XXs, YYs = np.meshgrid(x, y)
            X0 = nx/2-1
            Y0 = ny/2-1
            rrs = np.sqrt((XXs-X0)**2+(YYs-Y0)**2)
            theta = np.arctan2(-(XXs-X0), (YYs-Y0))  # notice atan(x/y)
            if add_noise == 'Yes':
                # add noise to Q and U Stokes arrays
                # noise array
                noise_array_Q = np.random.normal(
                    0.0, 0.01*Q.max(), size=nbpixels*nbpixels)
                noise_array_Q = noise_array_Q.reshape(nbpixels, nbpixels)
                Q += noise_array_Q
                noise_array_U = np.random.normal(
                    0.0, 0.01*U.max(), size=nbpixels*nbpixels)
                noise_array_U = noise_array_U.reshape(nbpixels, nbpixels)
                U += noise_array_U
            # add mask in polarized intensity Qphi image if mask_radius != 0
            if mask_radius != 0.0:
                pillbox = np.ones((nx, ny))
                imaskrad = mask_radius/cdelt  # since cdelt is pixel size in arcseconds
                pillbox[np.where(rrs < imaskrad)] = 0.

        # ------------
        # smooth image
        # ------------
        # beam area in pixel^2
        beam = (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
        # stdev lengths in pixel
        stdev_x = (bmaj/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
        stdev_y = (bmin/(2.*np.sqrt(2.*np.log(2.)))) / cdelt

        # a) case with no polarized scattering
        if (polarized_scat == 'No' and plot_tau == 'No'):
            # Call to Gauss_filter function
            if moment_order != 1:
                smooth = Gauss_filter(raw_intensity, stdev_x,
                                    stdev_y, bpaangle, Plot=False)
            else:
                smooth = raw_intensity

            # convert image from Jy/pixel to mJy/beam or microJy/beam
            # could be refined...
            if brightness_temp == 'Yes':
                convolved_intensity = smooth
            if brightness_temp == 'No':
                convolved_intensity = smooth * 1e3 * beam   # mJy/beam

            strflux = 'Flux of continuum emission (mJy/beam)'
            if gasspecies == 'co':
                strgas = r'$^{12}$CO'
            elif gasspecies == '13co':
                strgas = r'$^{13}$CO'
            elif gasspecies == 'c17o':
                strgas = r'C$^{17}$O'
            elif gasspecies == 'c18o':
                strgas = r'C$^{18}$O'
            elif gasspecies == 'hco+':
                strgas = r'HCO+'
            elif gasspecies == 'so':
                strgas = r'SO'
            else:
                strgas = gasspecies
            if gasspecies != 'so':
                strgas += r' ($%d \rightarrow %d$)' % (iline, iline-1)
            if gasspecies == 'so' and iline == 14:
                strgas += r' ($5_6 \rightarrow 4_5$)'

            if brightness_temp == 'Yes':
                # Gas RT and a single velocity channel
                if RTdust_or_gas == 'gas' and widthkms == 0.0:
                    strflux = strgas+' brightness temperature (K)'
                # Gas RT and mooment order 0 map
                if RTdust_or_gas == 'gas' and moment_order == 0 and widthkms != 0.0:
                    strflux = strgas+' integrated brightness temperature (K km/s)'
                if RTdust_or_gas == 'dust':
                    strflux = r'Brightness temperature (K)'
            else:
                # Gas RT and a single velocity channel
                if RTdust_or_gas == 'gas' and widthkms == 0.0:
                    strflux = strgas+' intensity (mJy/beam)'
                # Gas RT and mooment order 0 map
                if RTdust_or_gas == 'gas' and moment_order == 0 and widthkms != 0.0:
                    strflux = strgas+' integrated intensity (mJy/beam km/s)'
                if convolved_intensity.max() < 1.0:
                    convolved_intensity = smooth * 1e6 * beam   # microJy/beam
                    strflux = r'Flux of continuum emission ($\mu$Jy/beam)'
                    # Gas RT and a single velocity channel
                    if RTdust_or_gas == 'gas' and widthkms == 0.0:
                        strflux = strgas+' intensity ($\mu$Jy/beam)'
                    if RTdust_or_gas == 'gas' and moment_order == 0 and widthkms != 0.0:
                        strflux = strgas + \
                            ' integrated intensity ($\mu$Jy/beam km/s)'
            #
            if RTdust_or_gas == 'gas' and moment_order == 1:
                convolved_intensity = smooth
                # this is actually 'raw_intensity' since for moment 1 maps
                # the intensity in each channel map is already convolved,
                # so that we do not convolve a second time!...
                strflux = strgas+' velocity (km/s)'

        #
        if plot_tau == 'Yes':
            convolved_intensity = raw_intensity
            strflux = r'Absorption optical depth $\tau'

        # b) case with polarized scattering
        if RTdust_or_gas == 'dust' and polarized_scat == 'Yes':
            Q_smooth = Gauss_filter(Q, stdev_x, stdev_y, bpaangle, Plot=False)
            U_smooth = Gauss_filter(U, stdev_x, stdev_y, bpaangle, Plot=False)
            if mask_radius != 0.0:
                pillbox_smooth = Gauss_filter(
                    pillbox, stdev_x, stdev_y, bpaangle, Plot=False)
                Q_smooth *= pillbox_smooth
                U_smooth *= pillbox_smooth
            Q_phi = Q_smooth * np.cos(2*theta) + U_smooth * np.sin(2*theta)
            convolved_intensity = Q_phi
            strflux = 'Polarized intensity [arb. units]'

        # -------------------------------------
        # SP: save convolved flux map solution to fits
        # -------------------------------------
        hdu = fits.PrimaryHDU()
        hdu.header['BITPIX'] = -32
        hdu.header['NAXIS'] = 2  # 2
        hdu.header['NAXIS1'] = nbpixels
        hdu.header['NAXIS2'] = nbpixels
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
        hdu.header['CRPIX1'] = float((nbpixels+1.)/2.)
        hdu.header['CRPIX2'] = float((nbpixels+1.)/2.)
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
        if add_noise == 'Yes':
            substr = '_wn'+str(noise_dev_std)+'_JyBeam.fits'
            jybeamfileout = re.sub('.fits', substr, inbasename)
        else:
            jybeamfileout = re.sub('.fits', '_JyBeam.fits', inbasename)
        if polarized_scat == 'Yes':
            jybeamfileout = 'Qphi.fits'
        hdu.writeto(jybeamfileout, overwrite=True)

        # ----------------------------
        # if polarised imaging, first de-project Qphi image to multiply by R^2
        # then re-project back
        # ----------------------------
        if RTdust_or_gas == 'dust' and polarized_scat == 'Yes':
            hdu0 = fits.open(jybeamfileout)
            hdr0 = hdu0[0].header
            nx = int(hdr0['NAXIS1'])
            ny = nx
            if ((nx % 2) == 0):
                nx = nx+1
                ny = ny+1
            hdr1 = deepcopy(hdr0)
            hdr1['NAXIS1'] = nx
            hdr1['NAXIS2'] = ny
            hdr1['CRPIX1'] = (nx+1)/2
            hdr1['CRPIX2'] = (ny+1)/2

            # slightly modify original image such that centre is at middle of image -> odd number of cells
            image_centered = gridding(jybeamfileout, hdr1, fullWCS=False)
            fileout_centered = re.sub('.fits', 'centered.fits', jybeamfileout)
            fits.writeto(fileout_centered, image_centered, hdr1, overwrite=True)

            # rotate original, centred image by position angle (posangle)
            image_rotated = ndimage.rotate(image_centered, posangle, reshape=False)
            fileout_rotated = re.sub('.fits', 'rotated.fits', jybeamfileout)
            fits.writeto(fileout_rotated, image_rotated, hdr1, overwrite=True)
            hdr2 = deepcopy(hdr1)
            cosi = np.cos(inclination_input*np.pi/180.)
            hdr2['CDELT1'] = hdr2['CDELT1']*cosi

            # Then deproject with inclination via gridding interpolation function and hdr2
            image_stretched = gridding(fileout_rotated, hdr2)

            # rescale stretched image by r^2
            nx = hdr2['NAXIS1']
            ny = hdr2['NAXIS2']
            cdelt = abs(hdr2['CDELT1']*3600)  # in arcseconds
            (x0, y0) = (nx/2, ny/2)
            mymax = 0.0
            for j in range(nx):
                for k in range(ny):
                    dx = (j-x0)*cdelt
                    dy = (k-y0)*cdelt
                    rad = np.sqrt(dx*dx + dy*dy)
                    image_stretched[j, k] *= (rad*rad)
                    if (image_stretched[j, k] > mymax):
                        # cuidadin!
                        # if (rad <= truncation_radius and image_stretched[j,k] > mymax):
                        mymax = image_stretched[j, k]
                    # else:
                    #    image_stretched[j,k] = 0.0

            # Normalize PI intensity
            image_stretched /= mymax
            fileout_stretched = re.sub('.fits', 'stretched.fits', jybeamfileout)
            fits.writeto(fileout_stretched, image_stretched, hdr2, overwrite=True)

            # Then deproject via gridding interpolatin function and hdr1
            image_destretched = gridding(fileout_stretched, hdr1)

            # and finally de-rotate by -position angle
            final_image = ndimage.rotate(
                image_destretched, -posangle, reshape=False)

            # save final fits
            inbasename = os.path.basename('./'+outfile)
            if add_noise == 'Yes':
                substr = '_wn'+str(noise_dev_std)+'_JyBeam.fits'
                jybeamfileout = re.sub('.fits', substr, inbasename)
            else:
                jybeamfileout = re.sub('.fits', '_JyBeam.fits', inbasename)
            fits.writeto(jybeamfileout, final_image, hdr1, clobber=True)
            convolved_intensity = final_image
            os.system('rm -f Qphi*.fits')

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
        if ((nx % 2) == 0):
            dpix = 0.5
        else:
            dpix = 0.0
        dpix = 0.0
        a0 = cdelt*(nx//2.-dpix)   # >0
        a1 = -cdelt*(nx//2.+dpix)  # <0
        d0 = -cdelt*(nx//2.-dpix)  # <0
        d1 = cdelt*(nx//2.+dpix)   # >0
        # da positive definite
        if (minmaxaxis < abs(a0)):
            da = minmaxaxis
        else:
            da = maximum(abs(a0), abs(a1))
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
        if RTdust_or_gas == 'dust' and polarized_scat == 'Yes':
            min = 0.0
            max = 1.0  # cuidadin 1.0
        if RTdust_or_gas == 'gas' and moment_order == 1:
            min = -6.0
            max = 6.0
        CM = ax.imshow(convolved_intensity, origin='lower', cmap=mycolormap,
                    interpolation='bilinear', extent=[a0, a1, d0, d1], vmin=min, vmax=max, aspect='auto')
        '''
        X = a0 + (a1-a0)*arange(nx)/(nx-1.0)
        Y = d0 + (d1-d0)*arange(ny)/(ny-1.0)
        CM = ax.pcolormesh(X,Y,convolved_intensity,cmap=mycolormap,vmin=min,vmax=max)
        '''

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

        # Add a few contours in order 1 moment maps for gas emission
        if RTdust_or_gas == 'gas' and moment_order == 1:
            ax.contour(convolved_intensity, levels=10, color='black',
                    linestyles='-', origin='lower', extent=[a0, a1, d0, d1])

        # plot beam
        if plot_tau == 'No':
            from matplotlib.patches import Ellipse
            e = Ellipse(xy=[xlambda, dmin+0.166*da], width=bmin,
                        height=bmaj, angle=bpaangle+90.0)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('white')
            e.set_alpha(1.0)
            ax.add_artist(e)
        # plot beam
        '''
        if check_beam == 'Yes':
            from matplotlib.patches import Ellipse
            e = Ellipse(xy=[0.0,0.0], width=bmin, height=bmaj, angle=bpaangle+90.0)
            e.set_clip_box(ax.bbox)
            e.set_facecolor('white')
            e.set_alpha(1.0)
            ax.add_artist(e)
        '''

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
        if deproj_polar == 'Yes':
            currentdir = os.getcwd()
            alpha_min = 0.          # deg, PA of offset from the star
            Delta_min = 0.          # arcsec, amplitude of offset from the star
            RA = 0.0  # if input image is a prediction, star should be at the center
            DEC = 0.0  # note that this deprojection routine works in WCS coordinates
            cosi = np.cos(inclination_input*np.pi/180.)

            print('deprojection around PA [deg] = ', posangle)
            print('and inclination [deg] = ', inclination_input)

            # makes a new directory "deproj_polar_dir" and calculates a number
            # of products: copy of the input image [_fullim], centered at
            # (RA,DEC) [_centered], deprojection by cos(i) [_stretched], polar
            # image [_polar], etc. Also, a _radial_profile which is the
            # average radial intensity.
            exec_polar_expansions(jybeamfileout, 'deproj_polar_dir', posangle, cosi, RA=RA, DEC=DEC,
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
            if xaxisflip == 'Yes':
                jshift = int(nbpixels/2)
            else:
                jshift = int(nbpixels/2)
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
            if (minmaxaxis < maximum(abs(a0), abs(a1))):
                ymax = minmaxaxis
            else:
                ymax = maximum(abs(a0), abs(a1))
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
            if RTdust_or_gas == 'dust' and polarized_scat == 'Yes':
                min = 0.0
                max = 1.0  # cuidadin 1.0
            if RTdust_or_gas == 'gas' and moment_order == 1:
                min = -6.0
                max = 6.0
            CM = ax.imshow(convolved_intensity, origin='lower', cmap=mycolormap, interpolation='bilinear', extent=[
                        -180, 180, 0, maximum(abs(a0), abs(a1))], vmin=min, vmax=max, aspect='auto')   # (left, right, bottom, top)

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

            if axi_intensity == 'Yes':
                average_convolved_intensity = np.zeros(nbpixels)
                for j in range(nbpixels):
                    for i in range(nbpixels):
                        average_convolved_intensity[j] += convolved_intensity[j][i]/nbpixels

                # rkarr = np.linspace(0,minmaxaxis*distance,nbpixels)
                # rkarr = np.linspace(2,40,nbpixels)
                # rkarr = gas.rmed*gas.culength/1.5e11  # radius in a.u.
                # radius in arcseconds
                rkarr = np.linspace(
                    0, gas.rmed[-1]*gas.culength/1.5e11/distance, nbpixels)

                nb_noise = 0
                if add_noise == 'Yes':
                    nb_noise = 1

                file = open('axiconv%d.dat' % (nb_noise), 'w')
                for kk in range(nbpixels):
                    file.write('%s\t%s\t%s\n' % (str(rkarr[kk]), str(
                        np.mean(convolved_intensity[kk])), str(np.std(convolved_intensity[kk]))))
                file.close()

                fig = plt.figure(figsize=(8., 3.))
                ax = plt.gca()
                plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.25)

                ax.plot(rkarr*distance, average_convolved_intensity, color='k')
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                # Deprojected radius in arcsec
                ax.set_xlim(0, minmaxaxis*distance)
                ax.tick_params('both', labelsize=18)
                ax.set_xlabel('Orbital radius [au]', fontsize=18)
                # ax.set_ylabel(r'Axisymmetric convolved intensity [$\mu$Jy/beam]', family='monospace', fontsize=18)
                ax.set_ylabel(r'$I_C$ [$\mu$Jy/beam]', fontsize=18)

                plt.savefig('./'+'axi'+fileout, dpi=160)
                plt.clf()

            os.system('rm -rf deproj_polar_dir')
            os.chdir(currentdir)


    # =========================
    # 8. Compute 2D analytical solution to RT equation w/o scattering
    # =========================
    if calc_abs_map == 'Yes':
        print('--------- Computing 2D analytical solution to RT equation w/o scattering ----------')

        # ---------------------------
        # a) assume dust surface density != gas surface density (default case)
        # ---------------------------
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # CASE 1: FARGO2D simulation (Lagrangian particles)
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if dustdens_eq_gasdens == 'No' and fargo3d == 'No':
            # read information on the dust particles
            (rad, azi, vr, vt, Stokes, a) = np.loadtxt(
                dir+'/dustsystat'+str(on)+'.dat', unpack=True)

            # We need to recompute dustcube again as we sweeped it off the memory before. Since it
            # is now quick to compute it, we simply do it again!
            # Populate dust bins
            dust = np.zeros((nsec*nrad*nbin))
            for m in range(len(a)):   # sum over dust particles
                r = rad[m]
                t = azi[m]
                # radial index of the cell where the particle is
                if radialspacing == 'L':
                    i = int(np.log(r/gas.redge.min()) /
                            np.log(gas.redge.max()/gas.redge.min()) * nrad)
                else:
                    i = np.argmin(np.abs(gas.redge-r))
                if (i < 0 or i >= nrad):
                    sys.exit('pb with i = ', i,
                            ' in calc_abs_map step: I must exit!')
                # azimuthal index of the cell where the particle is
                # (general expression since grid spacing in azimuth is always arithmetic)
                j = int((t-gas.pedge.min())/(gas.pedge.max()-gas.pedge.min()) * nsec)
                if (j < 0 or j >= nsec):
                    sys.exit('pb with j = ', j,
                            ' in calc_abs_map step: I must exit!')
                # particle size
                pcsize = a[m]
                # find out which bin particle belongs to. Here we do nearest-grid point.
                ibin = int(np.log(pcsize/bins.min()) /
                        np.log(bins.max()/bins.min()) * nbin)
                if (ibin >= 0 and ibin < nbin):
                    k = ibin*nsec*nrad + j*nrad + i
                    dust[k] += 1
                    nparticles[ibin] += 1
                    avgstokes[ibin] += Stokes[m]

            for ibin in range(nbin):
                if nparticles[ibin] == 0:
                    nparticles[ibin] = 1
                avgstokes[ibin] /= nparticles[ibin]
                print(str(nparticles[ibin])+' grains between ' +
                    str(bins[ibin])+' and '+str(bins[ibin+1])+' meters')

            # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
            dustcube = dust.reshape(nbin, nsec, nrad)
            dustcube = np.swapaxes(dustcube, 1, 2)  # means nbin, nrad, nsec

            # Mass of gas in units of the star's mass
            Mgas = np.sum(gas.data*surface)
            print('Mgas / Mstar= '+str(Mgas) +
                ' and Mgas [kg] = '+str(Mgas*gas.cumass))

            frac = np.zeros(nbin)
            # finally compute dust surface density for each size bin
            for ibin in range(nbin):
                # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
                frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin] **
                            (4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
                # total mass of dust particles in current size bin 'ibin'
                M_i_dust = ratio * Mgas * frac[ibin]
                # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
                dustcube[ibin, :, :] *= M_i_dust / surface / nparticles[ibin]
                # conversion in g/cm^2
                # dimensions: nbin, nrad, nsec
                dustcube[ibin, :, :] *= (gas.cumass*1e3)/((gas.culength*1e2)**2.)

            # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
            if bin_small_dust == 'Yes':
                frac[0] *= 5e3
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(
                    "Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas")
                print("Mass fraction of bin 0 changed to: ", str(frac[0]))
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # radial index corresponding to 0.25"
                imin = np.argmin(np.abs(gas.rmed-1.2))
                # radial index corresponding to 0.6"
                imax = np.argmin(np.abs(gas.rmed-3.0))
                dustcube[0, imin:imax, :] = gas.data[imin:imax, :] * ratio * frac[0] * \
                    (gas.cumass*1e3)/((gas.culength*1e2)
                                    ** 2.)  # dimensions: nbin, nrad, nsec

        # -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # CASE 2: FARGO3D simulation (dust fluids)
        # -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if dustdens_eq_gasdens == 'No' and fargo3d == 'Yes':
            dust = np.zeros((nsec*nrad*nbin))
            dustcube = dust.reshape(nbin, nsec, nrad)
            dustcube = np.swapaxes(dustcube, 1, 2)  # means nbin, nrad, nsec

            for ibin in range(len(dust_id)):

                fileread = 'dust'+str(int(dust_id[ibin]))+'dens'+str(on)+'.dat'
                #print('ibin = ', ibin, ', read file = ',fileread)

                # directly read dust surface density for each dust fluid in code units
                dustcube[ibin, :, :] = Field(field=fileread, directory=dir).data

                # conversion in g/cm^2
                # dimensions: nbin, nrad, nsec
                dustcube[ibin, :, :] *= (gas.cumass*1e3)/((gas.culength*1e2)**2.)

                # decrease dust surface density inside mask radius
                # NB: mask_radius is in arcseconds
                rmask_in_code_units = mask_radius*distance*au/gas.culength/1e2
                for i in range(nrad):
                    if (gas.rmed[i] < rmask_in_code_units):
                        # *= ( (gas.rmed[i]/rmask_in_code_units)**(10.0) ) CUIDADIN!
                        dustcube[ibin, i, :] = 0.0

            print('Total dust mass [g] = ', np.sum(
                dustcube[:, :, :]*surface*(gas.culength*1e2)**2.))
            print('Total dust mass [Mgas] = ', np.sum(
                dustcube[:, :, :]*surface*(gas.culength*1e2)**2.)/(Mgas*gas.cumass*1e3))
            print('Total dust mass [Mstar] = ', np.sum(
                dustcube[:, :, :]*surface*(gas.culength*1e2)**2.)/(gas.cumass*1e3))

            # Total dust surface density
            dust_surface_density = np.sum(dustcube, axis=0)
            print(
                'Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())

        # ---------------------------
        # b) assume dust surface density = gas surface density
        # ---------------------------
        if dustdens_eq_gasdens == 'Yes':
            dust = np.zeros((nsec*nrad*nbin))
            # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
            dustcube = dust.reshape(nbin, nsec, nrad)
            dustcube = np.swapaxes(dustcube, 1, 2)  # means nbin, nrad, nsec
            frac = np.zeros(nbin)
            for ibin in range(nbin):
                frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin] **
                            (4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
                dustcube[ibin, :, :] = gas.data * ratio * frac[ibin] * \
                    (gas.cumass*1e3)/((gas.culength*1e2)
                                    ** 2.)  # dimensions: nbin, nrad, nsec

        # ---------------------------
        # We then need to recompute absorption mass opacities from dustkappa* files
        abs_opacity = np.zeros(nbin)
        lbda1 = wavelength * 1e3  # wavelength in microns
        if precalc_opac == 'Yes':
            opacdir = os.path.expanduser(opacity_dir)
            # Case where we use pre-calculated dustkappa* files located in opacity_dir files
            # whatever the composition, precomputed opacities are for dust sizes between 1 microns and 10 cm, with 50 bins
            sizemin_file = 1e-6          # in meters, do not edit!
            sizemax_file = 1e-1          # in meters, do not edit!
            nbfiles = 50                 # do not edit
            size_file = sizemin_file * \
                (sizemax_file/sizemin_file)**(np.arange(nbfiles)/(nbfiles-1.0))

        # Loop over size bins
        for k in range(nbin):
            if precalc_opac == 'No':
                # Case where we use dustkappa* files in current directory
                file = 'dustkappa_'+species+str(k)+'.inp'
                (lbda, kappa_abs, kappa_sca, g) = read_opacities(file)
                #(lbda, kappa_abs, kappa_sca, g) = np.loadtxt(file, unpack=True, skiprows=2)
                i1 = np.argmin(np.abs(lbda-lbda1))
                # linear interpolation (in log)
                l1 = lbda[i1-1]
                l2 = lbda[i1+1]
                k1 = kappa_abs[i1-1]
                k2 = kappa_abs[i1+1]
                abs_opacity[k] = (k1*np.log(l2/lbda1) + k2 *
                                np.log(lbda1/l1))/np.log(l2/l1)
                print('absorption opacity [cm^2/g] of bin ', k,
                    ' with average size ', bins[k], ' = ', abs_opacity[k])
            else:
                # Case where we use pre-calculated dustkappa* files in opacity_dir directory
                index_inf = int(
                    np.log(bins[k]/sizemin_file)/np.log(sizemax_file/sizemin_file) * nbfiles)
                if (index_inf < nbfiles-1):
                    index_sup = index_inf+1
                    file_index_inf = opacdir+'/dustkappa_' + \
                        species+str(index_inf)+'.inp'
                    file_index_sup = opacdir+'/dustkappa_' + \
                        species+str(index_sup)+'.inp'
                    (lbda_inf, kappa_abs_inf, kappa_sca_inf,
                    g_inf) = read_opacities(file_index_inf)
                    #np.loadtxt(file_index_inf, unpack=True, skiprows=2)
                    (lbda_sup, kappa_abs_sup, kappa_sca_sup,
                    g_sup) = read_opacities(file_index_sup)
                    #np.loadtxt(file_index_sup, unpack=True, skiprows=2)
                    i1_inf = np.argmin(np.abs(lbda_inf-lbda1))
                    l1 = lbda_inf[i1_inf-1]
                    l2 = lbda_inf[i1_inf+1]
                    k1_inf = kappa_abs_inf[i1_inf-1]
                    k2_inf = kappa_abs_inf[i1_inf+1]
                    i1_sup = np.argmin(np.abs(lbda_sup-lbda1))
                    k1_sup = kappa_abs_sup[i1_sup-1]
                    k2_sup = kappa_abs_sup[i1_sup+1]
                    abs_opacity[k] = k1_inf*np.log(l2/lbda1)*np.log(size_file[index_sup]/bins[k]) \
                        + k2_inf*np.log(lbda1/l1)*np.log(size_file[index_sup]/bins[k]) \
                        + k1_sup*np.log(l2/lbda1)*np.log(bins[k]/size_file[index_inf]) \
                        + k2_sup*np.log(lbda1/l1) * \
                        np.log(bins[k]/size_file[index_inf])
                    abs_opacity[k] /= (np.log(l2/l1) *
                                    np.log(size_file[index_sup]/size_file[index_inf]))
                if (index_inf == nbfiles-1):
                    file_index_inf = opacdir+'/dustkappa_' + \
                        species+str(index_inf)+'.inp'
                    (lbda_inf, kappa_abs_inf, kappa_sca_inf,
                    g_inf) = read_opacities(file_index_inf)
                    #np.loadtxt(file_index_inf, unpack=True, skiprows=2)
                    i1_inf = np.argmin(np.abs(lbda_inf-lbda1))
                    l1 = lbda_inf[i1_inf-1]
                    l2 = lbda_inf[i1_inf+1]
                    k1_inf = kappa_abs_inf[i1_inf-1]
                    k2_inf = kappa_abs_inf[i1_inf+1]
                    abs_opacity[k] = (k1_inf*np.log(l2/lbda1) +
                                    k2_inf*np.log(lbda1/l1))/np.log(l2/l1)
                # distinguish cases where index_inf = nbfiles-1 and index_inf >= nbfiles (extrapolation)
                print('absorption opacity (pre-calc) [cm^2/g] of bin ', k,
                    ' with average size ', bins[k], ' = ', abs_opacity[k])

        # kappa_abs as function of ibin, r and phi (2D array for each size bin)
        abs_opacity_2D = np.zeros((nbin, nrad, nsec))
        for i in range(nrad):
            for j in range(nsec):
                abs_opacity_2D[:, i, j] = abs_opacity    # nbin, nrad, nsec

        # Infer optical depth array
        # 2D array containing tau at each grid cell
        optical_depth = np.zeros(gas.data.shape)
        optical_depth = np.sum(dustcube*abs_opacity_2D, axis=0)    # nrad, nsec
        # divide by cos(inclination) since integral over ds = cos(i) x integral over dz
        optical_depth /= np.abs(np.cos(inclination*np.pi/180.0))
        optical_depth = optical_depth.reshape(nrad, nsec)
        optical_depth = np.swapaxes(optical_depth, 0, 1)  # means nsec, nrad

        print('max(optical depth) = ', optical_depth.max())

        # Get dust temperature
        if Tdust_eq_Thydro == 'No':
            Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
            Temp = Temp[4:]
            Temp = Temp.reshape(nbin, nsec, ncol, nrad)
            # Keep temperature of the largest dust species
            Temp = Temp[-1, :, :, :]
            # Temperature in the midplane (ncol/2 given that the grid extends on both sides about the midplane)
            # is it the midplane temperature that should be adopted? or a vertically averaged temperature?
            # avoid division by zero afterwards  (nsec, nrad)
            Tdust = Temp[:, ncol//2, :]+0.1
            # Free RAM memory
            del Temp
        else:
            Tdust = np.zeros(gas.data.shape)
            Tdust = np.swapaxes(Tdust, 0, 1)   # means nsec, nrad
            T_model = aspectratio*aspectratio*gas.cutemp * \
                gas.rmed**(-1.0+2.0*flaringindex)  # in Kelvin
            for j in range(nsec):
                Tdust[j, :] = T_model

        # Now get Bnu(Tdust)
        # Frequency in Hz; wavelength is currently in mm
        nu = c / (wavelength*1e-1)
        # 2D array containing Bnu(Tdust) at each grid cell
        Bnu = np.zeros(gas.data.shape)
        # in cgs: erg cm-2 sterad-1 = g s-2 ster-1
        Bnu = 2.0*h*(nu**3.0)/c/c/(np.exp(h*nu/kB/Tdust)-1.0)
        # Specific intensity on the simulation's polar grid in Jy/steradian:
        Inu_polar = Bnu * (1.0 - np.exp(-optical_depth)) * 1e23   # nsec, nrad

        # Now define Cartesian grid corresponding to the image plane -- we
        # overwrite the number of pixels to twice the number of radial cells
        nbpixels = 2*nrad
        #
        # recall computational grid: R = grid radius in code units, T = grid azimuth in radians
        R = gas.rmed
        T = gas.pmed
        # x- and y- coordinates of image plane
        minxinterf = -R.max()
        maxxinterf = R.max()
        xinterf = minxinterf + (maxxinterf-minxinterf) * \
            np.arange(nbpixels)/(nbpixels-1)
        yinterf = xinterf
        dxy = abs(xinterf[1]-xinterf[0])
        xgrid = xinterf+0.5*dxy
        ygrid = yinterf+0.5*dxy

        print('projecting polar specific intensity onto image plane...')
        # First do a rotation in disc plane by phiangle (clockwise), cartesian projection
        # (bilinear interpolation)
        phiangle_in_rad = phiangle*np.pi/180.0
        Inu_cart = np.zeros(nbpixels*nbpixels)
        Inu_cart = Inu_cart.reshape(nbpixels, nbpixels)
        for i in range(nbpixels):
            for j in range(nbpixels):
                xc = xgrid[i]*np.cos(phiangle_in_rad) + \
                    ygrid[j]*np.sin(phiangle_in_rad)
                yc = -xgrid[i]*np.sin(phiangle_in_rad) + \
                    ygrid[j]*np.cos(phiangle_in_rad)
                rc = np.sqrt(xc*xc + yc*yc)
                if ((rc >= R.min()) and (rc < R.max())):
                    phic = math.atan2(yc, xc) + np.pi  # result between 0 and 2pi
                    # expression for ir might not be general if simulation grid has an arithmetic spacing...
                    #ir = int(np.log(rc/R.min())/np.log(R.max()/R.min()) * (nrad-1.0))
                    ir = np.argmin(np.abs(R-rc))
                    #ir = int((rc-R.min())/(R.max()-R.min()) * nrad)
                    if (ir == nrad-1):
                        ir = nrad-2
                    # CB: since T = pmed, we need to use nsec below instead of nsec-1
                    jr = int((phic-T.min())/(T.max()-T.min()) * nsec)
                    if (jr == nsec):
                        phic = 0.0
                        Tjr = T.min()
                        jr = 0
                    else:
                        Tjr = T[jr]
                    if (jr == nsec-1):
                        Tjrp1 = T.max()
                        jrp1 = 0
                    else:
                        Tjrp1 = T[jr+1]
                        jrp1 = jr+1
                    Inu_cart[j, i] = Inu_polar[jr, ir] * (R[ir+1]-rc) * (Tjrp1-phic) \
                        + Inu_polar[jrp1, ir] * (R[ir+1]-rc) * (phic - Tjr) \
                        + Inu_polar[jr, ir+1] * (rc - R[ir]) * (Tjrp1-phic) \
                        + Inu_polar[jrp1, ir+1] * (rc - R[ir]) * (phic - Tjr)
                    Inu_cart[j, i] /= ((R[ir+1]-R[ir])*(Tjrp1-Tjr))
                    '''
                    if (Inu_cart[j,i] < 0):
                        sys.exit("Inu_cart < 0 in calc_abs_map: I must exit!")
                    '''
                else:
                    Inu_cart[j, i] = 0.0

        # Then project with inclination along line of sight:
        # with xaxisflip, only needs to be changed, not y (checked!)
        inclination_in_rad = inclination*np.pi/180.0
        Inu_cart2 = np.zeros(nbpixels*nbpixels)
        Inu_cart2 = Inu_cart2.reshape(nbpixels, nbpixels)
        for i in range(nbpixels):
            xc = xgrid[i]*np.cos(inclination_in_rad)   # || < xgrid[i]
            ir = int((xc-xgrid.min())/(xgrid.max()-xgrid.min()) * (nbpixels-1.0))
            if (ir < nbpixels-1):
                Inu_cart2[:, i] = (Inu_cart[:, ir]*(xgrid[ir+1]-xc) +
                                Inu_cart[:, ir+1]*(xc-xgrid[ir]))/(xgrid[ir+1]-xgrid[ir])
            else:
                Inu_cart2[:, i] = Inu_cart[:, nbpixels-1]

        # recast things when no xaxisflip! (flip in both x and y, checked!)
        if xaxisflip == 'No':
            Inu_cart2_orig = Inu_cart2
            Inu_cart2 = np.flipud(np.fliplr(Inu_cart2_orig))

        # Finally do a rotation in the image plane by posangle
        # add 90 degrees to be consistent with RADMC3D's convention for position angle
        posangle_in_rad = (posangle+90.0)*np.pi/180.0
        Inu_cart3 = np.zeros(nbpixels*nbpixels)
        Inu_cart3 = Inu_cart3.reshape(nbpixels, nbpixels)
        for i in range(nbpixels):
            for j in range(nbpixels):
                xc = xgrid[i]*np.cos(posangle_in_rad) + \
                    ygrid[j]*np.sin(posangle_in_rad)
                yc = -xgrid[i]*np.sin(posangle_in_rad) + \
                    ygrid[j]*np.cos(posangle_in_rad)
                ir = int((xc-xgrid.min())/(xgrid.max() -
                        xgrid.min()) * (nbpixels-1.0))
                jr = int((yc-ygrid.min())/(ygrid.max() -
                        ygrid.min()) * (nbpixels-1.0))
                if ((ir >= 0) and (jr >= 0) and (ir < nbpixels-1) and (jr < nbpixels-1)):
                    Inu_cart3[j, i] = Inu_cart2[jr, ir] * (xgrid[ir+1]-xc) * (ygrid[jr+1]-yc) \
                        + Inu_cart2[jr+1, ir] * (xgrid[ir+1]-xc) * (yc-ygrid[jr]) \
                        + Inu_cart2[jr, ir+1] * (xc-xgrid[ir]) * (ygrid[jr+1]-yc) \
                        + Inu_cart2[jr+1, ir+1] * (xc-xgrid[ir]) * (yc-ygrid[jr])
                    Inu_cart3[j, i] /= ((xgrid[ir+1]-xgrid[ir])
                                        * (ygrid[jr+1]-ygrid[jr]))
                else:
                    if ((ir >= nbpixels-1) and (jr < nbpixels-1)):
                        Inu_cart3[j, i] = 0.0  # Inu_cart2[jr-1,nbpixels-1]
                    if ((jr >= nbpixels-1) and (ir < nbpixels-1)):
                        Inu_cart3[j, i] = 0.0  # Inu_cart2[nbpixels-1,ir-1]
                    if ((ir >= nbpixels-1) and (jr >= nbpixels-1)):
                        Inu_cart3[j, i] = 0.0  # Inu_cart2[nbpixels-1,nbpixels-1]

        # Inu contains the specific intensity in Jy/steradian projected onto the image plane
        Inu = Inu_cart3
        # Disc distance in metres
        D = distance * 206265.0 * 1.5e11
        # Convert specific intensity from Jy/steradian to Jy/pixel^2
        pixsurf_ster = (dxy*gas.culength/D)**2
        Inu *= pixsurf_ster    # Jy/pixel
        print("Total flux of 2D method [Jy] = "+str(np.sum(Inu)))

        # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
        if add_noise == 'Yes':
            # beam area in pixel^2
            beam = (np.pi/(4.*np.log(2.)))*bmaj*bmin/(dxy**2.)
            # noise standard deviation in Jy per pixel (I've checked the expression below works well)
            noise_dev_std_Jy_per_pixel = noise_dev_std / np.sqrt(0.5*beam)  # 1D
            # noise array
            noise_array = np.random.normal(
                0.0, noise_dev_std_Jy_per_pixel, size=nbpixels*nbpixels)
            noise_array = noise_array.reshape(nbpixels, nbpixels)
            Inu += noise_array

        # pixel (cell) size in arcseconds
        dxy *= (gas.culength/1.5e11/distance)
        # beam area in pixel^2
        beam = (np.pi/(4.*np.log(2.)))*bmaj*bmin/(dxy**2.)
        # stdev lengths in pixel
        stdev_x = (bmaj/(2.*np.sqrt(2.*np.log(2.)))) / dxy
        stdev_y = (bmin/(2.*np.sqrt(2.*np.log(2.)))) / dxy

        # check beam is correctly handled by inserting a source point at the
        # origin of the raw intensity image
        if check_beam == 'Yes':
            Inu[nbpixels//2-1, nbpixels//2-1] = 500.0*Inu.max()

        # Call to Gauss_filter function
        print('convolution...')
        smooth2D = Gauss_filter(Inu, stdev_x, stdev_y, bpaangle, Plot=False)

        # convert image in mJy/beam or in microJy/beam
        # could be refined...
        convolved_Inu = smooth2D * 1e3 * beam   # mJy/beam
        strflux = 'Flux of continuum emission (mJy/beam)'
        if convolved_Inu.max() < 1.0:
            convolved_Inu = smooth2D * 1e6 * beam   # microJy/beam
            strflux = r'Flux of continuum emission ($\mu$Jy/beam)'

        # ---------------------------------
        # save 2D flux map solution to fits
        # ---------------------------------
        hdu = fits.PrimaryHDU()
        hdu.header['BITPIX'] = -32
        hdu.header['NAXIS'] = 2
        hdu.header['NAXIS1'] = nbpixels
        hdu.header['NAXIS2'] = nbpixels
        hdu.header['EPOCH'] = 2000.0
        hdu.header['EQUINOX'] = 2000.0
        hdu.header['LONPOLE'] = 180.0
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = float(0.0)
        hdu.header['CRVAL2'] = float(0.0)
        hdu.header['CDELT1'] = float(-1.*dxy)
        hdu.header['CDELT2'] = float(dxy)
        hdu.header['CUNIT1'] = 'arcsec    '
        hdu.header['CUNIT2'] = 'arcsec    '
        hdu.header['CRPIX1'] = float((nbpixels+1.)/2.)
        hdu.header['CRPIX2'] = float((nbpixels+1.)/2.)
        if strflux == 'Flux of continuum emission (mJy/beam)':
            hdu.header['BUNIT'] = 'milliJY/BEAM'
        if strflux == r'Flux of continuum emission ($\mu$Jy/beam)':
            hdu.header['BUNIT'] = 'microJY/BEAM'
        hdu.header['BTYPE'] = 'FLUX DENSITY'
        hdu.header['BSCALE'] = 1
        hdu.header['BZERO'] = 0
        # keep track of all parameters in params.dat file
        # for i in range(len(lines_params)):
        #    hdu.header[var[i]] = par[i]
        hdu.data = convolved_Inu
        inbasename = os.path.basename('./'+outfile)
        if add_noise == 'Yes':
            substr = '_wn'+str(noise_dev_std)+'_JyBeam2D.fits'
            jybeamfileout = re.sub('.fits', substr, inbasename)
        else:
            jybeamfileout = re.sub('.fits', '_JyBeam2D.fits', inbasename)
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
        # bottom to top (ie, north is the top).
        # CB: do not remove the multiplication by |cos(inclination)| !
        a0 = -xgrid[0]*gas.culength/1.5e11/distance * \
            np.abs(np.cos(inclination_in_rad))   # >0
        a1 = -a0
        # a1 = -xgrid[nbpixels-1]*gas.culength/1.5e11/distance*np.abs(np.cos(inclination_in_rad))    # <0
        d0 = xgrid[0]*gas.culength/1.5e11/distance * \
            np.abs(np.cos(inclination_in_rad))    # <0
        d1 = -d0
        # d1 = xgrid[nbpixels-1]*gas.culength/1.5e11/distance*np.abs(np.cos(inclination_in_rad))    # >0
        # da positive definite
        if (minmaxaxis < abs(a0)):
            da = minmaxaxis
        else:
            da = maximum(abs(a0), abs(a1))

        ax.set_xlim(da, -da)      # x (=R.A.) increases leftward
        mina = da
        maxa = -da
        xlambda = mina - 0.166*da
        ax.set_ylim(-da, da)
        # plt.ylim(-da,da)
        dmin = -da
        dmax = da

        # x- and y-ticks and labels
        ax.tick_params(top='on', right='on', length=5, width=1.0, direction='out')
        # ax.set_yticks(ax.get_xticks())    # set same ticks in x and y in cartesian
        ax.set_xlabel('RA offset [arcsec]')
        ax.set_ylabel('Dec offset [arcsec]')

        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'
        min = convolved_Inu.min()
        max = convolved_Inu.max()
        CM = ax.imshow(convolved_Inu, origin='lower', cmap=mycolormap, interpolation='bilinear', extent=[
                    a0, a1, d0, d1], vmin=min, vmax=max, aspect='auto')

        # Add wavelength in top-left corner
        strlambda = '$\lambda$=' + \
            str(round(wavelength, 2))+'mm'  # round to 2 decimals
        if wavelength < 0.01:
            strlambda = '$\lambda$='+str(round(wavelength*1e3, 2))+'$\mu$m'
        if RTdust_or_gas == 'dust':
            ax.text(xlambda, dmax-0.166*da, strlambda,
                    fontsize=20, color='white', weight='bold')

        # Add + sign at the origin
        ax.plot(0.0, 0.0, '+', color='white', markersize=10)

        # plot beam
        from matplotlib.patches import Ellipse
        e = Ellipse(xy=[xlambda, dmin+0.166*da], width=bmin,
                    height=bmaj, angle=bpaangle+90.0)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('white')
        e.set_alpha(0.8)
        ax.add_artist(e)

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb = plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')
        # title on top
        cax.xaxis.set_label_position('top')
        cax.set_xlabel(strflux)
        cax.xaxis.labelpad = 8

        plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
        plt.clf()

        # =====================
        # Compute deprojection and polar expansion (SP)
        # =====================
        if deproj_polar == 'Yes':
            currentdir = os.getcwd()
            alpha_min = 0.          # deg, PA of offset from the star
            Delta_min = 0.          # arcsec, amplitude of offset from the star
            RA = 0.0  # if input image is a prediction, star should be at the center
            DEC = 0.0  # note that this deprojection routine works in WCS coordinates
            cosi = np.cos(inclination_input*np.pi/180.)

            print('deprojection around PA [deg] = ', posangle)
            print('and inclination [deg] = ', inclination_input)

            # makes a new directory "deproj_polar_dir" and calculates a number
            # of products: copy of the input image [_fullim], centered at
            # (RA,DEC) [_centered], deprojection by cos(i) [_stretched], polar
            # image [_polar], etc. Also, a _radial_profile which is the
            # average radial intensity.
            exec_polar_expansions(jybeamfileout, 'deproj_polar_dir', posangle, cosi, RA=RA, DEC=DEC,
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
            if xaxisflip == 'Yes':
                jshift = int(nbpixels/4)
            else:
                jshift = int(nbpixels/4)
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
            if (minmaxaxis < maximum(abs(a0), abs(a1))):
                ymax = minmaxaxis
            else:
                ymax = maximum(abs(a0), abs(a1))
            ax.set_ylim(0, ymax)      # Deprojected radius in arcsec

            ax.tick_params(top='on', right='on', length=5,
                        width=1.0, direction='out')
            # ax.set_yticks((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7))
            ax.set_xticks((-180, -120, -60, 0, 60, 120, 180))
            ax.set_xlabel('Position angle [deg]')
            ax.set_ylabel('Radius [arcsec]')

            # imshow does a bilinear interpolation. You can switch it off by putting
            # interpolation='none'
            min = convolved_intensity.min()  # not exactly same as 0
            max = convolved_intensity.max()
            CM = ax.imshow(convolved_intensity, origin='lower', cmap=mycolormap, interpolation='bilinear', extent=[
                        -180, 180, 0, maximum(abs(a0), abs(a1))], vmin=min, vmax=max, aspect='auto')   # (left, right, bottom, top)

            # Add wavelength in bottom-left corner
            strlambda = '$\lambda$=' + \
                str(round(wavelength, 2))+'mm'  # round to 2 decimals
            if wavelength < 0.01:
                strlambda = '$\lambda$='+str(round(wavelength*1e3, 2))+'$\mu$m'
            if RTdust_or_gas == 'dust':
                ax.text(60, 0.02, strlambda, fontsize=16,
                        color='white', weight='bold')

            # plot color-bar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="2.5%", pad=0.12)
            cb = plt.colorbar(CM, cax=cax, orientation='horizontal')
            cax.xaxis.tick_top()
            cax.xaxis.set_tick_params(labelsize=20, direction='out')
            # title on top
            cax.xaxis.set_label_position('top')
            cax.set_xlabel(strflux)
            cax.xaxis.labelpad = 8

            plt.savefig('./'+fileout, dpi=160)
            plt.clf()

            os.system('rm -rf deproj_polar_dir')
            os.chdir(currentdir)


    print('--------- done! ----------')

if __name__ == "__main__":
    main()