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
import re

# -----------------------------
# Required librairies
# -----------------------------
from astropy.io import fits
from pylab import *

from field import *
from image import *
from makedustopac import *
from opacities import *
from particles import *
from plot import *
from radmc3d import *
from read_parameters import *


def main():

    parameters = dict()
    read_parameters_file(parameters)
    check_parameters_consistency(parameters)
    print_parameters(parameters)

    # gas density field:
    density = Field(field="rho", parameters=parameters)

    # number of grid cells in the radial and azimuthal directions
    nrad = density.nrad
    ncol = density.ncol
    nsec = density.nsec

    # volume of each grid cell (code units)
    # calculate_volume(density)

    volume = np.zeros((nsec, ncol, nrad))
    for i in range(nsec):
        for j in range(ncol):
            for k in range(nrad):
                volume[i, j, k] = (
                    density.rmed[k] ** 2
                    * np.sin(density.tmed[j])
                    * density.dr[k]
                    * density.dth[j]
                    * density.dp[i]
                )

    # Mass of gas in units of the star's mass
    Mgas = np.sum(density.data * volume)
    print(
        "Mgas / Mstar= " + str(Mgas) + " and Mgas [kg] = " + str(Mgas * density.cumass)
    )

    # Allocate arrays
    nbin = parameters["nbin"]
    # bins = np.asarray([0.0001, 0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100, 1000, 3000])
    bins = np.asarray([0.0001, 0.001, 0.01, 0.1, 0.2])
    particles_per_bin_per_cell = np.zeros(nbin * nsec * ncol * nrad)
    dust_cube = np.zeros((nbin, nsec, ncol, nrad))
    particles_per_bin = np.zeros(nbin)
    tstop_per_bin = np.zeros(nbin)

    # =========================
    # Compute dust mass volume density for each size bin
    # =========================
    if (
        parameters["RTdust_or_gas"] == "dust"
        and parameters["recalc_density"] == "Yes"
        and parameters["polarized_scat"] == "No"
    ):
        print("--------- computing dust mass volume density ----------")

        particle_data = Particles(
            ns=parameters["particle_file"], directory=parameters["dir"]
        )
        populate_dust_bins(
            density,
            particle_data,
            nbin,
            bins,
            particles_per_bin_per_cell,
            particles_per_bin,
            tstop_per_bin,
        )
        dust_cube = particles_per_bin_per_cell.reshape(
            (nbin, density.nsec, density.ncol, density.nrad)
        )
        frac = np.zeros(nbin)
        buf = 0.0
        # finally compute dust surface density for each size bin
        for ibin in range(nbin):
            # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
            frac[ibin] = (
                pow(bins[ibin + 1], (4.0 - parameters["pindex"]))
                - pow(bins[ibin], (4.0 - parameters["pindex"]))
            ) / (
                pow(parameters["amax"], (4.0 - parameters["pindex"]))
                - pow(parameters["amin"], (4.0 - parameters["pindex"]))
            )
            # total mass of dust particles in current size bin 'ibin'
            M_i_dust = parameters["ratio"] * Mgas * frac[ibin]
            buf += M_i_dust
            print("Dust mass [in units of Mstar] in species ", ibin, " = ", M_i_dust)
            # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
            dust_cube[ibin, :, :, :] *= M_i_dust / volume / particles_per_bin[ibin]
            # conversion in g/cm^2
            # dimensions: nbin, nrad, nsec
            dust_cube[ibin, :, :, :] *= (density.cumass * 1e3) / (
                (density.culength * 1e2) ** 2.0
            )

        # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
        if parameters["bin_small_dust"] == "Yes":
            frac[0] *= 5e3
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(
                "Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas"
            )
            print("Mass fraction of bin 0 changed to: ", str(frac[0]))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # radial index corresponding to 0.3"
            imin = np.argmin(np.abs(density.rmed - 1.4))
            # radial index corresponding to 0.6"
            imax = np.argmin(np.abs(density.rmed - 2.8))
            dust_cube[0, :, :, imin:imax] = (
                density.data[:, :, imin:imax]
                * parameters["ratio"]
                * frac[0]
                * (density.cumass * 1e3)
                / ((density.culength * 1e2) ** 2.0)
            )

        print(
            "Total dust mass [g] = ",
            np.sum(dust_cube[:, :, :, :] * volume * (density.culength * 1e2) ** 2.0),
        )
        print(
            "Total dust mass [Mgas] = ",
            np.sum(dust_cube[:, :, :, :] * volume * (density.culength * 1e2) ** 2.0)
            / (Mgas * density.cumass * 1e3),
        )
        print(
            "Total dust mass [Mstar] = ",
            np.sum(dust_cube[:, :, :, :] * volume * (density.culength * 1e2) ** 2.0)
            / (density.cumass * 1e3),
        )

        # Total dust surface density
        dust_surface_density = np.sum(dust_cube, axis=0)
        print(
            "Maximum dust surface density [in g/cm^2] is ", dust_surface_density.max()
        )

        DUSTOUT = open("dust_density.inp", "w")
        DUSTOUT.write("1 \n")  # iformat
        DUSTOUT.write(str(nrad * nsec * ncol) + " \n")  # n cells
        DUSTOUT.write(str(int(nbin)) + " \n")  # nbin size bins

        rhodustcube = np.zeros((nbin, nsec, ncol, nrad))

        # dust aspect ratio as function of ibin and r (or actually, R, cylindrical radius)
        hd = np.zeros((nbin, nrad))

        # gus aspect ratio
        hgas = np.zeros(nrad)
        for irad in range(nrad):
            hgas[irad] = (
                density.rmed[irad] * np.cos(density.tmed[:]) * density.data[0, :, irad]
            ).sum(axis=0) / (density.data[0, :, irad]).sum(axis=0)

        for ibin in range(nbin):
            if parameters["polarized_scat"] == "No":
                for irad in range(nrad):
                    hd[ibin, irad] = hgas[irad] / np.sqrt(
                        1.0
                        + tstop_per_bin[ibin]
                        / parameters["alphaviscosity"]
                        * (1.0 + 2.0 * tstop_per_bin[ibin])
                        / (1.0 + tstop_per_bin[ibin])
                    )
            else:
                print(
                    "Set of initial conditions not implemented for pluto yet. Only parameters['polarized_scat'] == 'No'"
                )
                sys.exit("I must exit!")

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
            density.tedge, density.pedge, density.redge
        )  # ncol+1, nrad+1, Nsec+1

        r2 = Redge * Redge
        jacob = r2[:-1, :-1, :-1] * np.sin(Cedge[:-1, :-1, :-1])
        dphi = Aedge[1:, :-1, :-1] - Aedge[:-1, :-1, :-1]  # same as 2pi/nsec
        dr = Redge[:-1, :-1, 1:] - Redge[:-1, :-1, :-1]  # same as Rsup-Rinf
        dtheta = Cedge[:-1, 1:, :-1] - Cedge[:-1, :-1, :-1]
        # volume of a cell in cm^3
        vol = (
            jacob * dr * dphi * dtheta * ((density.culength * 1e2) ** 3)
        )  # ncol, nrad, Nsec

        total_mass = np.sum(rhofield * vol)

        normalization_factor = (
            parameters["ratio"] * Mgas * (density.cumass * 1e3) / total_mass
        )
        rho_dust_cube = rho_dust_cube * normalization_factor
        print(
            "total dust mass after vertical expansion [g] = ",
            np.sum(np.sum(rho_dust_cube, axis=0) * vol),
            " as normalization factor = ",
            normalization_factor,
        )

        # write mass volume densities for all size bins
        for ibin in range(nbin):
            print("dust species in bin", ibin, "out of ", nbin - 1)
            for k in range(nsec):
                for j in range(ncol):
                    for i in range(nrad):
                        DUSTOUT.write(str(rho_dust_cube[ibin, k, j, i]) + " \n")

        # print max of dust's mass volume density at each colatitude
        for j in range(ncol):
            print(
                "max(rho_dustcube) [g cm-3] for colatitude index j = ",
                j,
                " = ",
                rho_dust_cube[:, :, j, :].max(),
            )

        DUSTOUT.close()

        # plot azimuthally-averaged density vs. radius and colatitude
        if parameters["plot_density"] == "Yes":
            plot_density(nbin, nsec, ncol, nrad, density, rho_dust_cube)

        # free RAM memory
        del rho_dust_cube, dust_cube, particles_per_bin_per_cell

    elif parameters["RTdust_or_gas"] == "gas":
        print(
            "Set of initial conditions not implemented for pluto yet. Only parameters['RTdust_or_gas'] == 'dust'"
        )
        sys.exit("I must exit!")
    elif parameters["polarized_scat"] == "Yes":
        print(
            "Set of initial conditions not implemented for pluto yet. Only parameters['polarized_scat'] == 'No'"
        )
        sys.exit("I must exit!")
    else:
        print(
            "--------- I did not compute dust densities (recalc_density = No in params.dat file) ----------"
        )

    # =========================
    # Compute dust opacities
    # =========================
    if parameters["RTdust_or_gas"] == "dust" and parameters["recalc_opac"] == "Yes":
        print("--------- computing dust opacities ----------")

        # Calculation of opacities uses the python scripts makedustopac.py and bhmie.py
        # which were written by C. Dullemond, based on the original code by Bohren & Huffman.

        logawidth = 0.05  # Smear out the grain size by 5% in both directions
        na = 20  # Use 10 grain size samples per bin size
        chop = 1.0  # Remove forward scattering within an angle of 5 degrees
        # Extrapolate optical constants beyond its wavelength grid, if necessary
        extrapol = True
        verbose = False  # If True, then write out status information
        ntheta = 181  # Number of scattering angle sampling points
        # link to optical constants file
        optconstfile = (
            os.path.expanduser(parameters["opacity_dir"])
            + "/"
            + parameters["species"]
            + ".lnk"
        )

        # The material density in gram / cm^3
        graindens = 2.0  # default density in g / cc
        if (
            parameters["species"] == "mix_2species_porous"
            or parameters["species"] == "mix_2species_porous_ice"
            or parameters["species"] == "mix_2species_porous_ice70"
        ):
            graindens = 0.1  # g / cc
        if (
            parameters["species"] == "mix_2species"
            or parameters["species"] == "mix_2species_60silicates_40ice"
        ):
            graindens = 1.7  # g / cc
        if parameters["species"] == "mix_2species_ice70":
            graindens = 1.26  # g / cc
        if parameters["species"] == "mix_2species_60silicates_40carbons":
            graindens = 2.7  # g / cc

        # Set up a wavelength grid (in cm) upon which we want to compute the opacities
        # 1 micron -> 1 cm
        lamcm = 10.0 ** np.linspace(0, 4, 200) * 1e-4

        # Set up an angular grid for which we want to compute the scattering matrix Z
        theta = np.linspace(0.0, 180.0, ntheta)

        for ibin in range(int(nbin)):
            # median grain size in cm in current bin size:
            agraincm = 10.0 ** (
                0.5 * (np.log10(1e2 * bins[ibin]) + np.log10(1e2 * bins[ibin + 1]))
            )

            print("====================")
            print("bin ", ibin + 1, "/", nbin)
            print(
                "grain size [cm]: ",
                agraincm,
                " with grain density [g/cc] = ",
                graindens,
            )
            print("====================")
            pathout = parameters["species"] + str(ibin)
            opac = compute_opac_mie(
                optconstfile,
                graindens,
                agraincm,
                lamcm,
                theta=theta,
                extrapolate=extrapol,
                logawidth=logawidth,
                na=na,
                chopforward=chop,
                verbose=verbose,
            )
            if parameters["scat_mode"] >= 3:
                print("Writing dust opacities in dustkapscatmat* files")
                write_radmc3d_scatmat_file(opac, pathout)
            else:
                print("Writing dust opacities in dustkappa* files")
                write_radmc3d_kappa_file(opac, pathout)
    else:
        print(
            "------- taking dustkap* opacity files in current directory (recalc_opac = No in params.dat file) ------ "
        )

    # Write dustopac.inp file even if we don't (re)calculate dust opacities
    if parameters["RTdust_or_gas"] == "dust":
        print("-> writing dust opacities")
        write_dustopac(
            species=parameters["species"],
            scat_mode=parameters["scat_mode"],
            nbin=parameters["nbin"],
        )
        if parameters["plot_opac"] == "Yes":
            print("-> plotting dust opacities")
            plot_opacities(
                species=parameters["species"],
                amin=parameters["amin"],
                amax=parameters["amax"],
                nbin=parameters["nbin"],
                lbda1=parameters["wavelength"] * 1e3,
            )

    print("-> writing radmc3d script")
    write_radmc3d_script(parameters)

    # =========================
    # Call to RADMC3D thermal solution and ray tracing
    # =========================
    if parameters["recalc_radmc"] == "Yes" or parameters["recalc_rawfits"] == "Yes":
        # Write other parameter files required by RADMC3D
        print("--------- printing auxiliary files ----------")

        # need to check why we need to output wavelength...
        if parameters["recalc_rawfits"] == "No":
            write_wavelength()
            write_stars(Rstar=parameters["rstar"], Tstar=parameters["teff"])
            # Write 3D spherical grid for RT computational calculation
            write_AMRgrid(density, Plot=False)

            # rto_style = 3 means that RADMC3D will write binary output files
            # setthreads corresponds to the number of threads (cores) over which radmc3d runs
            write_radmc3dinp(
                incl_dust=parameters["incl_dust"],
                incl_lines=parameters["incl_lines"],
                lines_mode=parameters["lines_mode"],
                nphot_scat=parameters["nb_photons_scat"],
                nphot=parameters["nb_photons"],
                rto_style=3,
                tgas_eq_tdust=parameters["tgas_eq_tdust"],
                modified_random_walk=1,
                scattering_mode_max=parameters["scat_mode"],
                setthreads=parameters["nbcores"],
            )

        # Add 90 degrees to position angle so that RADMC3D's definition of
        # position angle be consistent with observed position
        # angle, which is what we enter in the params.dat file
        M = RTmodel(
            distance=parameters["distance"],
            Lambda=parameters["wavelength"] * 1e3,
            label=parameters["label"],
            line=parameters["gasspecies"],
            iline=parameters["iline"],
            vkms=parameters["vkms"],
            widthkms=parameters["widthkms"],
            npix=parameters["nbpixels"],
            phi=parameters["phiangle"],
            incl=parameters["inclination"],
            posang=parameters["posangle"] + 90.0,
        )

        # Set dust / gas temperature if Tdust_eq_Thydro == 'Yes'
        if (
            parameters["recalc_rawfits"] == "No"
            and parameters["Tdust_eq_Thydro"] == "Yes"
            and parameters["RTdust_or_gas"] == "dust"
            and parameters["recalc_temperature"] == "Yes"
        ):
            print("--------- Writing temperature file (no mctherm) ----------")
            os.system("rm -f dust_temperature.bdat")  # avoid confusion!...
            TEMPOUT = open("dust_temperature.dat", "w")
            TEMPOUT.write("1 \n")  # iformat
            TEMPOUT.write(str(nrad * nsec * ncol) + " \n")  # n cells
            TEMPOUT.write(str(int(nbin)) + " \n")  # nbin size bins

            gas_temp = np.zeros((ncol, nrad, nsec))
            thydro = (
                parameters["aspectratio"]
                * parameters["aspectratio"]
                * density.cutemp
                * density.rmed ** (-1.0 + 2.0 * parameters["flaringindex"])
            )
            for k in range(nsec):
                for j in range(ncol):
                    gas_temp[j, :, k] = thydro

            # write dust temperature for all size bins
            for ibin in range(nbin):
                print(
                    "writing temperature of dust species in bin",
                    ibin,
                    "out of ",
                    nbin - 1,
                )
                for k in range(nsec):
                    for j in range(ncol):
                        for i in range(nrad):
                            TEMPOUT.write(str(gas_temp[j, i, k]) + " \n")
            TEMPOUT.close()
            del gas_temp

        # Now run RADMC3D
        if parameters["recalc_rawfits"] == "No":
            print("--------- Now executing RADMC3D ----------")
            os.system("./script_radmc")

        print("--------- exporting results in fits format ----------")
        outfile = exportfits(M, parameters)

        if parameters["plot_temperature"] == "Yes":
            plot_temperature(nbin, nsec, ncol, nrad, density, parameters)
    else:
        print(
            "------- I did not run RADMC3D, using existing .fits file for convolution "
        )
        print("------- (recalc_radmc = No in params.dat file) and final image ------ ")

        if parameters["RTdust_or_gas"] == "dust":
            outfile = (
                "image_"
                + str(parameters["label"])
                + "_lbda"
                + str(parameters["wavelength"])
                + "_i"
                + str(parameters["inclination"])
                + "_phi"
                + str(parameters["phiangle"])
                + "_PA"
                + str(parameters["posangle"])
            )
        else:
            print(
                "Set of initial conditions not implemented for pluto yet. Only parameters['RTdust_or_gas'] == 'dust'"
            )
            sys.exit("I must exit!")

        if parameters["secondorder"] == "Yes":
            outfile = outfile + "_so"
        if parameters["dustdens_eq_gasdens"] == "Yes":
            outfile = outfile + "_ddeqgd"
        if parameters["bin_small_dust"] == "Yes":
            outfile = outfile + "_bin0"

        outfile = outfile + ".fits"

    # =========================
    # Convolve raw flux with beam and produce final image
    # =========================
    if parameters["recalc_fluxmap"] == "Yes":
        print("--------- Convolving and writing final image ----------")

        f = fits.open("./" + outfile)

        # remove .fits extension
        outfile = os.path.splitext(outfile)[0]

        # add bmaj information
        outfile = outfile + "_bmaj" + str(parameters["bmaj"])

        outfile = outfile + ".fits"

        hdr = f[0].header
        # pixel size converted from degrees to arcseconds
        cdelt = np.abs(hdr["CDELT1"] * 3600.0)

        # get wavelength and convert it from microns to mm
        lbda0 = hdr["LBDAMIC"] * 1e-3

        # no polarized scattering: fits file directly contains raw intensity field
        if parameters["polarized_scat"] == "No":
            nx = hdr["NAXIS1"]
            ny = hdr["NAXIS2"]
            raw_intensity = f[0].data
            if parameters["recalc_radmc"] == "No" and parameters["plot_tau"] == "No":
                # sum over pixels
                print("Total flux [Jy] = " + str(np.sum(raw_intensity)))
            # check beam is correctly handled by inserting a source point at the
            # origin of the raw intensity image
            if parameters["check_beam"] == "Yes":
                raw_intensity[:, :] = 0.0
                raw_intensity[nx // 2 - 1, ny // 2 - 1] = 1.0
            # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
            if (
                parameters["add_noise"] == "Yes"
                and parameters["RTdust_or_gas"] == "dust"
                and parameters["plot_tau"] == "No"
            ):
                # beam area in pixel^2
                beam = (
                    (np.pi / (4.0 * np.log(2.0)))
                    * parameters["bmaj"]
                    * parameters["bmin"]
                    / (cdelt ** 2.0)
                )
                # noise standard deviation in Jy per pixel (I've checked the expression below works well)
                noise_dev_std_Jy_per_pixel = parameters["noise_dev_std"] / np.sqrt(
                    0.5 * beam
                )  # 1D
                # noise array
                noise_array = np.random.normal(
                    0.0,
                    noise_dev_std_Jy_per_pixel,
                    size=parameters["nbpixels"] * parameters["nbpixels"],
                )
                noise_array = noise_array.reshape(
                    parameters["nbpixels"], parameters["nbpixels"]
                )
                raw_intensity += noise_array
            if parameters["brightness_temp"] == "Yes":
                # beware that all units are in cgs! We need to convert
                # 'intensity' from Jy/pixel to cgs units!
                # pixel size in each direction in cm
                pixsize_x = cdelt * parameters["distance"] * au
                pixsize_y = pixsize_x
                # solid angle subtended by pixel size
                pixsurf_ster = (
                    pixsize_x
                    * pixsize_y
                    / parameters["distance"]
                    / parameters["distance"]
                    / pc
                    / pc
                )
                # convert intensity from Jy/pixel to erg/s/cm2/Hz/sr
                intensity_buf = raw_intensity / 1e23 / pixsurf_ster
                # beware that lbda0 is in mm right now, we need to have it in cm in the expression below
                raw_intensity = (h * c / kB / (lbda0 * 1e-1)) / np.log(
                    1.0 + 2.0 * h * c / intensity_buf / pow((lbda0 * 1e-1), 3.0)
                )
                # raw_intensity = np.nan_to_num(raw_intensity)
        else:
            print(
                "Set of initial conditions not implemented for pluto yet. Only parameters['polarized_scat'] == 'No'"
            )
            sys.exit("I must exit!")

        # ------------
        # smooth image
        # ------------
        # beam area in pixel^2
        beam = (
            (np.pi / (4.0 * np.log(2.0)))
            * parameters["bmaj"]
            * parameters["bmin"]
            / (cdelt ** 2.0)
        )
        # stdev lengths in pixel
        stdev_x = (parameters["bmaj"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / cdelt
        stdev_y = (parameters["bmin"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / cdelt

        # a) case with no polarized scattering
        if parameters["polarized_scat"] == "No" and parameters["plot_tau"] == "No":
            # Call to Gauss_filter function
            if parameters["moment_order"] != 1:
                smooth = Gauss_filter(
                    raw_intensity, stdev_x, stdev_y, parameters["bpaangle"], Plot=False
                )
            else:
                smooth = raw_intensity

            # convert image from Jy/pixel to mJy/beam or microJy/beam
            # could be refined...
            if parameters["brightness_temp"] == "Yes":
                convolved_intensity = smooth
            else:
                convolved_intensity = smooth * 1e3 * beam  # mJy/beam

            strflux = "Flux of continuum emission (mJy/beam)"
            if parameters["gasspecies"] == "co":
                strgas = r"$^{12}$CO"
            elif parameters["gasspecies"] == "13co":
                strgas = r"$^{13}$CO"
            elif parameters["gasspecies"] == "c17o":
                strgas = r"C$^{17}$O"
            elif parameters["gasspecies"] == "c18o":
                strgas = r"C$^{18}$O"
            elif parameters["gasspecies"] == "hco+":
                strgas = r"HCO+"
            elif parameters["gasspecies"] == "so":
                strgas = r"SO"
            else:
                strgas = parameters["gasspecies"]
            if parameters["gasspecies"] != "so":
                strgas += r" ($%d \rightarrow %d$)" % (
                    parameters["iline"],
                    parameters["iline"] - 1,
                )
            if parameters["gasspecies"] == "so" and parameters["iline"] == 14:
                strgas += r" ($5_6 \rightarrow 4_5$)"

            if parameters["brightness_temp"] == "Yes":
                if parameters["RTdust_or_gas"] == "dust":
                    strflux = r"Brightness temperature (K)"
            else:
                if convolved_intensity.max() < 1.0:
                    convolved_intensity = smooth * 1e6 * beam  # microJy/beam
                    strflux = r"Flux of continuum emission ($\mu$Jy/beam)"

        if parameters["plot_tau"] == "Yes":
            convolved_intensity = raw_intensity
            strflux = r"Absorption optical depth $\tau"

        # -------------------------------------
        # SP: save convolved flux map solution to fits
        # -------------------------------------
        hdu = fits.PrimaryHDU()
        hdu.header["BITPIX"] = -32
        hdu.header["NAXIS"] = 2  # 2
        hdu.header["NAXIS1"] = parameters["nbpixels"]
        hdu.header["NAXIS2"] = parameters["nbpixels"]
        hdu.header["EPOCH"] = 2000.0
        hdu.header["EQUINOX"] = 2000.0
        hdu.header["LONPOLE"] = 180.0
        hdu.header["CTYPE1"] = "RA---SIN"
        hdu.header["CTYPE2"] = "DEC--SIN"
        hdu.header["CRVAL1"] = float(0.0)
        hdu.header["CRVAL2"] = float(0.0)
        hdu.header["CDELT1"] = hdr["CDELT1"]
        hdu.header["CDELT2"] = hdr["CDELT2"]
        hdu.header["LBDAMIC"] = hdr["LBDAMIC"]
        hdu.header["CUNIT1"] = "deg     "
        hdu.header["CUNIT2"] = "deg     "
        hdu.header["CRPIX1"] = float((parameters["nbpixels"] + 1.0) / 2.0)
        hdu.header["CRPIX2"] = float((parameters["nbpixels"] + 1.0) / 2.0)
        if strflux == "Flux of continuum emission (mJy/beam)":
            hdu.header["BUNIT"] = "milliJY/BEAM"
        if strflux == r"Flux of continuum emission ($\mu$Jy/beam)":
            hdu.header["BUNIT"] = "microJY/BEAM"
        if strflux == "":
            hdu.header["BUNIT"] = ""
        hdu.header["BTYPE"] = "FLUX DENSITY"
        hdu.header["BSCALE"] = 1
        hdu.header["BZERO"] = 0
        del hdu.header["EXTEND"]
        # keep track of all parameters in params.dat file
        # for i in range(len(lines_params)):
        #    hdu.header[var[i]] = par[i]
        hdu.data = convolved_intensity
        inbasename = os.path.basename("./" + outfile)
        if parameters["add_noise"] == "Yes":
            substr = "_wn" + str(parameters["noise_dev_std"]) + "_JyBeam.fits"
            jybeamfileout = re.sub(".fits", substr, inbasename)
        else:
            jybeamfileout = re.sub(".fits", "_JyBeam.fits", inbasename)
        hdu.writeto(jybeamfileout, overwrite=True)

        plot_image(
            nx, cdelt, lbda0, strflux, convolved_intensity, jybeamfileout, parameters
        )

    print("--------- done! ----------")


if __name__ == "__main__":
    main()
