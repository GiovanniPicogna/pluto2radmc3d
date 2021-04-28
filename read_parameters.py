import os
import subprocess
import sys

param_file = "params.dat"


def read_parameter(parameter, default):
    command = 'awk " BEGIN{IGNORECASE=1} /^' + parameter + '/ " ' + param_file
    if sys.version_info[0] < 3:  # python 2.X
        buf = subprocess.check_output(command, shell=True)
    else:  # python 3.X
        buf = subprocess.getoutput(command)
    try:
        value = str(buf.split()[1])
    except:
        value = default
    return value


# =========================
# read parameter file
# =========================
def read_parameters_file(parameters):

    # directory where the pluto output files are
    default = "./"
    parameters["dir"] = str(read_parameter("dir", default))
    if len(parameters["dir"]) > 1:
        if parameters["dir"][-1] != "/":
            parameters["dir"] += "/"

    # number of the gas output file to read
    default = 0
    parameters["on"] = int(read_parameter("on", default))

    # number of the particle output file to read
    default = 0
    parameters["particle_file"] = int(read_parameter("particle_file", default))

    # xaxisflip
    default = "No"
    parameters["xaxisflip"] = str(read_parameter("xaxisflip", default))

    # override_units
    default = "No"
    parameters["override_units"] = str(read_parameter("override_units", default))

    # RTdust_or_gas
    default = "dust"
    parameters["RTdust_or_gas"] = str(read_parameter("RTdust_or_gas", default))

    # recalc_density
    default = "Yes"
    parameters["recalc_density"] = str(read_parameter("recalc_density", default))

    # recalc_density
    default = "No"
    parameters["dustdens_eq_gasdens"] = str(
        read_parameter("dustdens_eq_gasdens", default)
    )

    # recalc_temperature
    default = "Yes"
    parameters["recalc_temperature"] = str(
        read_parameter("recalc_temperature", default)
    )

    # recalc_opacity
    default = "Yes"
    parameters["recalc_opac"] = str(read_parameter("recalc_opac", default))

    # recalc_radmc
    default = "Yes"
    parameters["recalc_radmc"] = str(read_parameter("recalc_radmc", default))

    # recalc_rawfits
    default = "Yes"
    parameters["recalc_rawfits"] = str(read_parameter("recalc_rawfits", default))

    # recalc_fluxmap
    default = "No"
    parameters["recalc_fluxmap"] = str(read_parameter("recalc_fluxmap", default))

    # Tdust_eq_Thydro
    default = "Yes"
    parameters["Tdust_eq_Thydro"] = str(read_parameter("Tdust_eq_Thydro", default))

    # polarized_scat
    default = "No"
    parameters["polarized_scat"] = str(read_parameter("polarized_scat", default))

    # brightness_temp
    default = "No"
    parameters["brightness_temp"] = str(read_parameter("brightness_temp", default))

    # plot_density
    default = "Yes"
    parameters["plot_density"] = str(read_parameter("plot_density", default))

    # plot_temperature
    default = "No"
    parameters["plot_temperature"] = str(read_parameter("plot_temperature", default))

    # plot_tau
    default = "No"
    parameters["plot_tau"] = str(read_parameter("plot_tau", default))

    # plot_opac
    default = "No"
    parameters["plot_opac"] = str(read_parameter("plot_opac", default))

    # calc_abs_map
    default = "No"
    parameters["calc_abs_map"] = str(read_parameter("calc_abs_map", default))

    # add_noise
    default = "No"
    parameters["add_noise"] = str(read_parameter("add_noise", default))

    # noise_dev_std
    default = 1e-3
    parameters["noise_dev_std"] = float(read_parameter("noise_dev_std", default))

    # deproj_polar
    default = "No"
    parameters["deproj_polar"] = str(read_parameter("deproj_polar", default))

    # nsec
    default = 600
    parameters["nsec"] = int(read_parameter("nsec", default))

    # zmax_over_H
    default = 5.0
    parameters["zmax_over_H"] = float(read_parameter("zmax_over_H", default))

    # lines_mode
    default = 1
    parameters["lines_mode"] = int(read_parameter("lines_mode", default))

    # gasspecies
    default = "co"
    parameters["gasspecies"] = str(read_parameter("gasspecies", default))

    # iline
    default = 3
    parameters["iline"] = int(read_parameter("iline", default))

    # abundance
    default = 1e-4
    parameters["abundance"] = float(read_parameter("abundance", default))

    # vkms
    default = 0.0
    parameters["vkms"] = float(read_parameter("vkms", default))

    # widthkms
    default = 9.0
    parameters["widthkms"] = float(read_parameter("widthkms", default))

    # moment_order
    default = 0
    parameters["moment_order"] = int(read_parameter("moment_order", default))

    # linenlam
    default = 101
    parameters["linenlam"] = int(read_parameter("linenlam", default))

    # turbvel
    default = 0.0
    parameters["turbvel"] = float(read_parameter("turbvel", default))

    # photodissociation
    default = "Yes"
    parameters["photodissociation"] = str(read_parameter("photodissociation", default))

    # freezeout
    default = "No"
    parameters["freezeout"] = str(read_parameter("freezeout", default))

    # wavelength
    default = 1.04e-3
    parameters["wavelength"] = float(read_parameter("wavelength", default))

    # amin
    default = 1e-4
    parameters["amin"] = float(read_parameter("amin", default))

    # amax
    default = 1e4
    parameters["amax"] = float(read_parameter("amax", default))

    # pindex
    default = 3.5
    parameters["pindex"] = float(read_parameter("pindex", default))

    # ratio
    # default 2e-3
    parameters["ratio"] = float(read_parameter("ratio", default))

    # nbin
    default = 3
    parameters["nbin"] = int(read_parameter("nbin", default))

    # bin_small_dust
    default = "No"
    parameters["bin_small_dust"] = str(read_parameter("bin_small_dus" "t", default))

    # z_expansion
    default = "G"
    parameters["z_expansion"] = str(read_parameter("z_expansion", default))

    # species
    default = "mix_2species_60silicates_40carbons"
    parameters["species"] = str(read_parameter("species", default))

    # precalc_opac
    default = "Yes"
    parameters["precalc_opac"] = str(read_parameter("precalc_opac", default))

    # opacity_dir
    default = "./"
    parameters["opacity_dir"] = str(read_parameter("opacity_dir", default))

    # distance
    default = 100.0
    parameters["distance"] = float(read_parameter("distance", default))

    # inclination
    default = 30.0
    parameters["inclination"] = float(read_parameter("inclination", default))

    # phiangle
    default = 0.0
    parameters["phiangle"] = float(read_parameter("phiangle", default))

    # posangle
    default = -90.0
    parameters["posangle"] = float(read_parameter("posangle", default))

    # rstar
    default = 2.0
    parameters["rstar"] = float(read_parameter("rstar", default))

    # teff
    default = 7000.0
    parameters["teff"] = float(read_parameter("teff", default))

    # nbpixels
    default = 1024
    parameters["nbpixels"] = int(read_parameter("nbpixels", default))

    # scat_mode
    default = 5
    parameters["scat_mode"] = int(read_parameter("scat_mode", default))

    # nb_photons
    default = 3.0e8
    parameters["nb_photons"] = int(float(read_parameter("nb_photons", default)))

    # nb_photons_scat
    default = 3.0e8
    parameters["nb_photons_scat"] = int(
        float(read_parameter("nb_photons_scat", default))
    )

    # secondorder
    default = "Yes"
    parameters["secondorder"] = str(read_parameter("secondorder", default))

    # axi_intensity
    default = "No"
    parameters["axi_intensity"] = str(read_parameter("axi_intensity", default))

    # truncation_radius
    default = 0.0
    parameters["truncation_radius"] = float(
        read_parameter("truncation_radius", default)
    )

    # mask_radius
    default = 0.0
    parameters["mask_radius"] = float(read_parameter("mask_radius", default))

    # bmaj
    default = 0.05
    parameters["bmaj"] = float(read_parameter("bmaj", default))

    # bmin
    default = 0.05
    parameters["bmin"] = float(read_parameter("bmin", default))

    # bpaangle
    default = 0.0
    parameters["bpaangle"] = float(read_parameter("bpaangle", default))

    # check_beam
    default = "No"
    parameters["check_beam"] = str(read_parameter("check_beam", default))

    # minmaxaxis
    default = 0.55
    parameters["minmaxaxis"] = float(read_parameter("minmaxaxis", default))

    # Color map
    default = "nipy_spectral"
    parameters["mycolormap"] = str(read_parameter("mycolormap", default))

    # nbcores
    default = 4
    parameters["nbcores"] = int(read_parameter("nbcores", default))

    # set spherical grid, array allocation.

    # get the aspect ratio and flaring index used in the numerical simulation
    parameters["aspectratio"] = 0.1

    # get the flaring index used in the numerical simulation
    parameters["flaringindex"] = 0.0

    # get the alpha viscosity used in the numerical simulation
    try:
        command = 'awk " /^ALPHA/ " ' + parameters["dir"] + "/*.ini"
        if sys.version_info[0] < 3:
            buf = subprocess.check_output(command, shell=True)
        else:
            buf = subprocess.getoutput(command)
        parameters["alphaviscosity"] = float(buf.split()[1])
    # if no alphaviscosity, then try to see if a constant
    # kinematic viscosity has been used in the simulation
    except IndexError:
        command = 'awk " /^Viscosity/ " ' + parameters["dir"] + "/*.par"
        if sys.version_info[0] < 3:
            buf = subprocess.check_output(command, shell=True)
        else:
            buf = subprocess.getoutput(command)
        viscosity = float(buf.split()[1])
        # simply set constant alpha value as nu / h^2 (ok at code's unit of length)
        parameters["alphaviscosity"] = viscosity * (parameters["aspectratio"] ** (-2.0))

    # save copy of params.dat
    os.system("cp params.dat params_last.dat")


# =========================
# check parameters
# =========================
def check_parameters_consistency(parameters):
    if parameters["RTdust_or_gas"] == "dust":
        parameters["incl_dust"] = 1
        parameters["incl_lines"] = 0
        parameters["linenlam"] = 1
    if parameters["RTdust_or_gas"] == "gas":
        parameters["incl_lines"] = 1
        parameters["incl_dust"] = 0
        # currently in our RT gas calculations, the gas temperature has to
        # be that in the hydro simulation, it cannot be equal to the dust
        # temperature computed by a RT Monte-Carlo calculation
        parameters["Tdust_eq_Thydro"] = "Yes"
        if parameters["moment_order"] == 1 and parameters["inclination"] == 0.0:
            sys.exit(
                "To get an actual velocity map you need a non-zero disc inclination. Abort!"
            )
    if parameters["recalc_radmc"] == "Yes":
        parameters["recalc_fluxmap"] = "Yes"
    if parameters["recalc_rawfits"] == "Yes":
        parameters["recalc_fluxmap"] = "Yes"
    if parameters["Tdust_eq_Thydro"] == "Yes":
        parameters["tgas_eq_tdust"] = 0
    else:
        parameters["tgas_eq_tdust"] = 1
    if parameters["override_units"] == "Yes":
        if parameters["new_unit_length"] == 0.0:
            sys.exit(
                "override_units set to yes but new_unit_length is not defined in params.dat, I must exit!"
            )
        if parameters["new_unit_mass"] == 0.0:
            sys.exit(
                "override_units set to yes but new_unit_mass is not defined in params.dat, I must exit!"
            )
    if parameters["axi_intensity"] == "Yes":
        parameters["deproj_polar"] = "Yes"

    # x-axis flip means that we apply a mirror-symetry x -> -x to the 2D simulation plane,
    # which in practice is done by adding 180 degrees to the disc's inclination wrt the line of sight
    parameters["inclination_input"] = parameters["inclination"]
    if parameters["xaxisflip"] == "Yes":
        parameters["inclination"] = parameters["inclination"] + 180.0

    # this is how the beam position angle should be modified to be understood as
    # measuring East from North. You can check this by setting check_beam to Yes
    # in the params.dat parameter file
    parameters["bpaangle"] = -90.0 - parameters["bpaangle"]

    if parameters["moment_order"] == 1:
        parameters["mycolormap"] = "RdBu_r"


# =========================
# print parameters
# =========================
def print_parameters(parameters):

    print("--------- RT parameters ----------")
    print("RT in the dust or in gas lines? = ", parameters["RTdust_or_gas"])
    if parameters["RTdust_or_gas"] == "gas":
        print("what method for the line RT calculations ?", parameters["lines_mode"])
        print("gas species = ", parameters["gasspecies"])
        print("line in the rotational ladder", parameters["iline"])
        print("shift in the systemic velocity ?", parameters["vkms"])
        print("velocity width around systemic velocity ?", parameters["widthkms"])
        print(
            "number of wavelengths for multi-color gas images ?", parameters["linenlam"]
        )
        if parameters["turbvel"] != "cavity":
            print("turbulent velocity [m/s] = ", parameters["turbvel"])
        print("abundance of the gas species = ", parameters["abundance"])

    print("directory = ", parameters["dir"])
    print("output number  = ", parameters["on"])
    print("wavelength [mm] = ", parameters["wavelength"])
    print("do we plot optical depth? : ", parameters["plot_tau"])
    print("do we compute polarized intensity image? : ", parameters["polarized_scat"])
    print("dust minimum size [m] = ", parameters["amin"])
    print("dust maximum size [m] = ", parameters["amax"])
    print("minus slope of dust size distribution = ", parameters["pindex"])
    print("dust-to-gas mass ratio = ", parameters["ratio"])
    print("number of size bins = ", parameters["nbin"])
    print("disc distance [pc] = ", parameters["distance"])
    print("disc inclination [deg] = ", parameters["inclination"])
    print("disc phi angle [deg] = ", parameters["phiangle"])
    print("disc position angle [deg] = ", parameters["posangle"])
    print("beam major axis [arcsec] = ", parameters["bmaj"])
    print("beam minor axis [ascsec] = ", parameters["bmin"])
    print("beam position angle [deg] = ", parameters["bpaangle"])
    print("star radius [Rsun] = ", parameters["rstar"])
    print("star effective temperature [K] = ", parameters["teff"])
    print(
        "Do we display intensity in brightness temperature?: ",
        parameters["brightness_temp"],
    )
    print("number of grid cells in phi for RADMC3D = ", parameters["nsec"])
    print(
        "type of vertical expansion done for dust mass volume density = ",
        parameters["z_expansion"],
    )
    if parameters["z_expansion"] == "G":
        print("Hd / Hgas = 1 if R < truncation_radius, R^-2 decrease beyond")
    if parameters["z_expansion"] == "T":
        print("Hd / Hgas = sqrt(alpha/(alpha+St))")
    if parameters["z_expansion"] == "T2":
        print("Hd / Hgas = sqrt(10*alpha/(10*alpha+St))")
    if parameters["z_expansion"] == "F":
        print("Hd / Hgas = 0.7 x ((St + 1/St)/1000)^0.2 (Fromang & Nelson 09)")
    print("do we recompute all dust densities? : ", parameters["recalc_density"])
    print(
        "do we recompute all dust or gas temperatures? : ",
        parameters["recalc_temperature"],
    )
    print(
        "do we include a bin with small dust tightly coupled to the gas? : ",
        parameters["bin_small_dust"],
    )
    print(
        "what is the maximum altitude of the 3D grid in pressure scale heights? : ",
        parameters["zmax_over_H"],
    )
    print("do we recompute all dust opacities? : ", parameters["recalc_opac"])
    print("do we plot dust opacities? : ", parameters["plot_opac"])
    print(
        "do we use pre-calculated opacities, which are located in opacity_dir directory? : ",
        parameters["precalc_opac"],
    )
    print(
        "do we run RADMC3D calculation of temperature and ray tracing? : ",
        parameters["recalc_radmc"],
    )
    if parameters["recalc_radmc"] == "Yes":
        print(
            "how many cores do we use for radmc calculation? : ", parameters["nbcores"]
        )
    print(
        "do we recompute fits file of raw flux map from image.out? This is useful if radmc has been run on a /"
        "different platform: ",
        parameters["recalc_rawfits"],
    )
    print(
        "do we recompute convolved flux map from the output of RADMC3D calculation? : ",
        parameters["recalc_fluxmap"],
    )
    print(
        "do we compute 2D analytical solution to RT equation w/o scattering? : ",
        parameters["calc_abs_map"],
    )
    if parameters["calc_abs_map"] == "Yes":
        print(
            "if so, do we assume the dust surface density equal to the gas surface density? : ",
            parameters["dustdens_eq_gasdens"],
        )
    print(
        "do we take the gas (hydro) temperature for the dust temperature? : ",
        parameters["Tdust_eq_Thydro"],
    )
    print("do we add white noise to the raw flux maps? : ", parameters["add_noise"])
    if parameters["add_noise"] == "Yes":
        print("if so, level of noise in Jy / beam", parameters["noise_dev_std"])
    print(
        "number of pixels in each direction for flux map computed by RADMC3D = ",
        parameters["nbpixels"],
    )
    print("scattering mode max for RADMC3D = ", parameters["scat_mode"])
    print("number of photon packaged used by RADMC3D", parameters["nb_photons"])
    print(
        "number of photon packaged used by RADMC3D for scattering",
        parameters["nb_photons_scat"],
    )
    print(
        "type of dust particles for calculation of dust opacity files = ",
        parameters["species"],
    )
    print(
        "name of directory with .lnk dust opacity files = ", parameters["opacity_dir"]
    )
    print("x- and y-max in final image [arcsec] = ", parameters["minmaxaxis"])
    print(
        "do we use second-order integration for ray tracing in RADMC3D? : ",
        parameters["secondorder"],
    )
    print(
        "do we flip x-axis in disc plane? (mirror symmetry x -> -x in the simulation): ",
        parameters["xaxisflip"],
    )
    print(
        "do we check beam shape by adding a source point at the origin?: ",
        parameters["check_beam"],
    )
    print(
        "do we deproject the predicted image into polar coords?:",
        parameters["deproj_polar"],
    )  # SP
    print(
        "do we compute the axisymmetric profile of the convolved intensity?:",
        parameters["axi_intensity"],
    )
    print("do we plot radial temperature profiles? : ", parameters["plot_temperature"])
    print(
        "do we override code units in hydro simulation? : ",
        parameters["override_units"],
    )
    if parameters["override_units"] == "Yes":
        print("new unit of length in meters? : ", parameters["new_unit_length"])
        print("new unit of mass in kg? : ", parameters["new_unit_mass"])

    # label for the name of the image file created by RADMC3D
    if parameters["RTdust_or_gas"] == "dust":
        parameters["label"] = (
            parameters["dir"]
            + "_o"
            + str(parameters["on"])
            + "_p"
            + str(parameters["pindex"])
            + "_r"
            + str(parameters["ratio"])
            + "_a"
            + str(parameters["amin"])
            + "_"
            + str(parameters["amax"])
            + "_nb"
            + str(parameters["nbin"])
            + "_mode"
            + str(parameters["scat_mode"])
            + "_np"
            + str("{:.0e}".format(parameters["nb_photons"]))
            + "_nc"
            + str(parameters["nsec"])
            + "_z"
            + str(parameters["z_expansion"])
            + "_xf"
            + str(parameters["xaxisflip"])
            + "_Td"
            + str(parameters["Tdust_eq_Thydro"])
        )
    else:
        if parameters["widthkms"] == 0.0:
            parameters["label"] = (
                parameters["dir"]
                + "_o"
                + str(parameters["on"])
                + "_gas"
                + str(parameters["gasspecies"])
                + "_iline"
                + str(parameters["iline"])
                + "_lmode"
                + str(parameters["lines_mode"])
                + "_ab"
                + str("{:.0e}".format(parameters["abundance"]))
                + "_vkms"
                + str(parameters["vkms"])
                + "_turbvel"
                + str(parameters["turbvel"])
                + "_nc"
                + str(parameters["nsec"])
                + "_xf"
                + str(parameters["xaxisflip"])
                + "_Td"
                + str(parameters["Tdust_eq_Thydro"])
            )
        else:
            parameters["label"] = (
                parameters["dir"]
                + "_o"
                + str(parameters["on"])
                + "_gas"
                + str(parameters["gasspecies"])
                + "_iline"
                + str(parameters["iline"])
                + "_lmode"
                + str(parameters["lines_mode"])
                + "_ab"
                + str("{:.0e}".format(parameters["abundance"]))
                + "_widthkms"
                + str(parameters["widthkms"])
                + "_turbvel"
                + str(parameters["turbvel"])
                + "_nc"
                + str(parameters["nsec"])
                + "_xf"
                + str(parameters["xaxisflip"])
                + "_Td"
                + str(parameters["Tdust_eq_Thydro"])
            )

    print("gas aspect ratio = ", parameters["aspectratio"])
    print("gas flaring index = ", parameters["flaringindex"])
    print("gas alpha turbulent viscosity = ", parameters["alphaviscosity"])
