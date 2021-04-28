import math
import re
import sys

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def plot_density(nbin, nsec, ncol, nrad, density, rho):
    mycolormap = "nipy_spectral"
    matplotlib.rcParams.update({"font.size": 20})
    matplotlib.rc("font", family="Arial")
    print("--------- plotting density(R,theta) ----------")
    axidens = np.zeros((nbin, ncol, nrad))
    for l in range(nbin):
        for j in range(ncol):
            for i in range(nrad):
                for k in range(nsec):
                    axidens[l, j, i] += rho[l, k, j, i]
                axidens[l, j, i] /= nsec + 0.0
    X = density.rmed * density.culength / 1.5e11  # in au
    Z = np.abs(density.tmed - np.pi / 2.0)

    fig, ax = plt.subplots(
        nrows=2, ncols=math.ceil(nbin / 2), sharey=True, sharex=True, figsize=(8.0, 8.0)
    )
    for n in range(nbin):
        if n < nbin / 2:
            i = 0
        else:
            i = 1
            ax[i][j].set_xlabel("radius [au]")
        if n % 2 == 0:
            j = 0
            ax[i][j].set_ylabel("colatitude [rad]")
        else:
            j = j + 1
        ax[i][j].set_xlim(5, 50)
        ax[i][j].set_ylim(0, 0.25)
        ax[i][j].pcolormesh(
            X,
            Z,
            axidens[n],
            norm=colors.LogNorm(vmin=1.0e-18, vmax=axidens.max()),
            cmap=mycolormap,
        )
    plt.savefig("density.pdf", dpi=320)


def plot_temperature(nbin, nsec, ncol, nrad, density, parameters):
    # Plot midplane and surface temperature profiles
    if parameters["RTdust_or_gas"] == "dust":
        Temp = np.fromfile("dust_temperature.bdat", dtype="float64")
        Temp = Temp[4:]
        Temp = Temp.reshape(nbin, nsec, ncol, nrad)
        # Keep temperature of the largest dust species
        Temp = Temp[-1, :, :, :]
    else:
        print(
            "Set of initial conditions not implemented for pluto yet. Only parameters['RTdust_or_gas'] == 'dust'"
        )
        sys.exit("I must exit!")

    # Temperature in the midplane (ncol/2 given that the grid extends on both sides about the midplane)
    # not really in the midplane because theta=pi/2 is an edge colatitude...
    Tm = Temp[:, ncol // 2, :]
    # Temperature at one surface
    Ts = Temp[:, 0, :]
    # Azimuthally-averaged radial profiles
    axiTm = np.sum(Tm, axis=0) / nsec
    axiTs = np.sum(Ts, axis=0) / nsec
    fig = plt.figure(figsize=(4.0, 3.0))
    ax = fig.gca()
    S = density.rmed * density.culength / 1.5e11  # radius in a.u.
    # gas temperature in hydro simulation in Kelvin (assuming T in R^-1/2, no matter
    # the value of the gas flaring index in the simulation)
    Tm_model = (
        parameters["aspectratio"]
        * parameters["aspectratio"]
        * density.cutemp
        * density.rmed ** (-1.0 + 2.0 * parameters["flaringindex"])
    )
    ax.plot(S, axiTm, "bo", markersize=1.0, label="midplane")
    ax.plot(S, Tm_model, "b--", markersize=1.0, label="midplane hydro")
    ax.plot(S, axiTs, "rs", markersize=1.0, label="surface")
    ax.set_xlabel(r"$R ({\rm au})$", fontsize=12)
    ax.set_ylabel(r"$T ({\rm K})$", fontsize=12)
    # ax.set_xlim(20.0, 100.0) # cuidadin!
    ax.set_xlim(S.min(), S.max())
    # ax.set_ylim(10.0, 150.0)  # cuidadin!
    ax.set_ylim(Tm.min(), Ts.max())
    ax.tick_params(axis="both", direction="in", top="on", right="on")
    ax.tick_params(axis="both", which="minor", top="on", right="on", direction="in")
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(frameon=False)
    fig.add_subplot(ax)
    filenameT = "T_R_" + parameters["label"] + ".pdf"
    fig.savefig(filenameT, dpi=180, bbox_inches="tight")
    fig.clf()
    # Save radial profiles in an ascii file
    filenameT2 = "T_R_" + parameters["label"] + ".dat"
    TEMPOUT = open(filenameT2, "w")
    TEMPOUT.write(
        "# radius [au] \t T_midplane_radmc3d \t T_surface_radmc3d \t T_midplane_hydro\n"
    )
    for i in range(nrad):
        TEMPOUT.write(
            "%f \t %f \t %f \t %f\n" % (S[i], axiTm[i], axiTs[i], Tm_model[i])
        )
    TEMPOUT.close()
    # free RAM memory
    del Temp


def plot_image(
    nx, cdelt, lbda0, strflux, convolved_intensity, jybeamfileout, parameters
):
    # --------------------
    # plotting image panel
    # --------------------
    matplotlib.rcParams.update({"font.size": 20})
    matplotlib.rc("font", family="Arial")
    fontcolor = "white"

    # name of pdf file for final image
    fileout = re.sub(".fits", ".pdf", jybeamfileout)
    fig = plt.figure(figsize=(8.0, 8.0))
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
    a0 = cdelt * (nx // 2.0 - dpix)  # >0
    a1 = -cdelt * (nx // 2.0 + dpix)  # <0
    d0 = -cdelt * (nx // 2.0 - dpix)  # <0
    d1 = cdelt * (nx // 2.0 + dpix)  # >0
    # da positive definite
    if parameters["minmaxaxis"] < abs(a0):
        da = parameters["minmaxaxis"]
    else:
        da = np.max(abs(a0), abs(a1))
    mina = da
    maxa = -da
    xlambda = mina - 0.166 * da
    ax.set_ylim(-da, da)
    ax.set_xlim(da, -da)  # x (=R.A.) increases leftward
    dmin = -da
    dmax = da

    # x- and y-ticks and labels
    ax.tick_params(top="on", right="on", length=5, width=1.0, direction="out")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.set_xticks(ax.get_yticks())    # set same ticks in x and y in cartesian
    # ax.set_yticks(ax.get_xticks())    # set same ticks in x and y in cartesian
    ax.set_xlabel("RA offset [arcsec]")
    ax.set_ylabel("Dec offset [arcsec]")

    # imshow does a bilinear interpolation. You can switch it off by putting
    # interpolation='none'
    min = convolved_intensity.min()
    max = convolved_intensity.max()
    CM = ax.imshow(
        convolved_intensity,
        origin="lower",
        cmap=parameters["mycolormap"],
        interpolation="bilinear",
        extent=[a0, a1, d0, d1],
        vmin=min,
        vmax=max,
        aspect="auto",
    )

    # Add wavelength in top-left corner
    strlambda = "$\lambda$=" + str(round(lbda0, 2)) + "mm"  # round to 2 decimals
    if lbda0 < 0.01:
        strlambda = "$\lambda$=" + str(round(lbda0 * 1e3, 2)) + "$\mu$m"
    ax.text(
        xlambda,
        dmax - 0.166 * da,
        strlambda,
        fontsize=20,
        color="white",
        weight="bold",
        horizontalalignment="left",
    )

    # Add + sign at the origin
    ax.plot(0.0, 0.0, "+", color="white", markersize=10)
    """
    if check_beam == 'Yes':
        ax.contour(convolved_intensity,levels=[0.5*convolved_intensity.max()],color='black', linestyles='-',origin='lower',extent=[a0,a1,d0,d1])
    """

    # plot beam
    if parameters["plot_tau"] == "No":
        from matplotlib.patches import Ellipse

        e = Ellipse(
            xy=[xlambda, dmin + 0.166 * da],
            width=parameters["bmin"],
            height=parameters["bmaj"],
            angle=parameters["bpaangle"] + 90.0,
        )
        e.set_clip_box(ax.bbox)
        e.set_facecolor("white")
        e.set_alpha(1.0)
        ax.add_artist(e)

    # plot color-bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb = plt.colorbar(CM, cax=cax, orientation="horizontal")
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction="out")
    cax.xaxis.set_major_locator(plt.MaxNLocator(6))
    # title on top
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(strflux)
    cax.xaxis.labelpad = 8

    plt.savefig("./" + fileout, bbox_inches="tight", dpi=160)
    plt.clf()
