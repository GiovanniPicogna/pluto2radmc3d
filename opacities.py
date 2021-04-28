import matplotlib.pyplot as plt
from pluto2radmc3d import *


# -------------------------------------------------------------------
# writing dustopac
# -------------------------------------------------------------------
def write_dustopac(species=["ac_opct", "Draine_Si"], scat_mode=10, nbin=20):
    print("writing dustopac.inp")
    hline = "-----------------------------------------------------------------------------\n"
    OPACOUT = open("dustopac.inp", "w")

    lines0 = ["2 \t iformat (2)\n", str(nbin) + " \t species\n", hline]
    OPACOUT.writelines(lines0)
    # put first element to 10 if dustkapscatmat_species.inp input file, or 1 if dustkappa_species.inp input file
    if scat_mode >= 3:
        inputstyle = 10
    else:
        inputstyle = 1
    for i in range(nbin):
        lines = [
            str(inputstyle)
            + " \t in which form the dust opacity of dust species is to be read\n",
            "0 \t 0 = thermal grains\n",
            species + str(i) + " \t dustkap***.inp file\n",
            hline,
        ]
        OPACOUT.writelines(lines)
    OPACOUT.close()


# -------------------------------------------------------------------
# read opacities
# -------------------------------------------------------------------
def read_opacities(filein):
    params = open(filein, "r")
    lines_params = params.readlines()
    params.close()
    lbda = []
    kappa_abs = []
    kappa_sca = []
    g = []
    for line in lines_params:
        try:
            line.split()[0][0]  # check if blank line (GWF)
        except:
            continue
        if line.split()[0][0] == "#":  # check if line starts with a # (comment)
            continue
        else:
            if len(line.split()) == 4:
                l, a, s, gg = line.split()[0:4]
            else:
                continue
        lbda.append(float(l))
        kappa_abs.append(float(a))
        kappa_sca.append(float(s))
        g.append(float(gg))

    lbda = np.asarray(lbda)
    kappa_abs = np.asarray(kappa_abs)
    kappa_sca = np.asarray(kappa_sca)
    g = np.asarray(g)
    return [lbda, kappa_abs, kappa_sca, g]


# -------------------------------------------------------------------
# plotting opacities
# -------------------------------------------------------------------
def plot_opacities(
    species="mix_2species_porous", amin=0.1, amax=1000, nbin=10, lbda1=1e-3
):
    ax = plt.gca()
    ax.tick_params(axis="both", length=10, width=1)

    plt.xlabel(r"Dust size [meters]")
    plt.ylabel(r"Opacities $[{\rm cm}^2\;{\rm g}^{-1}]$")

    absorption1 = np.zeros(nbin)
    scattering1 = np.zeros(nbin)
    sizes = np.logspace(np.log10(amin), np.log10(amax), nbin)

    for k in range(nbin):
        filein = "dustkappa_" + species + str(k) + ".inp"
        (lbda, kappa_abs, kappa_sca, g) = read_opacities(filein)

        i1 = np.argmin(np.abs(lbda - lbda1))

        # interpolation in log
        l1 = lbda[i1 - 1]
        l2 = lbda[i1 + 1]
        k1 = kappa_abs[i1 - 1]
        k2 = kappa_abs[i1 + 1]
        ks1 = kappa_sca[i1 - 1]
        ks2 = kappa_sca[i1 + 1]
        absorption1[k] = (k1 * np.log(l2 / lbda1) + k2 * np.log(lbda1 / l1)) / np.log(
            l2 / l1
        )
        scattering1[k] = (ks1 * np.log(l2 / lbda1) + ks2 * np.log(lbda1 / l1)) / np.log(
            l2 / l1
        )

    # nice colors
    c20 = [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213),
        (140, 86, 75),
        (196, 156, 148),
        (227, 119, 194),
        (247, 182, 210),
        (127, 127, 127),
        (199, 199, 199),
        (188, 189, 34),
        (219, 219, 141),
        (23, 190, 207),
        (158, 218, 229),
    ]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(c20)):
        r, g, b = c20[i]
        c20[i] = (r / 255.0, g / 255.0, b / 255.0)

    lbda1 *= 1e-3  # in mm

    plt.loglog(
        sizes,
        absorption1,
        lw=2.0,
        linestyle="solid",
        color=c20[1],
        label="$\kappa_{abs}$ at " + str(lbda1) + " mm",
    )
    plt.loglog(
        sizes,
        absorption1 + scattering1,
        lw=2.0,
        linestyle="dashed",
        color=c20[1],
        label="$\kappa_{abs}$+$\kappa_{sca}$ at " + str(lbda1) + " mm",
    )
    plt.legend()

    plt.ylim(absorption1.min(), (absorption1 + scattering1).max())
    filesaveopac = "opacities_" + species + ".pdf"
    plt.savefig("./" + filesaveopac, bbox_inches="tight", dpi=160)
    plt.clf()
