"""
conversions.py
--------

Cosmological and instrumental
conversion functions for hera_pspec
"""
import numpy as np
from scipy import integrate
try:
    from astropy.cosmology import LambdaCDM
    _astropy = True
except:
    print("could not import astropy")
    _astropy = False


class SIunits:
    """
    fundamental constants in SI units

    c : speed of light m s-1
    ckm : speed of light in km s-1
    kb : boltzmann constant 2 kg s-2 K-1
    hp : planck constant m2 kg s-1
    sb : stefan-boltzmann constant W m-2 K-4
    f21 : 21cm frequency in Hz
    w21 : 21cm wavelength in meters
    """
    c = 2.99792458e8  # speed of light in m s-1
    ckm = c * 1e3  # speed of light in km s-1
    G = 6.67408e-11  # Newton constant m3 kg-1 s-2
    kb = 1.38064852e-23  # boltzmann constant m2 kg s-2 K-1
    hp = 6.62607004e-34  # planck constant m2 kg s-1
    sb = 5.670367e-8  # stefan boltzmann constant W m-2 K-4
    f21 = 1.420405751e9  # frequency of 21cm transition in Hz
    w21 = 0.211061140542  # 21cm wavelength in meters


class Cosmo(object):
    """
    Cosmo class for cosmological conversion functions.

    Default parameter values are Planck 2015 TT,TE,EE+lowP.
    (Table 4 of https://doi.org/10.1051/0004-6361/201525830)

    Distance measures come from:
    Hogg 1999 (astro-ph/9905116)
    Furlanetto 2006 (2006PhR...433..181F)

    Note that all distance measures are in Mpc, not h-1 Mpc.
    To convert these to h-1 Mpc, multiply by (self.H0 / 100)
    """
    # astropy load attr
    _astropy = _astropy

    # convert H0 from km s-1 Mpc-1 to s-1
    H0_2_invsec = 3.24078e-20

    def __init__(self, Om_L=0.68440, Om_b=0.04911, Om_c=0.26442, H0=67.31,
                 Om_M=None, Om_k=None):
        """


        Default parameter values are Planck 2015 TT,TE,EE+lowP.
        (Table 4 of https://doi.org/10.1051/0004-6361/201525830)

        Parameters:
        -----------
        Om_L : float, Omega Lambda naught, default=0.68440
            cosmological constant energy density fraction at z=0

        Om_b : float, Omega baryon naught, default=0.04911
            baryon energy density fraction at z=0

        Om_c : float, Omega cold dark matter naught, default=0.26442
            cdm energy density fraction at z=0

        H0 : float, Hubble constant, default=67.31 km s-1 Mpc-1

        Om_M : float, Omega matter naught, default=None [optional]
            matter energy density faction at z=0

        Om_k : float, Omega curvature naught, default=None [optional]
            curvature energy density at z=0
        """
        # Setup parameters
        if Om_M is not None:
            if np.isclose(Om_b + Om_c, Om_M, atol=1e-5) is False:
                Om_b = Om_M * 0.156635
                Om_c = Om_M * 0.843364
        else:
            Om_M = Om_b + Om_c

        if Om_k is None:
            Om_k = 1 - Om_L - Om_M

        self.Om_L = Om_L
        self.Om_b = Om_b
        self.Om_c = Om_c
        self.Om_M = Om_M
        self.Om_k = Om_k
        self.H0 = H0
        self.h = self.H0 / 100.0

        if _astropy:
            self.lcdm = LambdaCDM(H0=self.H0, Om0=self.Om_M, Ode0=self.Om_L)

    def f2z(self, freq, ghz=False):
        """
        convert frequency to redshift for 21cm line

        Parameters:
        -----------
        freq : frequency in Hz, type=float

        ghz : boolean, if True: assume freq is GHz

        Output:
        -------
        z : float
            redshift
        """
        if ghz:
            freq = freq * 1e9

        return (SIunits.f21 / freq - 1)

    def z2f(self, z, ghz=False):
        """
        convert redshift to frequency in Hz for 21cm line

        Parameters:
        -----------
        z : redshift, type=float

        ghz: boolean, if True: convert to GHz

        Output:
        -------
        freq : float
            frequency in Hz
        """
        freq = SIunits.f21 / (z + 1)
        if ghz:
            freq /= 1e9

        return freq

    def E(self, z):
        """
        ratio of hubble parameters: H(z) / H(z=0)
        Hogg99 Eqn. 14

        Parameters:
        -----------
        z : redshift, type=float
        """
        return np.sqrt(self.Om_M*(1+z)**3 + self.Om_k*(1+z)**2 + self.Om_L)

    def DC(self, z):
        """
        line-of-sight comoving distance in Mpc
        Hogg99 Eqn. 15

        Parameters:
        -----------
        z : redshift, type=float
        """
        return integrate.quad(lambda z: 1/self.E(z), 0, z)[0] * SIunits.ckm / self.H0 

    def DM(self, z):
        """
        transverse comoving distance in Mpc
        Hogg99 Eqn. 16

        Parameters:
        -----------
        z : redshift, type=float
        """
        DH = SIunits.ckm / self.H0
        if self.Om_k > 0:
            DM = DH * np.sinh(np.sqrt(self.Om_k) * self.DC(z) / DH) / np.sqrt(self.Om_k)
        elif self.Om_k < 0:
            DM = DH * np.sin(np.sqrt(np.abs(self.Om_k)) * self.DC(z) / DH) / np.sqrt(np.abs(self.Om_k))
        else:
            DM = self.DC(z)

        return DM

    def DA(self, z):
        """
        angular diameter (proper) distance in Mpc
        Hogg99 Eqn. 18

        Parameters:
        -----------
        z : redshift, type=float
        """
        return self.DM(z) / (1 + z)

    def dr2df(self, dr, z):
        """
        convert line-of-sight distance at redshift z
        to a frequency bandwidth in Hz for the 21cm line
        Furlanetto06 Eqn. 2

        Parameters:
        -----------
        dr : line-of-sight comoving distance, type=float
        z : redshift, type=float
        """
        return dr * self.E(z) * self.H0 / SIunits.ckm * SIunits.f21 / (1+z)**2

    def df2dr(self, df, z):
        """
        convert frequency bandwidth in Hz to line-of-sight
        comoving distance at redshift z
        Furlanetto06 Eqn. 2

        Parameters:
        -----------
        df : frequency bandwidth in Hz, type=float
        z : redshift, type=float
        """
        return df / self.E(z) / self.H0 * SIunits.ckm / SIunits.f21 * (1+z)**2

    def kperp2u(self, kperp, z):
        """
        convert from k-perpendicular wavevector to
        "u" wavevector in the uv-plane (wavelength) for the 21cm line.

        Parameters:
        -----------
        kperp : k-perpendicular, type=float
            transverse spatial wavevector in units Mpc-1.


        """
        pass


class Misc(object):
    """


    """
    pass



class Conversions(Misc, Cosmo):
    """
    Convert class for miscellaneous mathematical conversion
    functions. Some are tied to a cosmological model specified
    by the user, others are not. 




    """
    
    def __init__(self):
        """

        """










