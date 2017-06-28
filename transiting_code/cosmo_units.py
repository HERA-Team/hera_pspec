#
#  cosmo_units.py
#  
#
#  Created by Danny Jacobs on 7/6/10.
#  Copyright (c) 2010 __MyCompanyName__. All rights reserved.
#


import numpy as n
from scipy import integrate
#WMAP 7Year ML
#O_l = 0.7334
#O_M = 0.1334/0.75**2+0.217
#O_M = 0.15/0.71**2

#WMAP 5 year (in the abstract of astro-ph 0803.0586v2)
O_l = 0.742
O_M = 0.25656
Ho = 71.9
O_k = 1-O_l-O_M


#other constants
c = 3e8
ckm = c/1000. #km/s

f21 = 1.421e9 #GHz

def E(z):
    return n.sqrt(O_M*(1+z)**3 + O_k*(1+z)**2 +  O_l) 

def DM(z):
    return ckm/Ho*integrate.quad(lambda z: 1/E(z),0,z)[0]    
def DA(z):
    return DM(z)/(1+z)

def r2df(r,z):
    return r*Ho*f21*E(z)/(ckm*(1+z)**2)
    
def df2r(df,z):
        return df/(Ho*f21*E(z)/(ckm*(1+z)**2))

def kparr2eta(kparr,z):
    return kparr*ckm*(1+z)**2/(2*n.pi*Ho*f21*E(z))

def eta2kparr(eta,z):
    return eta*(2*n.pi*Ho*f21*E(z))/(ckm*(1+z)**2)

def kperp2u(kperp,z):
    return kperp*DA(z)/(2*n.pi)

def u2kperp(u,z):
    return u*2*n.pi/DA(z)
def r2theta(r,z):
    return r/DA(z)
def theta2r(theta,z):
    return theta*DA(z)
def f212z(f):
    return f21/f-1
