#-*- coding: UTF-8 -*- 
from __future__ import division
import os
import math
import random
import numpy as np
from astropy.cosmology import Planck15 as cosmo
import astropy.units as aus
from astropy.cosmology import z_at_value
import cosmolopy.distance as cd
from scipy import optimize
from scipy.stats import skewnorm


### constants, the units are SI ###

GG          = 6.67408e-11

m_earth     = 5.965e24

m_sun       = 1.9891e30

mpc         = 1e6*3.084e16

cc          = 2.998e8

R0          = 1e8

m_sun_in_s  = m_sun*GG/cc**3 # the unit is sec

mpc_in_s    = mpc/cc # the unit is sec

T_1yr       = 365*24*3600.0

tto         = 4*T_1yr

R_1AU       = 1.495978707e11

gamma_E     = 0.577

path_interp = "/home/ShuaiLiu/Projects/mulfreq_GW_astron/eLISA/files_exe/data_interp/"


### make directory ###

def make_directory(path):

  isExists = os.path.exists(path)

  if not isExists:

    os.makedirs(path)

    print(path+" is created successfully !")

    return True

  else:

    print(path+" has existed already !")

    return False



### random parameters ###


def merger_number_flat(tt):

  num_list = []; ln_rate_list = []; 
 
  np.random.seed(0)

  ln_rate_list = skewnorm.rvs(-2.6, loc = 4.3, scale = 1.2, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list


def merger_number_salp(tt):

  num_list = []; ln_rate_list = [];
 
  np.random.seed(1)

  ln_rate_list = skewnorm.rvs(-2.4, loc = 5.4, scale = 1.2, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list


def random_mass_flat(num):

  m1r_list = []; m2r_list = [];

  count = 0 

  while len(m1r_list) < num:

    count = count+1

    random.seed(count+num)

    ran1  = random.random()

    ran2  = random.random()

    if ran1 < ran2:

      ran3 = ran1

      ran1 = ran2

      ran2 = ran3

    m1r    = 5.0*20**ran1

    m2r    = 5.0*20**ran2

    if 10.0 < m1r+m2r < 100.0:

      m1r_list.append(m1r)

      m2r_list.append(m2r)

  return [m1r_list, m2r_list]


def random_mass_salp(num):

  m1r_list=[]; m2r_list=[];

  count = 0

  while len(m1r_list) < num:

    count = count+1
  
    random.seed(count+num)
    
    ran3  = random.random()
    
    ran4  = random.random()
    
    m1r   = 5*(1-ran3*(1-20**(-1.35)))**(-20.0/27)
    
    m2r   = 5.0+(m1r-5)*ran4

    if 10.0 < m1r+m2r < 100.0:
    
      m1r_list.append(m1r)
    
      m2r_list.append(m2r)

  return [m1r_list, m2r_list]


def random_comoving_distance(num):

  rcm_max  = np.array(cosmo.comoving_distance(2.0)) # the unit is Mpc

  rcm_list = [];
  
  for count in xrange(num):

    random.seed(count+3+num)

    ran5 = random.random()
    
    rcm  = rcm_max*ran5**(1.0/3)
    
    rcm_list.append(rcm)

  return rcm_list 



def random_redshift(num):

  zz0 = []; rrc0 = [];

  for line0 in open(path_interp+"redshift0_comoving_distance0.txt"):

    line1 = line0.split()

    zz0.append(float(line1[0]))

    rrc0.append(float(line1[1]))

  zzp = sorted(zz0); rrcp = sorted(rrc0);

  comoving_distance = random_comoving_distance(num)

  redshift          = np.interp(comoving_distance, rrcp, zzp)

  return redshift


def random_redshift_Newtoin(num):

  rcm_max = np.array(cosmo.comoving_distance(2.0)) # the unit is Mpc

  rcm_list = [];
  
  for count in xrange(num):

    random.seed(count+3+num)

    ran5 = random.random()
    
    rcm  = rcm_max*ran5**(1.0/3)
    
    rcm_list.append(rcm)

  zz_list = [];

  for ii in xrange(len(rcm_list)):
   
    def f(xx):

      return (np.array(cosmo.comoving_distance(xx))-rcm_list[ii])
  
    zz = optimize.newton(f, 1.0)
    
    zz_list.append(zz)

  return zz_list



def random_angles_bar(num):

  theta_s_bar_list = []; phi_s_bar_list = []; theta_l_bar_list = []; phi_l_bar_list = []; 
  
  for count in xrange(num):

    np.random.seed(count+num)

    ran7, ran8, ran9, ran10 = np.random.rand(4)

    theta_s_bar = np.arccos(2.0*ran7-1) 

    phi_s_bar   = 2.0*np.pi*ran8  

    theta_l_bar = np.arccos(2.0*ran9-1) 

    phi_l_bar   = 2.0*np.pi*ran10

    theta_s_bar_list.append(theta_s_bar)

    phi_s_bar_list.append(phi_s_bar)

    theta_l_bar_list.append(theta_l_bar)

    phi_l_bar_list.append(phi_l_bar)

  return [theta_s_bar_list, phi_s_bar_list, theta_l_bar_list, phi_l_bar_list]





def random_ttrc(num, tt):

  ttrc_list = [];

  for count in xrange(num):

    random.seed(count+10)

    ran12 = random.random()

    ttrc  = tt*ran12*365*24*3600.0
    
    ttrc_list.append(ttrc)

  return ttrc_list


def random_ttrc_critical(num, tt):

  ttrc_list = [];

  for count in xrange(num):

    random.seed(count+11)

    ran13 = random.random()

    ttrc  = (tt+ran13*10)*365*24*3600.0
    
    ttrc_list.append(ttrc)

  return ttrc_list


  
### calculate the luminosity-distance, the unit is Mpc ###

def lum_dis(zz):
  
  cosmo    = {'omega_M_0' : 0.306, 'omega_lambda_0' : 0.694, 'h' : 0.679}
  
  cosmo    = cd.set_omega_k_0(cosmo)
  
  distance = cd.luminosity_distance(zz, **cosmo)
  
  return distance

#def lum_dis(zz):
#  
#  distance = cosmo.luminosity_distance(zz)
#  
#  return distance


def angles_solar_to_dec(tt, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar):

  phi_bar      = 2.0*np.pi*tt/T_1yr


  cos_theta_s  = 1.0/2*np.cos(theta_s_bar)-np.sqrt(3)/2.0*np.sin(theta_s_bar)*np.cos(phi_bar-phi_s_bar)

  theta_s      = np.arccos(cos_theta_s)


  numerator1   = np.sqrt(3)*np.cos(theta_s_bar)+np.sin(theta_s_bar)*np.cos(phi_bar-phi_s_bar)

  denominator1 = 2*np.sin(theta_s_bar)*np.sin(phi_bar-phi_s_bar)

  if numerator1 < 0.0:

    phi_s = 2*np.pi*tt/T_1yr+np.arctan2(numerator1, denominator1)+2.0*np.pi

  else:

    phi_s = 2*np.pi*tt/T_1yr+np.arctan2(numerator1, denominator1)

 
  z_dot_n         = np.cos(theta_s)

  L_dot_z         = 1.0/2*np.cos(theta_l_bar)-np.sqrt(3)/2.0*np.sin(theta_l_bar)*np.cos(phi_bar-phi_l_bar)

  L_dot_n         = np.cos(theta_l_bar)*np.cos(theta_s_bar)+np.sin(theta_l_bar)*np.sin(theta_s_bar)*np.cos(phi_l_bar-phi_s_bar)

  n_dot_L_cross_z = 1.0/2*np.sin(theta_l_bar)*np.sin(theta_s_bar)*np.sin(phi_l_bar-phi_s_bar)-np.sqrt(3.0)/2*np.cos(phi_bar)*(np.cos(theta_l_bar)*np.sin(theta_s_bar)*np.sin(phi_s_bar)-np.cos(theta_s_bar)*np.sin(theta_l_bar)*np.sin(phi_l_bar))-np.sqrt(3.0)/2*np.sin(phi_bar)*(np.cos(theta_s_bar)*np.sin(theta_l_bar)*np.cos(phi_l_bar)-np.cos(theta_l_bar)*np.sin(theta_s_bar)*np.cos(phi_s_bar))

  numerator2      = L_dot_z-L_dot_n*z_dot_n

  denominator2    = n_dot_L_cross_z
  
  if numerator2 < 0.0:

    psi_s         = np.arctan2(numerator2, denominator2)+2.0*np.pi 

  else:

   psi_s          = np.arctan2(numerator2, denominator2) 


  return theta_s, phi_s, psi_s


### calculate the waveform h(f) in frequency domain ### 



def freq_red_i_f_0PN(mtr, mcr, ttcr, zz):

  coeff   = 5/(96*(np.pi)**(8.0/3))

  freqr_i = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr))**(-3.0/8)

  if ttcr <= tto/(1+zz):

    freqr_f = 1.0/(np.pi*6**(3.0/2))/(mtr*m_sun_in_s)
 
  else:
  
    freqr_f = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr-tto/(1+zz)))**(-3.0/8)

  freq_red_i = freqr_i/(1+zz)

  freq_red_f = freqr_f/(1+zz)

  return freq_red_i, freq_red_f


def freq_red_list_0PN(mtr, mcr, ttcr, zz):
 
  freq_red_i, freq_red_f = freq_red_i_f_0PN(mtr, mcr, ttcr, zz)

  index_i       = math.log(freq_red_i, 10)

  index_f       = math.log(freq_red_f, 10)

  index_range   = np.linspace(index_i, index_f, 40, endpoint = True)

  freq_red_list = np.array(np.power(10, index_range)) 

  return freq_red_list


### polarization phase ###

def phip(tt_red, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar):

  theta_s, phi_s, psi_s = angles_solar_to_dec(tt_red, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  phi_s1       = phi_s

  phi_s2       = phi_s-np.pi/4

  L_dot_n      = np.cos(theta_l_bar)*np.cos(theta_s_bar)+np.sin(theta_l_bar)*np.sin(theta_s_bar)*np.cos(phi_l_bar-phi_s_bar)

  F_plus1      = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s1)*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*phi_s1)*np.sin(2*psi_s)

  F_cross1     = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s1)*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*phi_s1)*np.cos(2*psi_s)

  F_plus2      = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s2)*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*phi_s2)*np.sin(2*psi_s)

  F_cross2     = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s2)*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*phi_s2)*np.cos(2*psi_s)

  numerator1   = 2*L_dot_n*F_cross1

  denominator1 = (1+L_dot_n**2)*F_plus1

  numerator2   = 2*L_dot_n*F_cross2

  denominator2 = (1+L_dot_n**2)*F_plus2

  phi_p1       = np.arctan2(numerator1, denominator1) 

  phi_p2       = np.arctan2(numerator2, denominator2)

  return phi_p1, phi_p2


### Eccentric phase ###

def phiE(f_red, mc_red, f0_red, ee0):

  phi_E = -4239.0/11696*(mc_red*m_sun_in_s*np.pi)**(-5.0/3)*(f0_red**(19.0/9)/f_red**(34.0/9))*ee0**2

  return phi_E


### Doppler phase ###

def phiD(f_red, tt_red, theta_s_bar, phi_s_bar):

  phi_bar = 2*np.pi*tt_red/T_1yr

  phi_D   = 2*np.pi*f_red*R_1AU/cc*np.sin(theta_s_bar)*np.cos(phi_bar-phi_s_bar)

  return phi_D

def psi_f_0PN(f_red, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, eta):

  xx    = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)

#  psi_f = 2*np.pi*f_red*ttc_red-phic+3.0/128*(np.pi*mc_red*m_sun_in_s*f_red)**(-5.0/3)*(1+0)

  psi_f = 2*np.pi*f_red*ttc_red-phic-np.pi/4+3.0/128*(mc_red*m_sun_in_s*np.pi*f_red)**(-5.0/3)*(alpha0+alpha2*xx+alpha3*xx**(3.0/2)+alpha4*xx**2)

  return psi_f 


#def psi_f_2PN(f_red, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, eta):
#
#  xx    = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)
#
##  psi_f = 2*np.pi*f_red*ttc_red-phic+3.0/128*(np.pi*mc_red*m_sun_in_s*f_red)**(-5.0/3)*(1+(3715.0/756+55.0/9*eta)*eta**(-2.0/5)*(np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)-16*np.pi*eta**(-3.0/5)*(np.pi*mc_red*m_sun_in_s*f_red)+(15293365.0/508032+27145.0/504*eta+3085.0/72*eta**2)*eta**(-4.0/5)*(np.pi*mc_red*m_sun_in_s*f_red)**(4.0/3))
#
#  psi_f = 2*np.pi*f_red*ttc_red-phic-np.pi/4+3.0/128*(mc_red*m_sun_in_s*np.pi*f_red)**(-5.0/3)*(alpha0+alpha2*xx+alpha3*xx**(3.0/2)+alpha4*xx**2+alpha5*xx**(5.0/2)+alpha6*xx**3+alpha7*xx**(7.0/2))
#
#  return psi_f 


def psi_f_2PN(f_red, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, eta):

  xx    = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)

  psi_f = 2*np.pi*f_red*ttc_red-phic-np.pi/4+3.0/128*(mc_red*m_sun_in_s*np.pi*f_red)**(-5.0/3)*(alpha0+alpha2*xx+alpha3*xx**(3.0/2)+alpha4*xx**2+alpha5*xx**(5.0/2)+alpha6*xx**3)

  return psi_f 

def tt_f_2PN(f_red, mc_red, ttc_red, tau0, tau2, tau3, tau4, tau5, tau6, tau_e, eta):

  xx   = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)

  tt_f = ttc_red-5.0/256*(mc_red*m_sun_in_s)**(-5.0/3)*(np.pi*f_red)**(-8.0/3)*(tau0+tau2*xx+tau3*xx**(3.0/2)+tau4*xx**2+tau5*xx**(5.0/2)+tau6*xx**3)+tau_e

  return tt_f 


def tt_f_0PN(f_red, mc_red, ttc_red):

  tt_f = ttc_red-5.0/256*(mc_red*m_sun_in_s)**(-5.0/3)*(np.pi*f_red)**(-8.0/3) 

  return tt_f


def ampli_f_0PN(dL, mc_red, f_red):   

  return np.sqrt(5.0/24)*(np.pi)**(-2.0/3)*1/(dL*mpc_in_s)*(mc_red*m_sun_in_s)**(5.0/6)*(f_red)**(-7.0/6)
  

# averageing over the pattern functions #

def hh_ff_2PN_APF(m1r, m2r, zz, ttcr, f0_red, ee0):
    
  mtr         = m1r+m2r
  
  mt_red      = mtr*(1+zz)
 
  mcr         = (m1r*m2r)**0.6/(m1r+m2r)**0.2
  
  mc_red      = mcr*(1+zz)
   
  eta         = m1r*m2r/mtr**2
 
  ttc_red     = ttcr*(1+zz)
  
  phic        = -np.pi/4
 
  dL          = lum_dis(zz)
   
  f_red_list  = freq_red_list_0PN(mtr, mcr, ttcr, zz)
 
  f_red_isco  = 1.0/(np.pi*6**(3.0/2)*mt_red*m_sun_in_s)

  xx          = (np.pi*mc_red*m_sun_in_s*f_red_list)**(2.0/3)*eta**(-2.0/5)

  xx0         = (np.pi*mc_red*m_sun_in_s*f_red_isco)**(2.0/3)*eta**(-2.0/5)

  alpha0      = 1
  
  alpha2      = 3715.0/756+55.0/9*eta
  
  alpha3      = -16*np.pi
  
  alpha4      = 15293365.0/508032+27145.0/504*eta+3085.0/72*eta**2

  alpha5      = (38645.0/756-65.0/9*eta)*(1+3.0/2*np.log(xx/xx0))*np.pi

  alpha6      = (11583231236531.0/4694215680-640.0/3*np.pi**2-6848.0/21*gamma_E-3424.0/21*np.log(16*xx)+(-15737765635.0/3048192+2255.0/12*np.pi**2)*eta+76055.0/1728*eta**2-127825.0/1296*eta**3)

  
  
  tau0        = 1

  tau2        = 4.0/3*(743.0/336+11.0/4*eta)

  tau3        = -8.0/5*(4*np.pi)

  tau4        = 3058673.0/508032+5429.0/504*eta+617.0/72*eta**2 
  
  tau5        = -(7729.0/252-13.0/3*eta)*np.pi

  tau6        = (-10052469856691.0/23471078400+128.0/3*np.pi**2+6848.0/105*gamma_E+3424.0/105*np.log(16*xx)+(3147553127.0/3048192-451.0/12*np.pi**2)*eta-15211.0/1728*eta**2+25565.0/1296*eta**3)

  tau_e       = 785.0/110008*(mc_red*m_sun_in_s)**(-5.0/3)*np.pi**(-8.0/3)*f0_red**(19.0/9)/f_red_list**(34.0/9)*ee0**2


  tt_red_list = tt_f_2PN(f_red_list, mc_red, ttc_red, tau0, tau2, tau3, tau4, tau5, tau6, tau_e, eta)

  psi_f       = psi_f_2PN(f_red_list, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, eta)
    
  phi_E       = phiE(f_red_list, mc_red, f0_red, ee0) 

  AA          = 1.0/(np.sqrt(30.0)*np.pi**(2.0/3))*(mc_red*m_sun_in_s)**(5.0/6)/(dL*mpc_in_s)

  h_f_list    = np.sqrt(3.0)/2*AA*f_red_list**(-7.0/6)*np.exp(1j*psi_f+1j*phi_E)

  return f_red_list, tt_red_list, h_f_list 


def hh_ff_2PN_NAPF(m1r, m2r, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):

  ff_list, tt_list, h_f_list = hh_ff_2PN_APF(m1r, m2r, zz, ttcr, f0_red, ee0)

  L_dot_n = np.cos(theta_l_bar)*np.cos(theta_s_bar)+np.sin(theta_l_bar)*np.sin(theta_s_bar)*np.cos(phi_l_bar-phi_s_bar)


  signal1_list = []; signal2_list = [];

  for ii in range(len(h_f_list)):

    ff = ff_list[ii]

    tt = tt_list[ii]

    hh = h_f_list[ii]

    theta_s, phi_s, psi_s = angles_solar_to_dec(tt, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)
  
    F_plus1   = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s)*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*phi_s)*np.sin(2*psi_s)
  
    F_cross1  = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s)*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*phi_s)*np.cos(2*psi_s)
    
    F_plus2   = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*(phi_s-np.pi/4))*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*(phi_s-np.pi/4))*np.sin(2*psi_s)
  
    F_cross2  = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*(phi_s-np.pi/4))*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*(phi_s-np.pi/4))*np.cos(2*psi_s)
  
    AA1_tilde = np.sqrt((1+L_dot_n**2)**2*F_plus1**2+4*L_dot_n**2*F_cross1**2)
    
    AA2_tilde = np.sqrt((1+L_dot_n**2)**2*F_plus2**2+4*L_dot_n**2*F_cross2**2)
  
    phi_p1, phi_p2 = 0, 0 #phip(tt, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)
  
    phi_D     = 0 #phiD(ff, tt, theta_s_bar, phi_s_bar)

  
    signal1   = hh*5.0/4*AA1_tilde*np.exp(-1j*(phi_p1+phi_D))
  
    signal2   = hh*5.0/4*AA2_tilde*np.exp(-1j*(phi_p2+phi_D))

    signal1_list.append(signal1)

    signal2_list.append(signal2)


  return np.array(ff_list), np.array(signal1_list), np.array(signal2_list)



def hh_ff_amplitude_0PN(m1r, m2r, zz, ttrc):

  mtr        = m1r+m2r

  mcr        = (m1r*m2r)**0.6/(m1r+m2r)**0.2

  mc_red     = mcr*(1+zz)

  dL         = lum_dis(zz)
 
  f_red_list = freq_red_list_0PN(mtr, mcr, ttrc, zz)

  h_ampli    = np.sqrt(4.0/5)*ampli_f_0PN(dL, mc_red, f_red_list) # the factor np.sqrt(4/5) comes from the averaging inclinaltion    
  return f_red_list, h_ampli


##########################################

def hh_ff_2PN_APF_p(mcr, eta, zz, ttcr, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
  
# The luminosity(redshift) couples with mass, time and frequency, we can not measure the mass time and frequency independly. So when we estimate the luminosity(redshift), we should make mass, time and frequency unchanged with redshift.

  mc_red      = mcr*(1+zz0)
   
  ttc_red     = ttcr*(1+zz0)
   
  phic        = -np.pi/4
 
  dL          = lum_dis(zz)
 
  mt_red      = mc_red/eta**(3.0/5)
  
  f_red_list  = freq_red_list_0PN(mtr0, mcr0, ttcr0, zz0)

  f_red_isco  = 1.0/(np.pi*6**(3.0/2)*mt_red*m_sun_in_s)

  xx          = (np.pi*mc_red*m_sun_in_s*f_red_list)**(2.0/3)*eta**(-2.0/5)

  xx0         = (np.pi*mc_red*m_sun_in_s*f_red_isco)**(2.0/3)*eta**(-2.0/5)

  alpha0      = 1
  
  alpha2      = 3715.0/756+55.0/9*eta
  
  alpha3      = -16*np.pi
  
  alpha4      = 15293365.0/508032+27145.0/504*eta+3085.0/72*eta**2
   
  alpha5      = (38645.0/756-65.0/9*eta)*(1+3.0/2*np.log(xx/xx0))*np.pi

  alpha6      = (11583231236531.0/4694215680-640.0/3*np.pi**2-6848.0/21*gamma_E-3424.0/21*np.log(16*xx)+(-15737765635.0/3048192+2255.0/12*np.pi**2)*eta+76055.0/1728*eta**2-127825.0/1296*eta**3)

  
  tau0        = 1

  tau2        = 4.0/3*(743.0/336+11.0/4*eta)

  tau3        = -8.0/5*(4*np.pi)

  tau4        = 3058673.0/508032+5429.0/504*eta+617.0/72*eta**2 

  tau5        = -(7729.0/252-13.0/3*eta)*np.pi

  tau6        = (-10052469856691.0/23471078400+128.0/3*np.pi**2+6848.0/105*gamma_E+3424.0/105*np.log(16*xx)+(3147553127.0/3048192-451.0/12*np.pi**2)*eta-15211.0/1728*eta**2+25565.0/1296*eta**3)

  tau_e       = 785.0/110008*(mc_red*m_sun_in_s)**(-5.0/3)*np.pi**(-8.0/3)*f0_red**(19.0/9)/f_red_list**(34.0/9)*ee0**2

#  f_red_list  = freq_red_list_0PN(mtr0, mcr0, ttcr0, zz0)

  tt_red_list = tt_f_2PN(f_red_list, mc_red, ttc_red, tau0, tau2, tau3, tau4, tau5, tau6, tau_e, eta)

  psi_f       = psi_f_2PN(f_red_list, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, eta)
  
  phi_E       = phiE(f_red_list, mc_red, f0_red, ee0) 
  
  AA          = 1.0/(np.sqrt(30)*np.pi**(2.0/3))*(mc_red*m_sun_in_s)**(5.0/6)/(dL*mpc_in_s)

  h_f_list    = np.sqrt(3.0)/2*AA*f_red_list**(-7.0/6)*np.exp(1j*psi_f+1j*phi_E)

  return f_red_list, tt_red_list, h_f_list 


def hh_ff_2PN_NAPF_p(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):

  ff_list, tt_list, h_f_list = hh_ff_2PN_APF_p(mcr, eta, zz, ttcr, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)

  L_dot_n = np.cos(theta_l_bar)*np.cos(theta_s_bar)+np.sin(theta_l_bar)*np.sin(theta_s_bar)*np.cos(phi_l_bar-phi_s_bar)

  incl    = np.arccos(L_dot_n)

  incl    = incl-del_incl

  L_dot_n = np.cos(incl)

  signal1_list = []; signal2_list = [];

  for ii in range(len(h_f_list)):

    ff = ff_list[ii]

    tt = tt_list[ii]

    hh = h_f_list[ii]

    theta_s, phi_s, psi_s = angles_solar_to_dec(tt, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)
  
    F_plus1   = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s)*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*phi_s)*np.sin(2*psi_s)
  
    F_cross1  = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s)*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*phi_s)*np.cos(2*psi_s)
    
    F_plus2   = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*(phi_s-np.pi/4))*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*(phi_s-np.pi/4))*np.sin(2*psi_s)
  
    F_cross2  = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*(phi_s-np.pi/4))*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*(phi_s-np.pi/4))*np.cos(2*psi_s)
  
    AA1_tilde = np.sqrt((1+L_dot_n**2)**2*F_plus1**2+4*L_dot_n**2*F_cross1**2)
    
    AA2_tilde = np.sqrt((1+L_dot_n**2)**2*F_plus2**2+4*L_dot_n**2*F_cross2**2)
  
    phi_p1, phi_p2 = phip(tt, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)
  
    phi_D     = phiD(ff, tt, theta_s_bar, phi_s_bar)
  
    signal1   = hh*5.0/4*AA1_tilde*np.exp(-1j*(phi_p1+phi_D))
  
    signal2   = hh*5.0/4*AA2_tilde*np.exp(-1j*(phi_p2+phi_D))

    signal1_list.append(signal1)

    signal2_list.append(signal2)


  return np.array(ff_list), np.array(signal1_list), np.array(signal2_list)




### calculate the Sn(f) ###


def Sn_f_TQ_SA(f_list):

  LL = np.sqrt(3.0)*10**8; Sa = 1e-30; Sx = 1e-24

  Snf_list = [];

  for ff1 in f_list:
    
    Snf1 = 20.0/3*1/LL**2*(4*Sa/(2*np.pi*ff1)**4*(1+1e-4/ff1)+Sx)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list


def Sn_f_TQ_NSA(f_list):

  LL        = np.sqrt(3.0)*10**8; Sa = 1e-30; Sx = 1e-24

  Snf_list = 1/LL**2*(4*Sa/(2*np.pi*f_list)**4*(1+1e-4/f_list)+Sx)*(1+(f_list/(0.41*cc/(2*LL)))**2)

  return Snf_list



### LISA ###

def Sh_f_LISA_04(f_list):

  Sh_NSA = 9.18e-52*f_list**(-4)+1.59e-41+9.18e-38*f_list**2

  Sh_gal = 2.1e-45*f_list**(-7/3)

  Sh_exgal = 4.2e-47*f_list**(-7/3)

  Sh_f = Sh_NSA+Sh_gal+Sh_exgal

  return Sh_f



def Sc(freq):
 
  alpha, beta, kappa, gamma, f_k = 0.171, 292, 1020, 1608, 0.00215 # 1yr
  
  Sc = 9e-45*freq**(-7.0/3)*np.exp(-freq**alpha+beta*freq*np.sin(kappa*freq))*(1.0+np.tanh(gamma*(f_k-freq)))
 
  return Sc
  

def Sn_f_LISA_SA(f_list):

  f_list = np.array(f_list)
 
  LL = 2.5e9; f_star = 19.09e-3;
  
  P_OMS = (1.5e-11)**2*(1+(2e-3/f_list)**4);
  
  Pacc = (3e-15)**2*(1+(0.4e-3/f_list)**2)*(1+(f_list/(8e-3))**4)
  
  Sc_f = 0 #Sc(f_list)

  #Sc_f = np.array([Sc(f_list[ii]) if f_list[ii] < 1e-1 else 0 for ii in range(len(f_list))])
  
  Snf = 20.0/3*LL**-2*(P_OMS+4.0*Pacc/(2*np.pi*f_list)**4)*(1+0.6*(f_list/f_star)**2)+Sc_f 
   
  return Snf 


def Sn_f_LISA_NSA(f_list):

  f_list = np.array(f_list)
 
  LL = 2.5e9; f_star = 19.09e-3;
  
  P_OMS = (1.5e-11)**2*(1+(2e-3/f_list)**4);
  
  Pacc = (3e-15)**2*(1+(0.4e-3/f_list)**2)*(1+(f_list/(8e-3))**4)
  
  Sc_f = 0 #Sc(f_list)

  #Sc_f = np.array([Sc(f_list[ii]) if f_list[ii] < 1e-1 else 0 for ii in range(len(f_list))])
  
  Snf = LL**-2*(P_OMS+4.0*Pacc/(2*np.pi*f_list)**4)*(1+0.6*(f_list/f_star)**2)+Sc_f 
  
  return Snf 

### eLISA, N1A1 ###

def Sn_f_N1A1(f_list):

  LL = 1.0e9; Snsn = 1.98e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-28/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list


### eLISA, N1A2 ###

def Sn_f_N1A2(f_list):

  LL = 2.0e9; Snsn = 2.22e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-28/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list

### eLISA, N1A5 ###

def Sn_f_N1A5(f_list):

  LL = 5.0e9; Snsn = 2.96e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-28/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list

### eLISA, N2A1 ###

def Sn_f_N2A1(f_list):

  LL = 1.0e9; Snsn = 1.98e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-30/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list


### eLISA, N2A2 ###

def Sn_f_N2A2(f_list):

  LL = 2.0e9; Snsn = 2.22e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-30/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list

def Sn_f_N2A2_NSA(f_list):

  LL = 2.0e9; Snsn = 2.22e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-30/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list

### eLISA, N2A5 ###

def Sn_f_N2A5(f_list):

  LL = 5.0e9; Snsn = 2.96e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-30/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list


def Sn_f_N2A5_NSA(f_list):

  LL = 5.0e9; Snsn = 2.96e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-30/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list

### calculate the SNR ###

def SNR_calc_0PN(m1r, m2r, zz, ttrc):

  ff, signal1_f = hh_ff_amplitude_0PN(m1r, m2r, zz, ttrc)

  Sn_f          = Sn_f_LISA_SA(ff)

  integrand1    = [signf1.conjugate()*signf1/Snf for (signf1, Snf) in zip(signal1_f, Sn_f)]
  
  SNR1_sq       = 4.0*np.trapz(integrand1, ff).real

  SNR2_sq       = SNR1_sq 
  
  SNR_sq        = SNR1_sq+SNR2_sq

  SNR           = np.sqrt(SNR_sq)

  return SNR



def SNR_calc_2PN(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):
  
  ff, signal1_f, signal2_f = hh_ff_2PN_NAPF(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0)

  Sn_f = Sn_f_LISA_NSA(ff)

  integrand1 = [signf1.conjugate()*signf1/Snf for (signf1, Snf) in zip(signal1_f, Sn_f)]
  
  integrand2 = [signf2.conjugate()*signf2/Snf for (signf2, Snf) in zip(signal2_f, Sn_f)]

  SNR1_sq = 4.0*np.trapz(integrand1, ff).real

  SNR2_sq = 4.0*np.trapz(integrand2, ff).real

  SNR_sq = SNR1_sq+SNR2_sq

  SNR = np.sqrt(SNR_sq)

  return SNR



def SNR_aLIGO(f_list, signal1_f, signal2_f, Sn_f):

  integrand1 = [signf1.conjugate()*signf1/Snf for (signf1, Snf) in zip(signal1_f, Sn_f)]
  
  #integrand2 = [signf2.conjugate()*signf2/Snf for (signf2, Snf) in zip(signal2_f, Sn_f)]

  SNR1 = np.sqrt(4.0*np.trapz(integrand1, f_list).real)

  #SNR2 = np.sqrt(4.0*np.trapz(integrand2, f_list).real)

  #SNR = np.sqrt(SNR1**2+SNR2**2)
  
  SNR = SNR1
  
  return SNR 



### parameters estimation ###

def inner_product(h1_f, h2_f, Sn_f, f_list):

  integrand = [(h1.conjugate()*h2+h1*h2.conjugate())/Snf for (h1, h2, Snf) in zip(h1_f, h2_f, Sn_f)]

  integral = 2.0*np.trapz(integrand, f_list)

  return integral


def m1rm2r(mcr, eta):

  aa = mcr*eta**(-3.0/5)

  bb = 4*eta**(-1.0/5)*mcr**2

  m1r = (aa+np.sqrt(aa**2-bb))/2

  m2r = (aa-np.sqrt(aa**2-bb))/2

  return m1r, m2r


def HH_ff(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):

  f_list, signal1_f, signal2_f = hh_ff_2PN_NAPF_p(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)

  return f_list, signal1_f, signal2_f



def parameters_estimate(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  nn = 8 

  del_pars_mat = np.zeros(shape = ([nn, nn]), dtype = float)

  del_pars = np.array([del_ln_mcr, del_ln_eta, del_zz, del_ttcr, del_theta_s_bar, del_phi_s_bar, del_incl, del_ln_ee0])

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      if ii == jj:

        del_pars_mat[ii, jj] = del_pars[ii]

      elif ii != jj:

        del_pars_mat[ii, jj] = 0.0

 
  mcr   = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1.0/5)
  
  eta   = m1r*m2r/(m1r+m2r)**2

  
  mcr0  = mcr

  mtr0  = m1r+m2r

  zz0   = zz
 
  ttcr0 = ttcr

  
  freq0, signal1_0, signal2_0 = HH_ff(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, 0, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)
  
  
  signal1_list = []; signal2_list = [];

  for kk in range(len(del_pars)):

    signal1_kk, signal2_kk = HH_ff(mcr*np.exp(-del_pars_mat[kk][0]), eta*np.exp(-del_pars_mat[kk][1]), zz-del_pars_mat[kk][2], ttcr-del_pars_mat[kk][3], theta_s_bar-del_pars_mat[kk][4], phi_s_bar-del_pars_mat[kk][5], theta_l_bar, phi_l_bar, del_pars_mat[kk][6], f0_red, ee0*np.exp(-del_pars_mat[kk][7]), mcr0, mtr0, zz0, ttcr0)[1:]
    
    signal1_list.append(signal1_kk)

    signal2_list.append(signal2_kk)


  Sn_f = Sn_f_LISA_NSA(freq0)

  del_pars[2] = (lum_dis(zz)-lum_dis(zz-del_zz))/lum_dis(zz)

  del_pars[3] = (1+zz)*del_pars[3]

  Gamma = np.zeros(shape = ([len(del_pars), len(del_pars)]), dtype = float)

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      Gamma[ii, jj] = (inner_product((-signal1_list[ii]+signal1_0)/del_pars[ii], (-signal1_list[jj]+signal1_0)/del_pars[jj], Sn_f, freq0)+inner_product((-signal2_list[ii]+signal2_0)/del_pars[ii], (-signal2_list[jj]+signal2_0)/del_pars[jj], Sn_f, freq0)).real


#  return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5], Gamma[6,6], Gamma[7,7] # Gamma_McMc, Gamma_etaeta, Gamma_DLDL, Gamma_tctc, Gamma_thetas_bar_thetas_bar, Gamma_phis_bar_phis_bar, Gamma_incl_incl, Gamma_ee0_ee0

  Sigma         = np.linalg.inv(Gamma.real)

  Del_Omega_bar = 2*np.pi*abs(np.sin(theta_s_bar))*np.sqrt((Sigma[4,4]*Sigma[5,5]-Sigma[4,5]**2))*(180.0/np.pi)**2

  dL            = lum_dis(zz)

  Del_V         = 4.0/3*(dL)**3*Del_Omega_bar*3e-4*np.sqrt(Sigma[2,2])

  #return np.sqrt(Sigma[3,3]), Del_Omega_bar, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), np.sqrt(Sigma[2,2]), Del_V, np.sqrt(Sigma[6,6]), np.sqrt(Sigma[7,7])


# Delta_tc, Delta_Omega_bar, Delta_Mc_d_Mc, Delta_eta_d_eta, Delta_DL_d_DL, Del_V, Delta_incl, Delta_e0_d_e0

  return Gamma

