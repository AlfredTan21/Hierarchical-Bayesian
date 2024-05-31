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
import functions_LISA_ecc as func_LISA

### constants ###

GG         = 6.67408e-11

m_earth    = 5.965e24

m_sun      = 1.9891e30

mpc        = 1e6*3.084e16

cc         = 2.998e8

R0         = 1e8 # the radius of TianQin satellites

m_sun_in_s = m_sun*GG/cc**3 # the unit is sec

mpc_in_s   = mpc/cc # the unit is sec

f0         = 1.0/(2*np.pi)*np.sqrt(GG*m_earth/R0**3) # the frequency of TianQin satellites motion

f_star     = 0.28 # the limit of low frequency of TianQin

tt_1yr     = 365*24*3600.0 # 1 year

tto        = 5.0*tt_1yr # 5 years

tto_unit   = tt_1yr/4.0 # 1/4 year, 3 months 

R_1AU      = 1.495978707e11

gamma_E    = 0.577

tto_unit2=365*24*3600/2#6个月
tto_unit1=365*24*3600/3#4个月
tto_unit0=365*24*3600/6#2个月
##########

path_interp = "/home/ShuaiLiu/Projects/mulfreq_GW_astron/TianQin/files_exe/data_interp"
path_interp_data = "/home/wl/aa/LS"

### make directory ###

def make_directory(path):

 isExists = os.path.exists(path)

 if not isExists:

   os.makedirs(path)

   print(path+" is created successfully!") 

   return True

 else:

   print(path+" has existed already!")

   return False



### random parameters ###

# generate random merger number of GW190521  

def merger_number_GW190521(tt):
  
  num_list = []; ln_rate_list = []; 
 
  np.random.seed(5)

  ln_rate_list = skewnorm.rvs(-3.1189, loc = -1.0624, scale = 1.4539, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list

# generate random merger rates of mass model log-flat

def merger_number_flat(tt):

  num_list = []; ln_rate_list = []; 
 
  np.random.seed(0)

  ln_rate_list = skewnorm.rvs(-1.0185, loc = 3.165, scale = 0.402, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list

# generate random merger rates of mass model power law 

def merger_number_power(tt):

  num_list = []; ln_rate_list = [];
 
  np.random.seed(1)

  ln_rate_list = skewnorm.rvs(-1.03, loc = 4.27, scale = 0.4114, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list

# generate random merger rates of mass model A 

def merger_number_AA(tt):

  num_list = []; ln_rate_list = []; 
 
  xi_a = 3.8913011; omega_a = 0.5298859; alpha_a = 0.8559331;

  np.random.seed(2)

  ln_rate_list = skewnorm.rvs(alpha_a, loc = xi_a, scale = omega_a, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list 

# generate random merger rates of mass model B 

def merger_number_BB(tt):

  num_list = []; ln_rate_list = [];

  xi_b = 4.2354147; omega_b = 0.5220094; alpha_b = -0.8423922;

  np.random.seed(3)

  ln_rate_list = skewnorm.rvs(alpha_b, loc = xi_b, scale = omega_b, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate

    num  = int(np.array(cosmo.comoving_volume(2.0))*1e-9*tt*rate)

    num_list.append(num)

  return num_list 

# generate random merger rates of mass model C 

def merger_number_CC(tt):

  num_list = []; ln_rate_list = [];

  xi_c = 3.9450482; omega_c = 0.5042277; alpha_c = 0.3156015;

  np.random.seed(4)

  ln_rate_list = skewnorm.rvs(alpha_c, loc = xi_c, scale = omega_c, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate
    #num  = int(np.array(cosmo.comoving_volume(2))*1e-9*tt*rate)

    num  = int(np.array(cosmo.comoving_volume(0.5))*1e-9*tt*rate)#2改成0.5,共动体积上限可以根据红移上限求得，20211014改成了1

    num_list.append(num)

  return num_list 
def merger_number_PP(tt):
####a是偏度
####
  num_list = []; ln_rate_list = [];

  xi_PP = 3.476045914; omega_PP =0.29900061; alpha_PP=-0.751844;

  np.random.seed(4)

  ln_rate_list = skewnorm.rvs(alpha_PP, loc = xi_PP, scale = omega_PP, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate
    #num  = int(np.array(cosmo.comoving_volume(2))*1e-9*tt*rate)

    num  = int(np.array(cosmo.comoving_volume(1))*1e-9*tt*rate)#2改成0.5,共动体积上限可以根据红移上限求得，20211014改成了1

    num_list.append(num)

  return num_list
def merger_number_PPO2(tt):
####a是偏度
####
  num_list = []; ln_rate_list = [];

  xi_PP = 3.012800553; omega_PP =0.32363215318; alpha_PP=0.83328395;

  np.random.seed(4)

  ln_rate_list = skewnorm.rvs(alpha_PP, loc = xi_PP, scale = omega_PP, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate
    #num  = int(np.array(cosmo.comoving_volume(2))*1e-9*tt*rate)

    num  = int(np.array(cosmo.comoving_volume(1))*1e-9*tt*rate)#2改成0.5,共动体积上限可以根据红移上限求得，20211014改成了1

    num_list.append(num)

  return num_list
def merger_number_BGP(tt):
####a是偏度
####
  num_list = []; ln_rate_list = [];

  xi_BGP = 3.33707960; omega_BGP =0.283548344; alpha_BGP=1.07623201;

  np.random.seed(4)

  ln_rate_list = skewnorm.rvs(alpha_BGP, loc = xi_BGP, scale = omega_BGP, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate
    #num  = int(np.array(cosmo.comoving_volume(2))*1e-9*tt*rate)

    num  = int(np.array(cosmo.comoving_volume(1))*1e-9*tt*rate)#2改成0.5,共动体积上限可以根据红移上限求得，20211014改成了1

    num_list.append(num)

  return num_list 
def merger_number_FM(tt):

  num_list = []; ln_rate_list = [];

  xi_FM = 3.42898841983; omega_FM =0.29537339; alpha_FM = -1.159327644;

  np.random.seed(4)

  ln_rate_list = skewnorm.rvs(alpha_FM, loc = xi_FM, scale = omega_FM, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate
    #num  = int(np.array(cosmo.comoving_volume(2))*1e-9*tt*rate)

    num  = int(np.array(cosmo.comoving_volume(1))*1e-9*tt*rate)#2改成0.5,共动体积上限可以根据红移上限求得，20211014改成了1

    num_list.append(num)

  return num_list 
def merger_number_PS(tt):
####a是偏度
####
  num_list = []; ln_rate_list = [];

  xi_PS= 3.37535290; omega_PS=0.23713572456; alpha_PS= 0.5527624095;

  np.random.seed(4)

  ln_rate_list = skewnorm.rvs(alpha_PS, loc = xi_PS, scale = omega_PS, size = 200, random_state = None)

  for ln_rate in ln_rate_list:

    rate = np.e**ln_rate
    #num  = int(np.array(cosmo.comoving_volume(2))*1e-9*tt*rate)

    num  = int(np.array(cosmo.comoving_volume(1))*1e-9*tt*rate)#2改成0.5,共动体积上限可以根据红移上限求得，20211014改成了1

    num_list.append(num)

  return num_list 



# generate m1r and m2r of GW190521 

def random_mass_GW190521():

  return 85, 66 

# generate random m1r and m2r of mass model log-flat

def random_mass_flat():

  ran1, ran2  = np.random.rand(2)

  if ran1 < ran2:

    ran3 = ran1

    ran1 = ran2

    ran2 = ran3

  m1r    = 5.0*10**ran1

  m2r    = 5.0*10**ran2

  return m1r, m2r

# generate random m1r and m2r of mass model power law 

def random_mass_power():

  ran3, ran4  = np.random.rand(2)
  
  m1r   = 5*(1-ran3*(1-10**(-1.3)))**(-10.0/13)
  
  m2r   = 5.0+(m1r-5)*ran4

  return m1r, m2r

# generate random m1r and m2r of mass model A 

def random_mass_AA(cdf_m1rp, m1rp, cdf_qp, qp):

  ran_m1r, ran_q = np.random.rand(2)
    
  m1r = np.interp(ran_m1r, cdf_m1rp, m1rp)

  qq  = np.interp(ran_q, cdf_qp, qp)

  m2r = m1r*qq

  return m1r, m2r 

# generate random m1r and m2r of mass model B 

def random_mass_BB(cdf_m1rp, m1rp, cdf_qp, qp):

  ran_m1r, ran_q = np.random.rand(2)
    
  m1r = np.interp(ran_m1r, cdf_m1rp, m1rp)

  qq  = np.interp(ran_q, cdf_qp, qp)

  m2r = m1r*qq

  return m1r, m2r 

#def random_mass_BB(ii):
#
#  cdf_m1rp = []; m1rp = []; cdf_qp = []; qp = [];
#
#  for line0 in open(path_interp+"/mass_models_ABC/cdf_m1b.txt"):
#    
#    cdf_m1b = line0.split()
#
#    cdf_m1rp.append(float(cdf_m1b[0]))
#
#    m1rp.append(float(cdf_m1b[1]))
#
#  for line1 in open(path_interp+"/mass_models_ABC/cdf_qb.txt"):
#    
#    cdf_qb = line1.split()
#
#    cdf_qp.append(float(cdf_qb[0]))
#
#    qp.append(float(cdf_qb[1]))
#
#  np.random.seed(8+ii)
#
#  ran_m1r  = np.random.rand()
#    
#  ran_q    = np.random.rand()
#
#  m1r = np.interp(ran_m1r, cdf_m1rp, m1rp)
#
#  qq  = np.interp(ran_q, cdf_qp, qp)
#
#  m2r = m1r*qq
#
#  return m1r, m2r 

# generate random m1r and m2r of mass model C 

def random_mass_CC(cdf_m1rp, m1rp, cdf_qp, qp):

  ran_m1r, ran_q = np.random.rand(2)
    
  m1r = np.interp(ran_m1r, cdf_m1rp, m1rp)

  qq  = np.interp(ran_q, cdf_qp, qp)

  m2r = m1r*qq

  return m1r, m2r 

def random_mass_CC1(ii):
#
  cdf_m1rp = []; m1rp = []; cdf_qp = []; qp = [];
#
  for line0 in open(path_interp_data+"/cdf_m1_q_ABC_published/cdf_m1c.txt"):
#    
    cdf_m1c = line0.split()
#
    cdf_m1rp.append(float(cdf_m1c[0]))
#
    m1rp.append(float(cdf_m1c[1]))
#
  for line1 in open(path_interp_data+"/cdf_m1_q_ABC_published/cdf_qc.txt"):
#    
    cdf_qc = line1.split()
#
    cdf_qp.append(float(cdf_qc[0]))
#
    qp.append(float(cdf_qc[1]))
#
  np.random.seed(9+ii)
#
  ran_m1r  = np.random.rand(ii)
#    
  ran_q    = np.random.rand(ii)
#
  m1r = np.interp(ran_m1r, cdf_m1rp, m1rp)
#
  qq  = np.interp(ran_q, cdf_qp, qp)
#
  m2r = m1r*qq
#
  return m1r, m2r 

# generate random comoving distance

#def random_comoving_distance(num):                                                                                        
#  rcm_max  = np.array(cosmo.comoving_distance(2.0)) # the unit is Mpc
#
#  rcm_list = [];
#        
#  for count in xrange(num):
#
#    random.seed(count+3+num)
#
#    ran5   = random.random()
#                        
#    rcm    = rcm_max*ran5**(1.0/3)
#                                
#    rcm_list.append(rcm)
#
#  return rcm_list 




def random_comoving_distance():

  rcm_max = np.array(cosmo.comoving_distance(2.0)) # the unit is Mpc

  ran5    = np.random.rand()
    
  rcm     = rcm_max*ran5**(1.0/3)
    
  return rcm 


# generate random redshift by interpolation method 

#def random_redshift(num):                                                                                                 
#  zz0 = []; rrc0 = [];
#
#  for line0 in open(path_interp+"/redshift0_comoving_distance0.txt"):
#    
#    line1 = line0.split()
#
#    zz0.append(float(line1[0]))
#
#    rrc0.append(float(line1[1]))
#
#    zzp = sorted(zz0); rrcp = sorted(rrc0);
#
#    comoving_distance = random_comoving_distance(num)
#
#    redshift = np.interp(comoving_distance, rrcp, zzp) 
#
#  return redshift



def random_redshift(rrcp, zzp):

  comoving_distance = random_comoving_distance()

  redshift = np.interp(comoving_distance, rrcp, zzp)

  return redshift

# generate random redshift by Newton-Raphson method 

def random_redshift_Newtoin(num):

  rcm_max  = np.array(cosmo.comoving_distance(2.0)) # the unit is Mpc

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



### unit row vector ###

def unit_sphere(theta, phi):

  x0 = np.sin(theta)*np.cos(phi)

  y0 = np.sin(theta)*np.sin(phi)

  z0 = np.cos(theta)

  return np.array([x0, y0, z0])

### unit column vector ###

def unit_sphere_column(theta, phi): 

  x0 = np.sin(theta)*np.cos(phi)

  y0 = np.sin(theta)*np.sin(phi)

  z0 = np.cos(theta)

  return np.array([[x0], [y0], [z0]])


def calc_psi_s(theta_s, phi_s0, theta_l, phi_l0):

  z0          = np.array([0.0, 0.0, 1.0])

  nn          = unit_sphere(theta_s, phi_s0)

  LL          = unit_sphere(theta_l, phi_l0)

  numerator   = np.dot(LL, z0)-np.dot(LL, nn)*np.dot(z0, nn)
  
  denominator = np.dot(nn, np.cross(LL, z0))

  if numerator < 0.0:

    psi_s = np.arctan2(numerator, denominator)+2.0*np.pi

  else:

    psi_s = np.arctan2(numerator, denominator)

  return psi_s


def calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar):
  
  theta_s, phi_s0 = angles_Ec_to_Dec_fg(theta_s_bar, phi_s_bar)

  theta_l, phi_l0 = angles_Ec_to_Dec_fg(theta_l_bar, phi_l_bar)

  nn   = unit_sphere(theta_s, phi_s0)

  LL   = unit_sphere(theta_l, phi_l0)

  incl = np.arccos(np.dot(LL, nn))

  return incl 



def random_angles_bar():

  ran6, ran7, ran8, ran9 = np.random.rand(4) 
  
  theta_s_bar = np.arccos(2.0*ran6-1) 

  phi_s_bar   = 2.0*np.pi*ran7  

  theta_l_bar = np.arccos(2.0*ran8-1) 

  phi_l_bar   = 2.0*np.pi*ran9

  return [theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar]

### rotation of coordinates system, angles which coordinate system rotates anticlockwise are positive ###

# rotate matrix of coordinates

def RotX(alphax, vector):

  cax, sax = np.cos(alphax), np.sin(alphax)

  RX       = np.array([[1, 0, 0], [0, cax, sax], [0, -sax, cax]])
  
  return np.dot(RX, vector)


def RotY(alphay, vector):

  cay, say = np.cos(alphay), np.sin(alphay)

  RY       = np.array([[cay, 0, -say], [0, 1, 0], [say, 0, cay]])

  return np.dot(RY, vector)


def RotZ(alphaz, vector):

  caz, saz = np.cos(alphaz), np.sin(alphaz)

  RZ       = np.array([[caz, saz, 0],[-saz, caz, 0],[0, 0, 1]])

  return np.dot(RZ, vector)



### from ecliptic coordinates to coordinates of first group 


def angles_Ec_to_Dec_fg(theta_bar, phi_bar):

  nn_bar     = unit_sphere_column(theta_bar, phi_bar)

  alpha_xbar = np.radians(-4.7052)-np.pi/2

  alpha_zbar = np.radians(120.4442)-np.pi/2

  nn_fg      = RotX(alpha_xbar, RotZ(alpha_zbar, nn_bar)) # Firstly, rotate the vector nn_fg_bar alpha_zbar clockwise along the zbar axis, secondly, rotate the vector alpha_xbar counterclockwise along the xbar axis

  xx_fg      = nn_fg[0][0]

  yy_fg      = nn_fg[1][0]

  zz_fg      = nn_fg[2][0]

  theta_fg    = np.arccos(zz_fg)

  if yy_fg < 0.0:

    phi0_fg = np.arctan2(yy_fg, xx_fg)+2.0*np.pi

  else:

    phi0_fg = np.arctan2(yy_fg, xx_fg)

  return theta_fg, phi0_fg


### from ecliptic coordinates to coordinates of second group

def angles_Ec_to_Dec_sg(theta_bar, phi_bar):

  nn_bar     = unit_sphere_column(theta_bar, phi_bar)

  alpha_ybar = -np.pi/2

  alpha_xbar = np.radians(-4.7052)-np.pi/2

  alpha_zbar = np.radians(120.4442)-np.pi/2

  nn_sg      = RotY(alpha_ybar, RotX(alpha_xbar, RotZ(alpha_zbar, nn_bar)))

  xx_sg      = nn_sg[0][0]

  yy_sg      = nn_sg[1][0]

  zz_sg      = nn_sg[2][0]

  theta_sg = np.arccos(zz_sg)

  if yy_sg < 0.0:

    phi0_sg = np.arctan2(yy_sg, xx_sg)+2.0*np.pi

  else:

    phi0_sg = np.arctan2(yy_sg, xx_sg)

  return theta_sg, phi0_sg



def random_ttrc(tt):

  ran12 = np.random.rand()

  ttrc  = tt*ran12*tt_1yr
    
  return ttrc


def random_ttrc_critical(tt):

  ran13 = np.random.rand()

  ttrc  = (tt+ran13*10)*tt_1yr
    
  return ttrc


  
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


def angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar):

  theta_s, phi_s0 = angles_Ec_to_Dec_fg(theta_s_bar, phi_s_bar)

  theta_l, phi_l0 = angles_Ec_to_Dec_fg(theta_l_bar, phi_l_bar)

  psi_s           = calc_psi_s(theta_s, phi_s0, theta_l, phi_l0)

  return theta_s, phi_s0, psi_s



def angles_in_res_func_sg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar):

  theta_s, phi_s0 = angles_Ec_to_Dec_sg(theta_s_bar, phi_s_bar)

  theta_l, phi_l0 = angles_Ec_to_Dec_sg(theta_l_bar, phi_l_bar)

  psi_s           = calc_psi_s(theta_s, phi_s0, theta_l, phi_l0)

  return theta_s, phi_s0, psi_s

### The frequency for one group of TianQin satellites ###

def freq_segmentate(mcr, mtr, zz, ttcr):

  # 计算全天平均的情况时频率划分为10，计算非全天平均的情况时划分为40
    
  coeff   = 5/(96*(np.pi)**(8.0/3))
   
  fr_isco = 1.0/(np.pi*6**(3.0/2)*mtr*m_sun_in_s)
   
  fr_list = []; f_red_list = []
   
  if ttcr <= tto_unit/(1+zz):
   
    fr_ii = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr))**(-3.0/8)
    
    fr_list.append(fr_ii)
    
    fr_list.append(fr_isco)
    
    
  elif tto_unit/(1+zz) < ttcr <= tto/(1+zz):
    
    factor = ttcr/(tto_unit/(1+zz))
     
    if ttcr % (tto_unit/(1+zz)) == 0:
     
      for ii in range(int(factor)):
      
        del_tr = ttcr-ii*tto_unit/(1+zz)
        
        fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
        
        fr_list.append(fr_ii)
      
      fr_list.append(fr_isco)

    else:
        
      for ii in range(int(factor)+1):
      
        del_tr = ttcr-ii*tto_unit/(1+zz)
       
        fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
       
        fr_list.append(fr_ii)
       
      fr_list.append(fr_isco)
       
      
  elif ttcr > tto/(1+zz):
   
    factor = int(tto/tto_unit)
    
    for ii in range(factor+1):

      del_tr = ttcr-ii*tto_unit/(1+zz)
  
      fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
  
      fr_list.append(fr_ii)

  
  f_red_list     = np.array(fr_list)/(1+zz)


  factor1        = int(len(f_red_list)/2)
# f_red_list中频率的个数除以2然后取整，例如int(3/2)=int(1.5)=1

  f_red_list[0] = f_red_list[0]-2*f0

  f_red_list[factor1*2-1] = f_red_list[factor1*2-1]+2*f0 # 加减2f0的原因是f的取值范围是[f_red_i-2f0, f_red_f+2f0]


  freq_red_list  = []

  for ii in range(2*factor1-1):

#    f_red_list1 = np.linspace(f_red_list[0+ii], f_red_list[1+ii], 1e5, endpoint = True)

    index_i     = math.log(f_red_list[0+ii], 10)
  
    index_f     = math.log(f_red_list[1+ii], 10)
  
    index_range = np.linspace(index_i, index_f, 40, endpoint = True)
  
    f_red_list1 = np.power(10, index_range) 
  
    freq_red_list.append(f_red_list1)


  return freq_red_list



### The frequency for two group of TianQin satellates ###

def freq_segmentate_two_group(mcr, mtr, zz, ttcr):
  
  # 计算全天平均的情况时频率划分为10，计算非全天平均的情况时划分为40
  
  coeff   = 5/(96*(np.pi)**(8.0/3))
   
  fr_isco = 1.0/(np.pi*6**(3.0/2)*mtr*m_sun_in_s)
   
  fr_list = []; f_red_list = []
   
  if ttcr <= tto_unit/(1+zz):
   
    fr_ii = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr))**(-3.0/8)
    
    fr_list.append(fr_ii)
    
    fr_list.append(fr_isco)
    
    
  elif tto_unit/(1+zz) < ttcr <= tto/(1+zz):
    
    factor = ttcr/(tto_unit/(1+zz))
     
    if ttcr % (tto_unit/(1+zz)) == 0:
     
      for ii in range(int(factor)):
      
        del_tr = ttcr-ii*tto_unit/(1+zz)
        
        fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
        
        fr_list.append(fr_ii)
      
      fr_list.append(fr_isco)

    else:
        
      for ii in range(int(factor)+1):
      
        del_tr = ttcr-ii*tto_unit/(1+zz)
       
        fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
       
        fr_list.append(fr_ii)
       
      fr_list.append(fr_isco)
       
      
  elif ttcr > tto/(1+zz):
   
    factor = int(tto/tto_unit)
    
    for ii in range(factor+1):

      del_tr = ttcr-ii*tto_unit/(1+zz)
  
      fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
  
      fr_list.append(fr_ii)

## first group

  f_red      = np.array(fr_list)/(1+zz) 

  factor1    = int(len(f_red)/2)

  f_red[0]   = f_red[0]-2*f0
  
  f_red[factor1*2-1]    = f_red[factor1*2-1]+2*f0
  
  freq_red_first_group  = []
 
  for ii in range(2*factor1-1):

#  f_red_ii = np.linspace(f_red[0+ii], f_red[1+ii], 1e3, endpoint = True)

    index_i     = math.log(f_red[0+ii], 10)
  
    index_f     = math.log(f_red[1+ii], 10)
  
    index_range = np.linspace(index_i, index_f, 40, endpoint = True)#原40个
  
    f_red_ii    = np.power(10, index_range) 
  
    freq_red_first_group.append(f_red_ii)


## second group

  f_red2     = np.array(fr_list)/(1+zz) 

  factor2    = int((len(f_red2)-1)/2)
 
  freq_red_second_group  = []
 
  if factor2 == 0:

    freq_second_group = []

  else:

    f_red2[1] = f_red2[1]-2*f0
    
    f_red2[2*factor2] = f_red2[2*factor2]+2*f0
    
    for ii in range(2*factor2-1):
  
#    f_red_ii = np.linspace(f_red[0+ii], f_red[1+ii], 100, endpoint = True)
  
      index_i     = math.log(f_red2[1+ii], 10)
    
      index_f     = math.log(f_red2[2+ii], 10)
    
      index_range = np.linspace(index_i, index_f, 40, endpoint = True)
    
      f_red_ii    = np.power(10, index_range) 
    
      freq_red_second_group.append(f_red_ii)

  return freq_red_first_group, freq_red_second_group



### The starting and ending frequecy of 0PN ###

def freq_red_i_f_0PN(mtr, mcr, ttcr, zz):

  coeff      = 5/(96*(np.pi)**(8.0/3))

  freqr_i    = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr))**(-3.0/8)

  if ttcr   <= tto/(1+zz):

    freqr_f  = 1.0/(np.pi*6**(3.0/2))/(mtr*m_sun_in_s)
 
  else:
  
    freqr_f  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr-tto/(1+zz)))**(-3.0/8)

  freq_red_i = freqr_i/(1+zz)

  freq_red_f = freqr_f/(1+zz)

  return freq_red_i, freq_red_f



### polarization phase ###

#def phip(f_red, tt_red, incl, theta_s, phi_s0, psi_s):
#
#  phi_s       = phi_s0+2*np.pi*f_red*tt_red
#
#  F_plus      = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s)*np.cos(2*psi_s)-np.cos(theta_s)*np.sin(2*phi_s)*np.sin(2*psi_s)
#
#  F_cross     = 1.0/2*(1+np.cos(theta_s)**2)*np.cos(2*phi_s)*np.sin(2*psi_s)+np.cos(theta_s)*np.sin(2*phi_s)*np.cos(2*psi_s)
#
#  numerator   = 2*np.cos(incl)*F_cross
#
#  denominator = (1+np.cos(incl)**2)*F_plus
#
#  phi_p       = np.arctan2(numerator, denominator)
#
#  return phi_p


### Eccentric phase ###

def phiE(f_red, mc_red, f0_red, ee0):

  phi_E = -4239.0/11696*(mc_red*m_sun_in_s*np.pi)**(-5.0/3)*(f0_red**(19.0/9)/f_red**(34.0/9))*ee0**2 

  return phi_E


### Doppler phase ###

def phiD(f_red, tt_red, theta_s_bar, phi_s_bar):

  phi_D = 2*np.pi*f_red*R_1AU/cc*np.sin(theta_s_bar)*np.cos(2*np.pi*tt_red/tt_1yr-phi_s_bar)

  return phi_D


### The phase of 0PN ###

def psi_f_0PN(f_red, mc_red, ttc_red, phic, alpha0, eta):

  xx    = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)

  psi_f = 2*np.pi*f_red*ttc_red-phic-np.pi/4+3.0/128*(mc_red*m_sun_in_s*np.pi*f_red)**(-5.0/3)*(alpha0)

  return psi_f 

### The phase of 2PN ###

#def psi_f_2PN(f_red, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, eta):
#
#  xx    = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)
#
#  psi_f = 2*np.pi*f_red*ttc_red-phic-np.pi/4+3.0/128*(mc_red*m_sun_in_s*np.pi*f_red)**(-5.0/3)*(alpha0+alpha2*xx+alpha3*xx**(3.0/2)+alpha4*xx**2+alpha5*xx**(5.0/2)+alpha6*xx**3+alpha7*xx**(7.0/2))
#
#
#  return psi_f 

def psi_f_2PN(f_red, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, eta):

  xx    = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)

  psi_f = 2*np.pi*f_red*ttc_red-phic-np.pi/4+3.0/128*(mc_red*m_sun_in_s*np.pi*f_red)**(-5.0/3)*(alpha0+alpha2*xx+alpha3*xx**(3.0/2)+alpha4*xx**2+alpha5*xx**(5.0/2)+alpha6*xx**3)


  return psi_f 


### The 2PN waveform which does not contain polarization mode ###

def hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_list, tt_red, theta_s_bar, phi_s_bar, f0_red, ee0):
 
  mt_red     = mc_red/eta**(3.0/5)

  f_red_isco = 1.0/(np.pi*6**(3.0/2)*mt_red*m_sun_in_s)
  
  xx         = (np.pi*mc_red*m_sun_in_s*f_red_list)**(2.0/3)*eta**(-2.0/5)
 
  xx0        = (np.pi*mc_red*m_sun_in_s*f_red_isco)**(2.0/3)*eta**(-2.0/5)
 
  alpha0     = 1
  
  alpha2     = 3715.0/756+55.0/9*eta
  
  alpha3     = -16*np.pi
  
  alpha4     = 15293365.0/508032+27145.0/504*eta+3085.0/72*eta**2

  alpha5     = (38645.0/756-65.0/9*eta)*(1+3.0/2*np.log(xx/xx0))*np.pi

  alpha6     = (11583231236531.0/4694215680-640.0/3*np.pi**2-6848.0/21*gamma_E-3424.0/21*np.log(16*xx)+(-15737765635.0/3048192+2255.0/12*np.pi**2)*eta+76055.0/1728*eta**2-127825.0/1296*eta**3)

#  alpha7     = (77096675.0/254016+378515.0/1512*eta-74045.0/756*eta**2)*np.pi
  
  ampli_f    = ampli_f_0PN(dL, mc_red, f_red_list)

#  psi_f      = psi_f_2PN(f_red_list, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, eta)
   
  psi_f      = psi_f_2PN(f_red_list, mc_red, ttc_red, phic, alpha0, alpha2, alpha3, alpha4, alpha5, alpha6, eta)
 
  phi_D      = phiD(f_red_list, tt_red, theta_s_bar, phi_s_bar)

  phi_E      = phiE(f_red_list, mc_red, f0_red, ee0)

  hh0_ff     = ampli_f*np.exp(1j*psi_f-1j*phi_D+1j*phi_E)

  return hh0_ff 



### The time of 2PN, 旋近阶段经历的时间###

def tt_f_2PN(f_red, mc_red, ttc_red, eta, f0_red, ee0):
  
  xx      = (np.pi*mc_red*m_sun_in_s*f_red)**(2.0/3)*eta**(-2.0/5)

  tau0    = 1

  tau2    = 4.0/3*(743.0/336+11.0/4*eta)

  tau3    = -8.0/5*(4*np.pi)

  tau4    = 3058673.0/508032+5429.0/504*eta+617.0/72*eta**2

  tau5    = -(7729.0/252-13.0/3*eta)*np.pi

  tau6    = (-10052469856691.0/23471078400+128.0/3*np.pi**2+6848.0/105*gamma_E+3424.0/105*np.log(16*xx)+(3147553127.0/3048192-451.0/12*np.pi**2)*eta-15211.0/1728*eta**2+25565.0/1296*eta**3)

  tau_e   = 785.0/110008*(mc_red*m_sun_in_s)**(-5.0/3)*np.pi**(-8.0/3)*f0_red**(19.0/9)/f_red**(34.0/9)*ee0**2

#  tau7    = (-15419335.0/127008-75703.0/756*eta+14809.0378*eta**2)*np.pi
  
#  tt_f    = ttc_red-5.0/256*(mc_red*m_sun_in_s)**(-5.0/3)*(np.pi*f_red)**(-8.0/3)*(tau0+tau2*xx+tau3*xx**(3.0/2)+tau4*xx**2+tau5*xx**(5.0/2)+tau6*xx**3+tau7*xx**(7.0/2))

  tt_f    = ttc_red-5.0/256*(mc_red*m_sun_in_s)**(-5.0/3)*(np.pi*f_red)**(-8.0/3)*(tau0+tau2*xx+tau3*xx**(3.0/2)+tau4*xx**2+tau5*xx**(5.0/2)+tau6*xx**3)+tau_e

  return tt_f 


### The time of 0PN ###

def tt_f_0PN(f_red, mc_red, ttc_red):

  tt_f = ttc_red-5.0/256*(mc_red*m_sun_in_s)**(-5.0/3)*(np.pi*f_red)**(-8.0/3) 

  return tt_f


### The amplitude of waveform of 0PN ###

def ampli_f_0PN(dL, mc_red, f_red):   

  return np.sqrt(5.0/24)*(np.pi)**(-2.0/3)*1/(dL*mpc_in_s)*(mc_red*m_sun_in_s)**(5.0/6)*(f_red)**(-7.0/6)


def hp_hc_fm2f0(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, aa, bb, theta_s_bar, phi_s_bar, f0_red, ee0):


  remainder         = np.array(np.floor(tt_0PN_fm2f0/tto_unit)%2, dtype = complex)

  remainder[index]  = [bb]*len(index)

  index0            = np.where(remainder == aa)[0]

  hh0_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_m2f0[index0], tt_2PN_fm2f0[index0], theta_s_bar, phi_s_bar, f0_red, ee0)
  
  remainder[index0] = hh0_ff 

  index1            = np.where(remainder == bb)[0]

  remainder[index1] = [0.0]*len(index1)

  hp_minus2f0_list  = chip*remainder

  hc_minus2f0_list  = chic*remainder

  return np.array(hp_minus2f0_list), np.array(hc_minus2f0_list)


def hp_hc_fp2f0(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, aa, bb, theta_s_bar, phi_s_bar, f0_red, ee0):
 

  remainder         = np.array(np.floor(tt_0PN_fp2f0/tto_unit)%2, dtype = complex)

  remainder[index]  = [bb]*len(index)

  index0            = np.where(remainder == aa)[0]

  hh0_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_p2f0[index0], tt_2PN_fp2f0[index0], theta_s_bar, phi_s_bar, f0_red, ee0)

  remainder[index0] = hh0_ff 

  index1            = np.where(remainder == bb)[0]

  remainder[index1] = [0.0]*len(index1)

  hp_plus2f0_list   = chip*remainder

  hc_plus2f0_list   = chic*remainder

  return np.array(hp_plus2f0_list), np.array(hc_plus2f0_list)




### The waveform of 2PN for first group of TianQin satellite ### 

def hh_ff_2PN_seg(m1r, m2r, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0):
    
  mtr            = m1r+m2r
  
  mt_red         = mtr*(1+zz)
  
  mcr            = (m1r*m2r)**0.6/(m1r+m2r)**0.2
  
  mc_red         = mcr*(1+zz)
  
  ttc_red        = ttcr*(1+zz)
  
  phic           = -np.pi/4
 
  dL             = lum_dis(zz)

  eta            = m1r*m2r/(m1r+m2r)**2
 
  vv             = np.cos(incl)
  
  chip           = (1+vv**2)/2
  
  chic           = -1j*vv

  freq_red_list0         = freq_segmentate(mcr, mtr, zz, ttcr)


  freq_red_i, freq_red_f = freq_red_list0[0][0]+2*f0, freq_red_list0[-1][-1]-2*f0 # 加减2f0的目的是算出f_red_i 和 f_red_f 方便为下面判断波形是否为0

  freq_red_list1 = [];

  for kk in range(len(freq_red_list0)):

    fk_list        = freq_red_list0[kk]

    freq_red_list1 = list(freq_red_list1)+list(fk_list)
    

  f_red_m2f0      = np.array(freq_red_list1)-2*f0*1

  f_red_p2f0      = np.array(freq_red_list1)+2*f0*1 


  hp_minus2f0_list = []; hp_plus2f0_list = [];
  
  hc_minus2f0_list = []; hc_plus2f0_list = [];

# hplus(f-2f0) and hcross(f-2f0)
  
  index                              = list(np.where(f_red_m2f0 < freq_red_i)[0])+list(np.where(f_red_m2f0 > freq_red_f)[0]) 
  
  tt_0PN_fm2f0                       = tt_f_0PN(f_red_m2f0, mc_red, ttc_red)
  
  tt_2PN_fm2f0                       = tt_f_2PN(f_red_m2f0, mc_red, ttc_red, eta, f0_red, ee0)

  hp_minus2f0_list, hc_minus2f0_list = hp_hc_fm2f0(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)

# hplus(f+2f0) and hcross(f+2f0)

  index                            = list(np.where(f_red_p2f0 < freq_red_i)[0])+list(np.where(f_red_p2f0 > freq_red_f)[0]) 
   
  tt_0PN_fp2f0                     = tt_f_0PN(f_red_p2f0, mc_red, ttc_red)
   
  tt_2PN_fp2f0                     = tt_f_2PN(f_red_p2f0, mc_red, ttc_red, eta, f0_red, ee0)
 
  hp_plus2f0_list, hc_plus2f0_list = hp_hc_fp2f0(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)
 
  return np.array(freq_red_list1), np.array(hp_plus2f0_list), np.array(hp_minus2f0_list), np.array(hc_plus2f0_list), np.array(hc_minus2f0_list)  



def hh_ff_2PN_seg_p(mcr, eta, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
   
  mc_red0        = mcr0*(1+zz0)

  ttc_red0       = ttcr0*(1+zz0)
   
  mc_red         = mcr*(1+zz0)

  ttc_red        = ttcr*(1+zz0)
  
  phic           = -np.pi/4
 
  dL             = lum_dis(zz)

  vv             = np.cos(incl)
  
  chip           = (1+vv**2)/2
  
  chic           = -1j*vv

  freq_red_list0         = freq_segmentate(mcr0, mtr0, zz0, ttcr0)

  freq_red_i, freq_red_f = freq_red_list0[0][0]+2*f0, freq_red_list0[-1][-1]-2*f0

  
  freq_red_list1 = [];

  for kk in range(len(freq_red_list0)):

    fk_list        = freq_red_list0[kk]

    freq_red_list1 = list(freq_red_list1)+list(fk_list)
  
  
  f_red_m2f0      = np.array(freq_red_list1)-2*f0*1

  f_red_p2f0      = np.array(freq_red_list1)+2*f0*1 


  hp_minus2f0_list = []; hp_plus2f0_list = [];
  
  hc_minus2f0_list = []; hc_plus2f0_list = [];

# hplus(f-2f0) and hcross(f-2f0)

  index                              = list(np.where(f_red_m2f0 < freq_red_i)[0])+list(np.where(f_red_m2f0 > freq_red_f)[0]) 
   
  tt_0PN_fm2f0                       = tt_f_0PN(f_red_m2f0, mc_red0, ttc_red0)
   
  tt_2PN_fm2f0                       = tt_f_2PN(f_red_m2f0, mc_red, ttc_red, eta, f0_red, ee0)

  hp_minus2f0_list, hc_minus2f0_list = hp_hc_fm2f0(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)

# hplus(f+2f0) and hcross(f+2f0)


  index                            = list(np.where(f_red_p2f0 < freq_red_i)[0])+list(np.where(f_red_p2f0 > freq_red_f)[0]) 
   
  tt_0PN_fp2f0                     = tt_f_0PN(f_red_p2f0, mc_red0, ttc_red0)
   
  tt_2PN_fp2f0                     = tt_f_2PN(f_red_p2f0, mc_red, ttc_red, eta, f0_red, ee0)

  hp_plus2f0_list, hc_plus2f0_list = hp_hc_fp2f0(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)

  return np.array(freq_red_list1), np.array(hp_plus2f0_list), np.array(hp_minus2f0_list), np.array(hc_plus2f0_list), np.array(hc_minus2f0_list)  


### The waveform of 2PN for two group of TianQin satellite ### 


def hh_ff_2PN_seg_two_group(m1r, m2r, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0):
    
  mtr            = m1r+m2r
  
  mt_red         = mtr*(1+zz)
  
  mcr            = (m1r*m2r)**0.6/(m1r+m2r)**0.2
  
  mc_red         = mcr*(1+zz)
  
  ttc_red        = ttcr*(1+zz)
  
  phic           = -np.pi/4
 
  dL             = lum_dis(zz)

  eta            = m1r*m2r/mtr**2
 
  vv             = np.cos(incl)  # incl is invariant in first group and second group
  
  chip           = (1+vv**2)/2
  
  chic           = -1j*vv

### The abbreviations for first group and second group are fg and sg respectively

  f_red_fg, f_red_sg    = freq_segmentate_two_group(mcr, mtr, zz, ttcr)
    
  #f_red_fg              = freq_segmentate(mcr, mtr, zz, ttcr)

## The first group (fg for short)     

  f_red_i_fg, f_red_f_fg = f_red_fg[0][0]+2*f0, f_red_fg[-1][-1]-2*f0

  f_red_list1_fg = [];

  for kk in range(len(f_red_fg)):

    fk_list        = f_red_fg[kk]

    f_red_list1_fg = list(f_red_list1_fg)+list(fk_list)



  
  f_red_m2f0_fg  = np.array(f_red_list1_fg)-2*f0 

  f_red_p2f0_fg  = np.array(f_red_list1_fg)+2*f0 


  
  hp_m2f0_fg = []; hp_p2f0_fg = [];
  
  hc_m2f0_fg = []; hc_p2f0_fg = [];

# hplus(f-2f0) and hcross(f-2f0) of first group

  index_fg               = list(np.where(f_red_m2f0_fg < f_red_i_fg)[0])+list(np.where(f_red_m2f0_fg > f_red_f_fg)[0]) 
   
  tt_0PN_fm2f0_fg        = tt_f_0PN(f_red_m2f0_fg, mc_red, ttc_red)
  
  tt_2PN_fm2f0_fg        = tt_f_2PN(f_red_m2f0_fg, mc_red, ttc_red, eta, f0_red, ee0)

  hp_m2f0_fg, hc_m2f0_fg = hp_hc_fm2f0(index_fg, tt_0PN_fm2f0_fg, tt_2PN_fm2f0_fg, f_red_m2f0_fg, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)


# hplus(f+2f0) and hcross(f+2f0) of first group

  index_fg               = list(np.where(f_red_p2f0_fg < f_red_i_fg)[0])+list(np.where(f_red_p2f0_fg > f_red_f_fg)[0]) 
  
  tt_0PN_fp2f0_fg        = tt_f_0PN(f_red_p2f0_fg, mc_red, ttc_red)
   
  tt_2PN_fp2f0_fg        = tt_f_2PN(f_red_p2f0_fg, mc_red, ttc_red, eta, f0_red, ee0)
 
  hp_p2f0_fg, hc_p2f0_fg = hp_hc_fp2f0(index_fg, tt_0PN_fp2f0_fg, tt_2PN_fp2f0_fg, f_red_p2f0_fg, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)
 

## The second group (sg for short)

  if f_red_sg == []:

    f_red_list1_sg = []; tt_red_2PN_sg = []; 

    hp_m2f0_sg = []; hp_p2f0_sg = [];
  
    hc_m2f0_sg = []; hc_p2f0_sg = [];

  else:

    f_red_i_sg, f_red_f_sg = f_red_sg[0][0]+2*f0, f_red_sg[-1][-1]-2*f0

    f_red_list1_sg = [];

    for kk in range(len(f_red_sg)):

      fk_list        = f_red_sg[kk]

      f_red_list1_sg = list(f_red_list1_sg)+list(fk_list)

   

    f_red_m2f0_sg  = np.array(f_red_list1_sg)-2*f0 

    f_red_p2f0_sg  = np.array(f_red_list1_sg)+2*f0 

    hp_m2f0_sg = []; hp_p2f0_sg = [];
  
    hc_m2f0_sg = []; hc_p2f0_sg = [];

#   hplus(f-2f0) and hcross(f-2f0) of second group

    index_sg               = list(np.where(f_red_m2f0_sg < f_red_i_sg)[0])+list(np.where(f_red_m2f0_sg > f_red_f_sg)[0]) 
   
    tt_0PN_fm2f0_sg        = tt_f_0PN(f_red_m2f0_sg, mc_red, ttc_red)
     
    tt_2PN_fm2f0_sg        = tt_f_2PN(f_red_m2f0_sg, mc_red, ttc_red, eta, f0_red, ee0)
 
    hp_m2f0_sg, hc_m2f0_sg = hp_hc_fm2f0(index_sg, tt_0PN_fm2f0_sg, tt_2PN_fm2f0_sg, f_red_m2f0_sg, mc_red, eta, dL, ttc_red, phic, chip, chic, 1, 0, theta_s_bar, phi_s_bar, f0_red, ee0)
  

#   hplus(f+2f0) and hcross(f+2f0) of second group

    index_sg               = list(np.where(f_red_p2f0_sg < f_red_i_sg)[0])+list(np.where(f_red_p2f0_sg > f_red_f_sg)[0]) 
    
    tt_0PN_fp2f0_sg        = tt_f_0PN(f_red_p2f0_sg, mc_red, ttc_red)
     
    tt_2PN_fp2f0_sg        = tt_f_2PN(f_red_p2f0_sg, mc_red, ttc_red, eta, f0_red, ee0)
  
    hp_p2f0_sg, hc_p2f0_sg = hp_hc_fp2f0(index_sg, tt_0PN_fp2f0_sg, tt_2PN_fp2f0_sg, f_red_p2f0_sg, mc_red, eta, dL, ttc_red, phic, chip, chic, 1, 0, theta_s_bar, phi_s_bar, f0_red, ee0)
   

  return np.array(f_red_list1_fg), np.array(hp_p2f0_fg), np.array(hp_m2f0_fg), np.array(hc_p2f0_fg), np.array(hc_m2f0_fg), np.array(f_red_list1_sg), np.array(hp_p2f0_sg), np.array(hp_m2f0_sg), np.array(hc_p2f0_sg), np.array(hc_m2f0_sg)  


def hh_ff_2PN_seg_two_group_p(mcr, eta, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
   
  mc_red0        = mcr0*(1+zz0)

  ttc_red0       = ttcr0*(1+zz0)
   
  mc_red         = mcr*(1+zz0)

  ttc_red        = ttcr*(1+zz0)
  
  phic           = -np.pi/4
 
  dL             = lum_dis(zz)

  vv             = np.cos(incl)  # incl is constant in first group and second group
  
  chip           = (1+vv**2)/2
  
  chic           = -1j*vv

### The abbreviations for first group and second group are fg and sg respectively

  f_red_fg, f_red_sg = freq_segmentate_two_group(mcr0, mtr0, zz0, ttcr0)

## The first group (fg for short)

  f_red_i_fg, f_red_f_fg = f_red_fg[0][0]+2*f0, f_red_fg[-1][-1]-2*f0

  f_red_list1_fg = [];

  for kk in range(len(f_red_fg)):

    fk_list        = f_red_fg[kk]

    f_red_list1_fg = list(f_red_list1_fg)+list(fk_list)
  
  
  
  
  f_red_m2f0_fg  = np.array(f_red_list1_fg)-2*f0 

  f_red_p2f0_fg  = np.array(f_red_list1_fg)+2*f0 


  
  hp_m2f0_fg = []; hp_p2f0_fg = [];
  
  hc_m2f0_fg = []; hc_p2f0_fg = [];

# hplus(f-2f0) and hcross(f-2f0) of first group

  index_fg               = list(np.where(f_red_m2f0_fg < f_red_i_fg)[0])+list(np.where(f_red_m2f0_fg > f_red_f_fg)[0]) 
  
  tt_0PN_fm2f0_fg        = tt_f_0PN(f_red_m2f0_fg, mc_red0, ttc_red0)
   
  tt_2PN_fm2f0_fg        = tt_f_2PN(f_red_m2f0_fg, mc_red, ttc_red, eta, f0_red, ee0)
 
  hp_m2f0_fg, hc_m2f0_fg = hp_hc_fm2f0(index_fg, tt_0PN_fm2f0_fg, tt_2PN_fm2f0_fg, f_red_m2f0_fg, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)
 
# hplus(f+2f0) and hcross(f+2f0) of first group

  index_fg               = list(np.where(f_red_p2f0_fg < f_red_i_fg)[0])+list(np.where(f_red_p2f0_fg > f_red_f_fg)[0]) 
    
  tt_0PN_fp2f0_fg        = tt_f_0PN(f_red_p2f0_fg, mc_red0, ttc_red0)
  
  tt_2PN_fp2f0_fg        = tt_f_2PN(f_red_p2f0_fg, mc_red, ttc_red, eta, f0_red, ee0)

  hp_p2f0_fg, hc_p2f0_fg = hp_hc_fp2f0(index_fg, tt_0PN_fp2f0_fg, tt_2PN_fp2f0_fg, f_red_p2f0_fg, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, theta_s_bar, phi_s_bar, f0_red, ee0)
 

## The second group (sg for short)

  if f_red_sg == []:

    f_red_list1_sg = []; tt_red_2PN_sg = []; 

    hp_m2f0_sg = []; hp_p2f0_sg = [];
  
    hc_m2f0_sg = []; hc_p2f0_sg = [];

  else:

    f_red_i_sg, f_red_f_sg = f_red_sg[0][0]+2*f0, f_red_sg[-1][-1]-2*f0

    f_red_list1_sg = [];

    for kk in range(len(f_red_sg)):

      fk_list        = f_red_sg[kk]

      f_red_list1_sg = list(f_red_list1_sg)+list(fk_list)
    
    
    
    
    f_red_m2f0_sg  = np.array(f_red_list1_sg)-2*f0 

    f_red_p2f0_sg  = np.array(f_red_list1_sg)+2*f0 

    hp_m2f0_sg = []; hp_p2f0_sg = [];
  
    hc_m2f0_sg = []; hc_p2f0_sg = [];

#   hplus(f-2f0) and hcross(f-2f0) of second group

    index_sg               = list(np.where(f_red_m2f0_sg < f_red_i_sg)[0])+list(np.where(f_red_m2f0_sg > f_red_f_sg)[0]) 
   
    tt_0PN_fm2f0_sg        = tt_f_0PN(f_red_m2f0_sg, mc_red0, ttc_red0)
     
    tt_2PN_fm2f0_sg        = tt_f_2PN(f_red_m2f0_sg, mc_red, ttc_red, eta, f0_red, ee0)
   
    hp_m2f0_sg, hc_m2f0_sg = hp_hc_fm2f0(index_sg, tt_0PN_fm2f0_sg, tt_2PN_fm2f0_sg, f_red_m2f0_sg, mc_red, eta, dL, ttc_red, phic, chip, chic, 1, 0, theta_s_bar, phi_s_bar, f0_red, ee0)
   
#   hplus(f+2f0) and hcross(f+2f0) of second group

    index_sg               = list(np.where(f_red_p2f0_sg < f_red_i_sg)[0])+list(np.where(f_red_p2f0_sg > f_red_f_sg)[0]) 
    
    tt_0PN_fp2f0_sg        = tt_f_0PN(f_red_p2f0_sg, mc_red0, ttc_red0)
     
    tt_2PN_fp2f0_sg        = tt_f_2PN(f_red_p2f0_sg, mc_red, ttc_red, eta, f0_red, ee0)
  
    hp_p2f0_sg, hc_p2f0_sg = hp_hc_fp2f0(index_sg, tt_0PN_fp2f0_sg, tt_2PN_fp2f0_sg, f_red_p2f0_sg, mc_red, eta, dL, ttc_red, phic, chip, chic, 1, 0, theta_s_bar, phi_s_bar, f0_red, ee0)


  return np.array(f_red_list1_fg), np.array(hp_p2f0_fg), np.array(hp_m2f0_fg), np.array(hc_p2f0_fg), np.array(hc_m2f0_fg), np.array(f_red_list1_sg), np.array(hp_p2f0_sg), np.array(hp_m2f0_sg), np.array(hc_p2f0_sg), np.array(hc_m2f0_sg)  



### The response signal of first group of TianQin satellites ###

def hh_ff_res(theta_s, phi_s0, psi_s, hpf_plus_2f0, hpf_minus_2f0, hcf_plus_2f0, hcf_minus_2f0):

  phi_s01   = phi_s0; phi_s02 = phi_s0-np.pi/4.0

  hf_plus1  = 1.0/4*(1.0+np.cos(theta_s)**2)*np.cos(2*psi_s)*(np.exp(-2j*phi_s01)*hpf_plus_2f0+np.exp(2j*phi_s01)*hpf_minus_2f0)-1.0j/2*np.cos(theta_s)*np.sin(2*psi_s)*(np.exp(-2j*phi_s01)*hpf_plus_2f0-np.exp(2j*phi_s01)*hpf_minus_2f0)

  hf_cross1 = 1.0/4*(1.0+np.cos(theta_s)**2)*np.sin(2*psi_s)*(np.exp(-2j*phi_s01)*hcf_plus_2f0+np.exp(2j*phi_s01)*hcf_minus_2f0)+1.0j/2*np.cos(theta_s)*np.cos(2*psi_s)*(np.exp(-2j*phi_s01)*hcf_plus_2f0-np.exp(2j*phi_s01)*hcf_minus_2f0)

  hf_plus2  = 1.0/4*(1.0+np.cos(theta_s)**2)*np.cos(2*psi_s)*(np.exp(-2j*phi_s02)*hpf_plus_2f0+np.exp(2j*phi_s02)*hpf_minus_2f0)-1.0j/2*np.cos(theta_s)*np.sin(2*psi_s)*(np.exp(-2j*phi_s02)*hpf_plus_2f0-np.exp(2j*phi_s02)*hpf_minus_2f0)

  hf_cross2 = 1.0/4*(1.0+np.cos(theta_s)**2)*np.sin(2*psi_s)*(np.exp(-2j*phi_s02)*hcf_plus_2f0+np.exp(2j*phi_s02)*hcf_minus_2f0)+1.0j/2*np.cos(theta_s)*np.cos(2*psi_s)*(np.exp(-2j*phi_s02)*hcf_plus_2f0-np.exp(2j*phi_s02)*hcf_minus_2f0)


  signal1   = np.sqrt(3.0)/2*(hf_plus1+hf_cross1)

  signal2   = np.sqrt(3.0)/2*(hf_plus2+hf_cross2)
  
  return signal1, signal2 



### The response signal of two group of TianQin satellites ###


def hh_ff_res_two_group(theta_s_fg, phi_s0_fg, psi_s_fg, hp_p2f0_fg, hp_m2f0_fg, hc_p2f0_fg, hc_m2f0_fg, theta_s_sg, phi_s0_sg, psi_s_sg, hp_p2f0_sg, hp_m2f0_sg, hc_p2f0_sg, hc_m2f0_sg):

## the first group of satellites

  signal1_fg, signal2_fg = hh_ff_res(theta_s_fg, phi_s0_fg, psi_s_fg, hp_p2f0_fg, hp_m2f0_fg, hc_p2f0_fg, hc_m2f0_fg)
  
## the second group of satellites

  if hp_p2f0_sg == []:

    signal1_sg = []; signal2_sg = [];

  else:
    
    signal1_sg, signal2_sg = hh_ff_res(theta_s_sg, phi_s0_sg, psi_s_sg, hp_p2f0_sg, hp_m2f0_sg, hc_p2f0_sg, hc_m2f0_sg)

  return signal1_fg, signal2_fg, signal1_sg, signal2_sg 



### calculate the Sn(f) ###


def Sn_f_ET_D(f_list):

  f_list_p = []; asd_list_p = [];

  for line0 in open(path_interp+"/ET_D.txt"):

    line1 = line0.split()

    f_list_p.append(float(line1[0]))

    asd_list_p.append(float(line1[1]))

  asd_list = np.interp(f_list, f_list_p, asd_list_p)

  Snf_list = np.array(asd_list)**2;

  return Snf_list



def Sn_f_TQ_SA(f_list):

  LL       = np.sqrt(3.0)*10**8; Sa = 1e-30; Sx = 1e-24

  Snf_list = [];

  for ff1 in f_list:
    
    Snf1 = 20.0/3*1/LL**2*(4*Sa/(2*np.pi*ff1)**4*(1+1e-4/ff1)+Sx)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list


def Sn_f_TQ_NSA(f_list):

  LL = np.sqrt(3.0)*10**8; Sa = 1e-30; Sx = 1e-24

  Snf_list = [];

  Snf_list = 1/LL**2*(4*Sa/(2*np.pi*f_list)**4*(1+1e-4/f_list)+Sx)*(1+(f_list/(0.41*cc/(2*LL)))**2)

  return Snf_list

def Pn_f_TQ_NSA(f_list): # the TianQin noise curve

  LL = np.sqrt(3.0)*10**8; Sa = 1e-30; Sx = 1e-24

  Pnf_list = [];

  Pnf_list = 1/LL**2*(4*Sa/(2*np.pi*f_list)**4*(1+1e-4/f_list)+Sx)*(1+0*(f_list/(0.41*cc/(2*LL)))**2)

  return Pnf_list


def Sn_f_TQ_NSA_shock(f_list):

  LL = np.sqrt(3.0)*10**8; Sa = 1e-30; Sx = 1e-24

  xp = []; yp = [];

  for line0 in open('./RR.txt'):

    line1 = line0.split()

    xp.append(float(line1[0]))

    yp.append(float(line1[1]))

  rr = np.interp(f_list/f_star, xp, yp)

  Snf_list = [];

  Snf_list = 3.0/20*(1/LL**2*(4*Sa/(2*np.pi*f_list)**4*(1+1e-4/f_list)+Sx)/rr)

  return Snf_list

### LISA ###


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

### eLISA, N2A5 ###

def Sn_f_N2A5(f_list):

  LL = 5.0e9; Snsn = 2.96e-23; Snomn = 2.65e-23

  Snf_list = [];

  for ff1 in f_list:

    Snacc = 9e-30/(2*np.pi*ff1)**4*(1+1e-4/ff1)
    
    Snf1 = 20.0/3*1/LL**2*(4*Snacc+Snsn+Snomn)*(1+(ff1/(0.41*cc/(2*LL)))**2)

    Snf_list.append(Snf1)

  return Snf_list


### calculate the SNR ###

def inner_product_SNR(h1_f, h2_f, Sn_f, f_list):

  integrand = [(h1.conjugate()*h2)/Snf for (h1, h2, Snf) in zip(h1_f, h2_f, Sn_f)]

  integral  = 4.0*np.trapz(integrand, f_list).real

  return integral





def SNR_calc(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):

  theta_s, phi_s0, psi_s = angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                   = calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  ff, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0 = hh_ff_2PN_seg(m1r, m2r, zz, incl, ttrc, theta_s_bar, phi_s_bar, f0_red, ee0)

  Sn_f                   = Sn_f_TQ_NSA(ff)

  signal1_f, signal2_f   = hh_ff_res(theta_s, phi_s0, psi_s, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0)

  SNR1_sq                = inner_product_SNR(signal1_f, signal1_f, Sn_f, ff) 

  SNR2_sq                = inner_product_SNR(signal2_f, signal2_f, Sn_f, ff) 

  SNR_sq                 = SNR1_sq+SNR2_sq

  SNR                    = np.sqrt(SNR_sq)

  return SNR


def SNR_calc_two_group(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):
  
  theta_s_fg, phi_s0_fg, psi_s_fg = angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  theta_s_sg, phi_s0_sg, psi_s_sg = angles_in_res_func_sg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                            = calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  ff_fg, hp_p2f0_fg, hp_m2f0_fg, hc_p2f0_fg, hc_m2f0_fg, ff_sg, hp_p2f0_sg, hp_m2f0_sg, hc_p2f0_sg, hc_m2f0_sg = hh_ff_2PN_seg_two_group(m1r, m2r, zz, incl, ttrc, theta_s_bar, phi_s_bar, f0_red, ee0)

  Sn_f_fg                 = Sn_f_TQ_NSA(ff_fg)

  Sn_f_sg                 = Sn_f_TQ_NSA(ff_sg)

  signal1_f_fg, signal2_f_fg, signal1_f_sg, signal2_f_sg = hh_ff_res_two_group(theta_s_fg, phi_s0_fg, psi_s_fg, hp_p2f0_fg, hp_m2f0_fg, hc_p2f0_fg, hc_m2f0_fg, theta_s_sg, phi_s0_sg, psi_s_sg, hp_p2f0_sg, hp_m2f0_sg, hc_p2f0_sg, hc_m2f0_sg)

# first group of satellites

  SNR1_sq_fg              = inner_product_SNR(signal1_f_fg, signal1_f_fg, Sn_f_fg, ff_fg) 

  SNR2_sq_fg              = inner_product_SNR(signal2_f_fg, signal2_f_fg, Sn_f_fg, ff_fg) 

  SNR_sq_fg               = SNR1_sq_fg+SNR2_sq_fg

# second group of satellites

  
  SNR1_sq_sg              = inner_product_SNR(signal1_f_sg, signal1_f_sg, Sn_f_sg, ff_sg) 

  SNR2_sq_sg              = inner_product_SNR(signal2_f_sg, signal2_f_sg, Sn_f_sg, ff_sg) 

  SNR_sq_sg               = SNR1_sq_sg+SNR2_sq_sg

  SNR_sq                  = SNR_sq_fg+SNR_sq_sg

# the total SNR of two group of satellites

  SNR                     = np.sqrt(SNR_sq)

  return SNR 


def horizon_Newton(ttc_red, qq, SNR_threshold):

  Mr_total_list = []; zz_list = []; lum_dis_list = []

  def SNR11(zz):

#    return func_LISA.SNR_calc_0PN(m1r, m2r, zz, (ttc_red-tt_1yr)/(1+zz))-SNR_threshold
   
#    return SNR_calc_0PN_SA(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold

#    return SNR_calc_0PN_two_group_SA(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold
    
#    return SNR_calc_0PN_TQ_LISA_SA(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold # tc_red = 5yr for both TQ and LISA
     
#    return SNR_calc_0PN_TQ_LISA_SA_horizon(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold # tc_red = 5yr and 4yr for TQ and LISA respectively. Choose this 
   
#    return SNR_calc_0PN_TQ_two_group_LISA_SA(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold # tc_red = 5yr for both TQ and LISA

    return SNR_calc_0PN_TQ_two_group_LISA_SA_horizon(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold # tc=5yr and 4yr for TQ and LISA respectively. Choose this

################################################################
#    return SNR_calc(m1r, m2r, zz, ttc_red/(1+zz), theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0)-SNR_threshold
    
#    return SNR_calc_two_group(m1r, m2r, zz, ttc_red/(1+zz), theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0)-SNR_threshold
    

#    return func_LISA.SNR_calc_2PN(m1r, m2r, zz, ttc_red/(1+zz), theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0)-SNR_threshold

#  for Mr_total in np.logspace(1, 3, 20, endpoint = True):
  
  for Mr_total in [151]: # Mr_total of GW190521

    m1r = Mr_total/(1+qq); m2r = Mr_total*qq/(1+qq)

    zz = optimize.newton(SNR11, 0.00001)
    
    DL = lum_dis(zz) 

    Mr_total_list.append(Mr_total)

    lum_dis_list.append(DL)
        
    zz_list.append(zz)

  return np.array(Mr_total_list), np.array(lum_dis_list), np.array(zz_list) 


def horizon_ET_D_Newton(ttc_red, qq, SNR_threshold):

  Mr_total_list = []; zz_list = []; lum_dis_list = []

  def SNR11(zz):

    return SNR_calc_0PN_ET_D_SA(m1r, m2r, zz, ttc_red/(1+zz))-SNR_threshold

  for Mr_total in np.logspace(1, 3, 10, endpoint = True):

    m1r = Mr_total/(1+qq); m2r = Mr_total*qq/(1+qq)

#    zz = optimize.newton(SNR11, 0.00001)
    
    zz = optimize.newton(SNR11, 0.01)

    DL = lum_dis(zz) 

    Mr_total_list.append(Mr_total)

    lum_dis_list.append(DL)
        
    zz_list.append(zz)

#   print Mr_total, zz, lum_dis(zz) 

  return np.array(Mr_total_list), np.array(lum_dis_list), np.array(zz_list) 



def SNR_aLIGO(f_list, signal1_f, signal2_f, Sn_f):

  integrand1 = [signf1.conjugate()*signf1/Snf for (signf1, Snf) in zip(signal1_f, Sn_f)]
  
  #integrand2 = [signf2.conjugate()*signf2/Snf for (signf2, Snf) in zip(signal2_f, Sn_f)]

  SNR1 = np.sqrt(4.0*np.trapz(integrand1, f_list).real)

  #SNR2 = np.sqrt(4.0*np.trapz(integrand2, f_list).real)

  #SNR = np.sqrt(SNR1**2+SNR2**2)
  
  SNR = SNR1
  
  return SNR 



###      The part of parameters estimation     ###

def inner_product(h1_f, h2_f, Sn_f, f_list):

  integrand = [(h1.conjugate()*h2+h1*h2.conjugate())/Snf for (h1, h2, Snf) in zip(h1_f, h2_f, Sn_f)]

  integral  = 2.0*np.trapz(integrand, f_list)

  return integral


#def m1rm2r(mcr, eta):
#
#  aa  = mcr*eta**(-3.0/5)
#
#  bb  = 4*eta**(-1.0/5)*mcr**2
#
#  m1r = (aa+np.sqrt(aa**2-bb))/2
#
#  m2r = (aa-np.sqrt(aa**2-bb))/2
#
#  return m1r, m2r


def HH_ff(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
  
  theta_s, phi_s0, psi_s = angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                   = calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                   = incl-del_incl

  ff, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0                       = hh_ff_2PN_seg_p(mcr, eta, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)

  signal1, signal2       = hh_ff_res(theta_s, phi_s0, psi_s, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0)

  return ff, signal1, signal2




def HH_ff_two_group(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
   
  theta_s_fg, phi_s0_fg, psi_s_fg = angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)
  
  theta_s_sg, phi_s0_sg, psi_s_sg = angles_in_res_func_sg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                            = calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                            = incl-del_incl
 
  ff_fg, hp_p2f0_fg, hp_m2f0_fg, hc_p2f0_fg, hc_m2f0_fg, ff_sg, hp_p2f0_sg, hp_m2f0_sg, hc_p2f0_sg, hc_m2f0_sg = hh_ff_2PN_seg_two_group_p(mcr, eta, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)

  signal1_fg, signal2_fg, signal1_sg, signal2_sg = hh_ff_res_two_group(theta_s_fg, phi_s0_fg, psi_s_fg, hp_p2f0_fg, hp_m2f0_fg, hc_p2f0_fg, hc_m2f0_fg, theta_s_sg, phi_s0_sg, psi_s_sg, hp_p2f0_sg, hp_m2f0_sg, hc_p2f0_sg, hc_m2f0_sg)

  return ff_fg, ff_sg, signal1_fg, signal2_fg, signal1_sg, signal2_sg


def parameters_estimate(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  nn           = 8 

  del_pars_mat = np.zeros(shape = ([nn, nn]), dtype = float)

  del_pars     = np.array([del_ln_mcr, del_ln_eta, del_zz, del_ttcr, del_theta_s_bar, del_phi_s_bar, del_incl, del_ln_ee0])

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      if ii == jj:

        del_pars_mat[ii, jj] = del_pars[ii]

      elif ii != jj:

        del_pars_mat[ii, jj] = 0.0

  
  mcr0  = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1/5)

  mtr0  = m1r+m2r

  zz0   = zz

  ttcr0 = ttcr

# 在改变要估计的参数的时候，取固定不变的参数(mcr0, mtr0, ttcr0)的目的是保持频率不随待估计的参数改变而改变
  
  mcr   = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1.0/5)

  eta   = m1r*m2r/(m1r+m2r)**2

  freq0, signal1_0, signal2_0 = HH_ff(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, 0, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)
  
  signal1_list = []; signal2_list = [];


  for kk in range(len(del_pars)):

    signal1_kk, signal2_kk = HH_ff(mcr*np.exp(-del_pars_mat[kk,0]), eta*np.exp(-del_pars_mat[kk,1]), zz-del_pars_mat[kk,2], ttcr-del_pars_mat[kk,3], theta_s_bar-del_pars_mat[kk,4], phi_s_bar-del_pars_mat[kk,5], theta_l_bar, phi_l_bar, del_pars_mat[kk,6], f0_red, ee0*np.exp(-del_pars_mat[kk,7]), mcr0, mtr0, zz0, ttcr0)[1:]

    signal1_list.append(signal1_kk)

    signal2_list.append(signal2_kk)


  Sn_f  = Sn_f_TQ_NSA(freq0) 

  del_pars[2] = (lum_dis(zz)-lum_dis(zz-del_zz))/lum_dis(zz) # 把del_zz转化为del_DL/DL

  del_pars[3] = (1+zz)*del_pars[3] # del_pars[3] = del_ttcr, 因为估计的参数为红移之后的量，所以使其乘以(1+zz)

  Gamma = np.zeros(shape = ([len(del_pars), len(del_pars)]), dtype = float)

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      Gamma[ii, jj] = (inner_product((-signal1_list[ii]+signal1_0)/del_pars[ii], (-signal1_list[jj]+signal1_0)/del_pars[jj], Sn_f, freq0)+inner_product((-signal2_list[ii]+signal2_0)/del_pars[ii], (-signal2_list[jj]+signal2_0)/del_pars[jj], Sn_f, freq0)).real


  #return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5], Gamma[6,6], Gamma[7,7] 
# Gamma_McMc, Gamma_etatea, Gamma_DL_DL, Gamma_tctc, Gamma_theta_s_bar_theta_s_bar, Gamma_phi_s_bar_phi_s_bar, Gamma_incl_incl, Gamma_e0_e0

  Sigma         = np.linalg.inv(Gamma.real)

  Del_Omega_bar = 2*np.pi*abs(np.sin(theta_s_bar))*np.sqrt((Sigma[4,4]*Sigma[5,5]-Sigma[4,5]**2))*(180.0/np.pi)**2
 
  dL            = lum_dis(zz)

  Del_V         = 4.0/3*(dL)**3*Del_Omega_bar*3e-4*np.sqrt(Sigma[2,2])

  #return np.sqrt(Sigma[3,3]), Del_Omega_bar, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), np.sqrt(Sigma[2,2]), Del_V, np.sqrt(Sigma[6,6]), np.sqrt(Sigma[7,7])

  #return Sigma[0,1]

  return Gamma



### parameters estimation of two group of satellites ###

def parameters_estimate_two_group(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  nn           = 8 

  del_pars_mat = np.zeros(shape = ([nn, nn]), dtype = float)

  del_pars     = np.array([del_ln_mcr, del_ln_eta, del_zz, del_ttcr, del_theta_s_bar, del_phi_s_bar, del_incl, del_ln_ee0])

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      if ii == jj:

        del_pars_mat[ii, jj] = del_pars[ii]

      elif ii != jj:

        del_pars_mat[ii, jj] = 0.0
  
  
  mcr0  = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1.0/5)

  mtr0  = m1r+m2r
  
  zz0   = zz

  ttcr0 = ttcr

  
  mcr   = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1.0/5)

  eta   = m1r*m2r*(m1r+m2r)**(-2)

  freq0_fg, freq0_sg, signal1_0_fg, signal2_0_fg, signal1_0_sg, signal2_0_sg = HH_ff_two_group(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, 0, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)
  
  signal1_list_fg = []; signal2_list_fg = [];

  signal1_list_sg = []; signal2_list_sg = [];

  for kk in range(len(del_pars)):

    signal1_kk_fg, signal2_kk_fg, signal1_kk_sg, signal2_kk_sg = HH_ff_two_group(mcr*np.exp(-del_pars_mat[kk][0]), eta*np.exp(-del_pars_mat[kk][1]), zz-del_pars_mat[kk][2], ttcr-del_pars_mat[kk][3], theta_s_bar-del_pars_mat[kk][4], phi_s_bar-del_pars_mat[kk][5], theta_l_bar, phi_l_bar, del_pars_mat[kk,6], f0_red, ee0*np.exp(-del_pars_mat[kk, 7]), mcr0, mtr0, zz0, ttcr0)[2:]

    signal1_list_fg.append(signal1_kk_fg)

    signal2_list_fg.append(signal2_kk_fg)

    signal1_list_sg.append(signal1_kk_sg)

    signal2_list_sg.append(signal2_kk_sg)

  Sn_f_fg  = Sn_f_TQ_NSA(freq0_fg) 

  Sn_f_sg  = Sn_f_TQ_NSA(freq0_sg) 
  
  del_pars[2] = (lum_dis(zz)-lum_dis(zz-del_zz))/lum_dis(zz) 

  del_pars[3] = (1+zz)*del_pars[3]

  Gamma = np.zeros(shape = ([len(del_pars), len(del_pars)]), dtype = float)

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      Gamma[ii, jj] = (inner_product((-signal1_list_fg[ii]+signal1_0_fg)/del_pars[ii], (-signal1_list_fg[jj]+signal1_0_fg)/del_pars[jj], Sn_f_fg, freq0_fg)+inner_product((-signal2_list_fg[ii]+signal2_0_fg)/del_pars[ii], (-signal2_list_fg[jj]+signal2_0_fg)/del_pars[jj], Sn_f_fg, freq0_fg) + inner_product((-signal1_list_sg[ii]+signal1_0_sg)/del_pars[ii], (-signal1_list_sg[jj]+signal1_0_sg)/del_pars[jj], Sn_f_sg, freq0_sg)+inner_product((-signal2_list_sg[ii]+signal2_0_sg)/del_pars[ii], (-signal2_list_sg[jj]+signal2_0_sg)/del_pars[jj], Sn_f_sg, freq0_sg)).real

#  return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5], Gamma[6,6], Gamma[7,7] # Gamma_McMc, Gamma_etatea, Gamma_DLDL, Gamma_tctc, Gamma_theta_s_bar_theta_s_bar, Gamma_phi_s_bar_phi_s_bar, Gamma_incl_incl, Gamma_e0_e0

  Sigma         = np.linalg.inv(Gamma.real)

  Del_Omega_bar = 2*np.pi*abs(np.sin(theta_s_bar))*np.sqrt((Sigma[4,4]*Sigma[5,5]-Sigma[4,5]**2))*(180.0/np.pi)**2
  
  dL            = lum_dis(zz)

  Del_V         = 4.0/3*(dL)**3*Del_Omega_bar*3e-4*np.sqrt(Sigma[2,2])
###  
#  return np.sqrt(Sigma[3,3]), Del_Omega_bar, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), np.sqrt(Sigma[2,2]), Del_V, np.sqrt(Sigma[6,6]), np.sqrt(Sigma[7,7]) 

  return Gamma


############## Filter the Parameters of Source using sky and inclination average (0PN)  of TQ constellation I+II+LISA############## 这样筛选出来的源参数用于计算TQ constellation I, TQ constellation I+II, TQ constellation I+LISA, 和TQ constellation I+II+LISA的SNR(no sky and inclination average, 2PN) 


def freq_red_list_0PN(mtr, mcr, ttcr, zz):

  freq_red_i, freq_red_f = freq_red_i_f_0PN(mtr, mcr, ttcr, zz)

  index_i       = math.log(freq_red_i, 10)

  index_f       = math.log(freq_red_f, 10)

  index_range   = np.linspace(index_i, index_f, 20, endpoint = True)

  freq_red_list = np.array(np.power(10, index_range))

  return freq_red_list


### amplitude of 0PN ###

def hh_ff_amplitude_0PN_seg(m1r, m2r, zz, ttcr):
    
  mtr            = m1r+m2r
  
  mt_red         = mtr*(1+zz)
  
  mcr            = (m1r*m2r)**0.6/(m1r+m2r)**0.2
  
  mc_red         = mcr*(1+zz)
  
  ttc_red        = ttcr*(1+zz)
  
  dL             = lum_dis(zz)

  
  freq_red_list0         = freq_segmentate(mcr, mtr, zz, ttcr)


  freq_red_list1 = [];

  for kk in range(len(freq_red_list0)):

    fk_list        = freq_red_list0[kk]

    freq_red_list1 = list(freq_red_list1)+list(fk_list)
    
  freq_red_list1 = np.array(freq_red_list1)

  tt_0PN            = tt_f_0PN(freq_red_list1, mc_red, ttc_red)
  
  remainder         = np.array(np.floor(tt_0PN/tto_unit)%2, dtype = complex)

  index0            = np.where(remainder == 0)[0]
   
  ampli_f           = np.sqrt(4.0/5)*ampli_f_0PN(dL, mc_red, freq_red_list1[index0]) # the factor np.sqrt(4.0/5) comes from the averaging inclination

  remainder[index0] = ampli_f 

  index1            = np.where(remainder == 1)[0]

  remainder[index1] = [0.0]*len(index1)

  ampli_f_list      = remainder

  return np.array(freq_red_list1), np.array(ampli_f_list)




def hh_ff_amplitude_0PN_seg_two_group(m1r, m2r, zz, ttcr):
    
  mtr            = m1r+m2r
  
  mt_red         = mtr*(1+zz)
  
  mcr            = (m1r*m2r)**0.6/(m1r+m2r)**0.2
  
  mc_red         = mcr*(1+zz)
  
  ttc_red        = ttcr*(1+zz)
  
  dL             = lum_dis(zz)


  freq_red_list1 = freq_red_list_0PN(mtr, mcr, ttcr, zz) 
 
  ampli_f_list   = np.sqrt(4.0/5)*ampli_f_0PN(dL, mc_red, freq_red_list1) # the factor np.sqrt(4.0/5) comes from the averaging inclination

  return np.array(freq_red_list1), np.array(ampli_f_list)


### SNR, ET-D, 0PN, sky and inclination average ###

def SNR_calc_0PN_ET_D_SA(m1r, m2r, zz, ttrc):

  ff, signal_f = func_LISA.hh_ff_amplitude_0PN(m1r, m2r, zz, ttrc)
  
  signal_f     = np.sqrt(5/4)*(1/5)*signal_f
  # times sqrt(5/4) is to cancel sqrt(4/5) in signal_f, times 1/5 is to get the sky and inclination averaged signal
  
  Sn_f         = 4.0/3*Sn_f_ET_D(ff)
  # times 4/3 to convert it to a triangle-shape PSD
  
  SNR_sq       = inner_product_SNR(signal_f, signal_f, Sn_f, ff) 

  SNR          = np.sqrt(3*SNR_sq)
  # ET-D includes 3 interferometers

  return SNR


### SNR, TianQin, 0PN average ###

def SNR_calc_0PN_SA(m1r, m2r, zz, ttrc):


  ff, signal1_f     = hh_ff_amplitude_0PN_seg(m1r, m2r, zz, ttrc)
  
  Sn_f              = Sn_f_TQ_SA(ff)

  SNR1_sq           = inner_product_SNR(signal1_f, signal1_f, Sn_f, ff) 

  SNR2_sq           = SNR1_sq 

  SNR_sq            = SNR1_sq+SNR2_sq

  SNR               = np.sqrt(SNR_sq)

  return SNR 


def SNR_calc_0PN_two_group_SA(m1r, m2r, zz, ttrc):

  
  ff, signal1_f     = hh_ff_amplitude_0PN_seg_two_group(m1r, m2r, zz, ttrc)

  Sn_f              = Sn_f_TQ_SA(ff)

  SNR1_sq           = inner_product_SNR(signal1_f, signal1_f, Sn_f, ff) 

  SNR2_sq           = SNR1_sq 

  SNR_sq            = SNR1_sq+SNR2_sq

  SNR               = np.sqrt(SNR_sq)

  return SNR 


### SNR, 0PN, TianQin+LISA, average ###

def SNR_calc_0PN_TQ_LISA_SA(m1r, m2r, zz, ttrc):

  SNR_LISA    = func_LISA.SNR_calc_0PN(m1r, m2r, zz, ttrc)

  SNR_LISA_sq = SNR_LISA**2

  SNR_TQ      = SNR_calc_0PN_SA(m1r, m2r, zz, ttrc) 

  SNR_TQ_sq   = SNR_TQ**2

  SNR_TQ_LISA = np.sqrt(SNR_LISA_sq+SNR_TQ_sq)

  return SNR_TQ_LISA


def SNR_calc_0PN_TQ_LISA_SA_horizon(m1r, m2r, zz, ttrc):

  ttrc_LISA   = ttrc-tt_1yr/(1+zz)

  SNR_LISA    = func_LISA.SNR_calc_0PN(m1r, m2r, zz, ttrc_LISA)

  SNR_LISA_sq = SNR_LISA**2

  SNR_TQ      = SNR_calc_0PN_SA(m1r, m2r, zz, ttrc) 

  SNR_TQ_sq   = SNR_TQ**2

  SNR_TQ_LISA = np.sqrt(SNR_LISA_sq+SNR_TQ_sq)

  return SNR_TQ_LISA


def SNR_calc_0PN_TQ_two_group_LISA_SA(m1r, m2r, zz, ttrc):
  
  ff, signal1_f     = hh_ff_amplitude_0PN_seg_two_group(m1r, m2r, zz, ttrc)

# the SNR of TianQin(two group)

  Sn_f_TQ              = Sn_f_TQ_SA(ff)

  SNR1_sq_TQ           = inner_product_SNR(signal1_f, signal1_f, Sn_f_TQ, ff) 

  SNR2_sq_TQ           = SNR1_sq_TQ 

  SNR_sq_TQ            = SNR1_sq_TQ+SNR2_sq_TQ

# the SNR of LISA

  Sn_f_LISA              = Sn_f_LISA_SA(ff)

  SNR1_sq_LISA           = inner_product_SNR(signal1_f, signal1_f, Sn_f_LISA, ff) 

  SNR2_sq_LISA           = SNR1_sq_LISA 

  SNR_sq_LISA            = SNR1_sq_LISA+SNR2_sq_LISA

# the total SNR (TQ+LISA)

  SNR_TQ_LISA            = np.sqrt(SNR_sq_TQ+SNR_sq_LISA)

  return SNR_TQ_LISA


def SNR_calc_0PN_TQ_two_group_LISA_SA_horizon(m1r, m2r, zz, ttrc):
  
  ttrc_LISA             = ttrc-tt_1yr/(1+zz)

  SNR_LISA              = func_LISA.SNR_calc_0PN(m1r, m2r, zz, ttrc_LISA)

  SNR_LISA_sq           = SNR_LISA**2

  SNR_TQ_two_group      = SNR_calc_0PN_two_group_SA(m1r, m2r, zz, ttrc) 

  SNR_TQ_two_group_sq   = SNR_TQ_two_group**2

  SNR_TQ_two_group_LISA = np.sqrt(SNR_LISA_sq+SNR_TQ_two_group_sq)
 
  return SNR_TQ_two_group_LISA


##################### Calculate the SNR of TQ+LISA #######################(2PN, no sky and inclination average)

### SNR, 2PN, TianQin+LISA, no average ###

def SNR_calc_TQ_LISA(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):

  SNR_LISA    = func_LISA.SNR_calc_2PN(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0)

  SNR_LISA_sq = SNR_LISA**2

  SNR_TQ      = SNR_calc(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0) 

  SNR_TQ_sq   = SNR_TQ**2

  SNR_TQ_LISA = np.sqrt(SNR_LISA_sq+SNR_TQ_sq)

  return SNR_TQ_LISA

### SNR, 2PN, TianQin_two_group+LISA, no average ###

def SNR_calc_TQ_two_group_LISA(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):

  SNR_LISA    = func_LISA.SNR_calc_2PN(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0)

  SNR_LISA_sq = SNR_LISA**2

  SNR_TQ      = SNR_calc_two_group(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0) 

  SNR_TQ_sq   = SNR_TQ**2

  SNR_TQ_LISA = np.sqrt(SNR_LISA_sq+SNR_TQ_sq)

  return SNR_TQ_LISA

##################### Parameter Estimation TQ+LISA #######################

def parameters_estimate_TQ_LISA(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  Gamma_TQ   = parameters_estimate(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0)

  Gamma_LISA = func_LISA.parameters_estimate(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0)

  Gamma      = np.array(Gamma_TQ)+np.array(Gamma_LISA)

#  return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5] , Gamma[6,6], Gamma[7,7] # Gamma_McMc, Gamma_etatea, Gamma_DLDL, Gamma_tctc, Gamma_theta_s_bar_theta_s_bar, Gamma_phi_s_bar_phi_s_bar, Gamma_incl_incl, Gamma_e0_e0

  Sigma         = np.linalg.inv(Gamma.real)

  Del_Omega_bar = 2*np.pi*abs(np.sin(theta_s_bar))*np.sqrt((Sigma[4,4]*Sigma[5,5]-Sigma[4,5]**2))*(180.0/np.pi)**2
  
  
  dL            = lum_dis(zz)

  Del_V         = 4.0/3*(dL)**3*Del_Omega_bar*3e-4*np.sqrt(Sigma[2,2])
  
  return np.sqrt(Sigma[3,3]), Del_Omega_bar, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), np.sqrt(Sigma[2,2]), Del_V, np.sqrt(Sigma[6,6]), np.sqrt(Sigma[7,7])




def parameters_estimate_TQ_two_group_LISA(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  Gamma_TQ_two_group = parameters_estimate_two_group(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0)

  Gamma_LISA         = func_LISA.parameters_estimate(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0)

  Gamma              = np.array(Gamma_TQ_two_group)+np.array(Gamma_LISA)

#  return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5]# Gamma_McMc, Gamma_etatea, Gamma_DLDL, Gamma_tctc, Gamma_theta_s_bar_theta_s_bar, Gamma_phi_s_bar_phi_s_bar

  Sigma         = np.linalg.inv(Gamma.real)

  Del_Omega_bar = 2*np.pi*abs(np.sin(theta_s_bar))*np.sqrt((Sigma[4,4]*Sigma[5,5]-Sigma[4,5]**2))*(180.0/np.pi)**2
  
  
  dL            = lum_dis(zz)

  Del_V         = 4.0/3*(dL)**3*Del_Omega_bar*3e-4*np.sqrt(Sigma[2,2])
  
  return np.sqrt(Sigma[3,3]), Del_Omega_bar, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), np.sqrt(Sigma[2,2]), Del_V, np.sqrt(Sigma[6,6]), np.sqrt(Sigma[7,7])
#########################################################
#########################################################自己加的程序
def freq_segmentate_4_2(mcr, mtr, zz, ttcr):#质量红移并合时间

# 计算全天平均的情况时频率划分为10，计算非全天平均的情况时划分为40
    
  coeff   = 5/(96*(np.pi)**(8.0/3))
   
  fr_isco = 1.0/(np.pi*6**(3.0/2)*mtr*m_sun_in_s)
   
  fr_list = []; f_red_list = []
   
  if ttcr <= tto_unit1/(1+zz):
   
    fr_ii = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(ttcr))**(-3.0/8)
    
    fr_list.append(fr_ii)
    fr_list.append(fr_isco)
    
  if tto_unit1/(1+zz)< ttcr <= tto/(1+zz):
      factor = int(ttcr/(tto_unit2/(1+zz)))*2
      if ttcr % ((tto_unit2)/(1+zz))==0:#(余数<4)#
          for ii in range(int(factor)):
              if (ii % 2) == 0:#偶数
                  del_tr = ttcr-int(ii/2)*tto_unit2/(1+zz)
                  fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
                  fr_list.append(fr_ii)
                  
              else:#奇数
                  del_tr = ttcr-int(ii/2)*tto_unit2/(1+zz)-(tto_unit2-tto_unit1)/(1+zz)
                  fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
                  fr_list.append(fr_ii)
                  
        
        
      else:#(余数>4)
        #test=int(factor)+1
          for ii in range(int(factor)+1):
              if (ii % 2) == 0:#偶数
                  del_tr = ttcr-int(ii/2)*tto_unit2/(1+zz)
                  fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
                  fr_list.append(fr_ii)
        
              else:#奇数
                  del_tr = ttcr-int(ii/2)*tto_unit2/(1+zz)-(tto_unit2-tto_unit1)/(1+zz)
                  fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
                  fr_list.append(fr_ii)  
      fr_list.append(fr_isco)     
    
  elif ttcr > tto/(1+zz):
      factor = int(tto/(tto_unit2))*2#最大值取整
      for ii in range(factor+1):
          if (ii % 2) == 0:#偶数
              del_tr = ttcr-int(ii/2)*tto_unit2/(1+zz)
              fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
              fr_list.append(fr_ii)  
          else:#奇数
              del_tr = ttcr-int(ii/2)*tto_unit2/(1+zz)-(tto_unit2-tto_unit1)/(1+zz)   
              fr_ii  = (8.0/(3*coeff)*(mcr*m_sun_in_s)**(5.0/3)*(del_tr))**(-3.0/8)
              fr_list.append(fr_ii)
      
       
                   
  f_red_list     = np.array(fr_list)/(1+zz)


  factor1        = int(len(f_red_list)/2)
# f_red_list中频率的个数除以2然后取整，例如int(3/2)=int(1.5)=1
  #print(f_red_list)
  f_red_list[0] = f_red_list[0]-2*f0

  f_red_list[factor1*2-1] = f_red_list[factor1*2-1]+2*f0 # 加减2f0的原因是f的取值范围是[f_red_i-2f0, f_red_f+2f0]


  freq_red_list  = []

  for ii in range(2*factor1-1):

#    f_red_list1 = np.linspace(f_red_list[0+ii], f_red_list[1+ii], 1e5, endpoint = True)

    index_i     = math.log(f_red_list[0+ii], 10)
  
    index_f     = math.log(f_red_list[1+ii], 10)
  
    index_range = np.linspace(index_i, index_f, 40, endpoint = True)
  
    f_red_list1 = np.power(10, index_range) 
  
    freq_red_list.append(f_red_list1)


  return freq_red_list

def hh_ff_2PN_seg_4_2(m1r, m2r, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0):
    
  mtr            = m1r+m2r
  
  mt_red         = mtr*(1+zz)
  
  mcr            = (m1r*m2r)**0.6/(m1r+m2r)**0.2
  
  mc_red         = mcr*(1+zz)
  
  ttc_red        = ttcr*(1+zz)
  
  phic           = -np.pi/4
 
  dL             = lum_dis(zz)

  eta            = m1r*m2r/(m1r+m2r)**2
 
  vv             = np.cos(incl)
  
  chip           = (1+vv**2)/2
  
  chic           = -1j*vv

  freq_red_list0         = freq_segmentate_4_2(mcr, mtr, zz, ttcr)


  freq_red_i, freq_red_f = freq_red_list0[0][0]+2*f0, freq_red_list0[-1][-1]-2*f0 # 加减2f0的目的是算出f_red_i 和 f_red_f 方便为下面判断波形是否为0

  freq_red_list1 = [];

  for kk in range(len(freq_red_list0)):

    fk_list        = freq_red_list0[kk]

    freq_red_list1 = list(freq_red_list1)+list(fk_list)
    

  f_red_m2f0      = np.array(freq_red_list1)-2*f0*1

  f_red_p2f0      = np.array(freq_red_list1)+2*f0*1 


  hp_minus2f0_list = []; hp_plus2f0_list = [];
  
  hc_minus2f0_list = []; hc_plus2f0_list = [];

# hplus(f-2f0) and hcross(f-2f0)
  
  index                              = list(np.where(f_red_m2f0 < freq_red_i)[0])+list(np.where(f_red_m2f0 > freq_red_f)[0]) 
  
  tt_0PN_fm2f0                       = tt_f_0PN(f_red_m2f0, mc_red, ttc_red)
  
  tt_2PN_fm2f0                       = tt_f_2PN(f_red_m2f0, mc_red, ttc_red, eta, f0_red, ee0)
#########bb做了修改 余数2
  hp_minus2f0_list, hc_minus2f0_list = hp_hc_fm2f0_4_2(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0,1, 2, theta_s_bar, phi_s_bar, f0_red, ee0)

# hplus(f+2f0) and hcross(f+2f0)

  index                            = list(np.where(f_red_p2f0 < freq_red_i)[0])+list(np.where(f_red_p2f0 > freq_red_f)[0]) 
   
  tt_0PN_fp2f0                     = tt_f_0PN(f_red_p2f0, mc_red, ttc_red)
   
  tt_2PN_fp2f0                     = tt_f_2PN(f_red_p2f0, mc_red, ttc_red, eta, f0_red, ee0)
#########bb做了修改 余数2 
  hp_plus2f0_list, hc_plus2f0_list = hp_hc_fp2f0_4_2(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0,1,2, theta_s_bar, phi_s_bar, f0_red, ee0)
 
  return np.array(freq_red_list1), np.array(hp_plus2f0_list), np.array(hp_minus2f0_list), np.array(hc_plus2f0_list), np.array(hc_minus2f0_list)

####2+4中的修改完整版f-2f0
def hp_hc_fm2f0_2_4(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, aa, bb,cc, theta_s_bar, phi_s_bar, f0_red, ee0):


    remainder         = np.array(np.floor(tt_0PN_fm2f0/tto_unit0)%3, dtype = complex)

    remainder[index]  = [bb]*len(index)

    index0            = np.where(remainder == aa)[0]#+np.where(remainder == bb)[0]#除以6的余数0
    
    
    hh0_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_m2f0[index0], tt_2PN_fm2f0[index0], theta_s_bar, phi_s_bar, f0_red, ee0)
  
    remainder[index0] = hh0_ff 
    index1            = np.where(remainder == bb)[0]#除以6的余数2
    index2            = np.where(remainder == cc)[0]#bb=1 cc=2
    remainder[index1] = [0.0]*len(index1)#中括号是list数据类型
    remainder[index2] = [0.0]*len(index2)#中括号是list数据类型
    hp_minus2f0_list  = chip*remainder

    hc_minus2f0_list  = chic*remainder

    return np.array(hp_minus2f0_list), np.array(hc_minus2f0_list)
####2+4的f+2f0
def hp_hc_fp2f0_2_4(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, aa, bb,cc, theta_s_bar, phi_s_bar, f0_red, ee0):
 

  remainder         = np.array(np.floor(tt_0PN_fp2f0/tto_unit0)%3, dtype = complex)#/2个月/3求余数

  remainder[index]  = [bb]*len(index)

  index0            = np.where(remainder == aa)[0]#+np.where(remainder == bb)[0]

  #index_gap=index0 +index1 

  hh0_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_p2f0[index0], tt_2PN_fp2f0[index0], theta_s_bar, phi_s_bar, f0_red, ee0)

  remainder[index0] = hh0_ff 

  index1            = np.where(remainder == bb)[0]#除以6的余数1
  index2            = np.where(remainder == cc)[0]#bb=1 cc=2只有余数是2的不行
  remainder[index1] = [0.0]*len(index1)#中括号是list数据类型
  remainder[index2] = [0.0]*len(index2)#中括号是list数据类型


  hp_plus2f0_list   = chip*remainder

  hc_plus2f0_list   = chic*remainder

  return np.array(hp_plus2f0_list), np.array(hc_plus2f0_list)
####4+2中的修改完整版f-2f0
def hp_hc_fm2f0_4_2(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, aa, bb,cc, theta_s_bar, phi_s_bar, f0_red, ee0):


    remainder         = np.array(np.floor(tt_0PN_fm2f0/tto_unit0)%3, dtype = complex)

    remainder[index]  = [cc]*len(index)

    index0            = np.where(remainder == aa)[0]#+np.where(remainder == bb)[0]#除以6的余数0
    
    
    hh0_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_m2f0[index0], tt_2PN_fm2f0[index0], theta_s_bar, phi_s_bar, f0_red, ee0)
  
    remainder[index0] = hh0_ff 
    index1            = np.where(remainder == bb)[0]#除以6的余数2
    hh1_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_m2f0[index1], tt_2PN_fm2f0[index1], theta_s_bar, phi_s_bar, f0_red, ee0)
    remainder[index1] = hh1_ff
    index2            = np.where(remainder == cc)[0]#bb=1 cc=2
    #remainder[index1] = [0.0]*len(index1)#中括号是list数据类型
    remainder[index2] = [0.0]*len(index2)#中括号是list数据类型
    hp_minus2f0_list  = chip*remainder

    hc_minus2f0_list  = chic*remainder

    return np.array(hp_minus2f0_list), np.array(hc_minus2f0_list)
####4+2的f+2f0
def hp_hc_fp2f0_4_2(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, aa, bb,cc, theta_s_bar, phi_s_bar, f0_red, ee0):
 

  remainder         = np.array(np.floor(tt_0PN_fp2f0/tto_unit0)%3, dtype = complex)#/2个月/3求余数

  remainder[index]  = [cc]*len(index)

  index0            = np.where(remainder == aa)[0]#+np.where(remainder == bb)[0]

  #index_gap=index0 +index1 

  hh0_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_p2f0[index0], tt_2PN_fp2f0[index0], theta_s_bar, phi_s_bar, f0_red, ee0)

  remainder[index0] = hh0_ff 

  index1            = np.where(remainder == bb)[0]#除以6的余数1
  hh1_ff            = hh0_ff_2PN(mc_red, eta, dL, ttc_red, phic, f_red_p2f0[index1], tt_2PN_fp2f0[index1], theta_s_bar, phi_s_bar, f0_red, ee0)
  remainder[index1] = hh1_ff
  index2            = np.where(remainder == cc)[0]#bb=1 cc=2只有余数是2的不行
  #remainder[index1] = [0.0]*len(index1)#中括号是list数据类型
  remainder[index2] = [0.0]*len(index2)#中括号是list数据类型


  hp_plus2f0_list   = chip*remainder

  hc_plus2f0_list   = chic*remainder

  return np.array(hp_plus2f0_list), np.array(hc_plus2f0_list)

def SNR_calc_4_2(m1r, m2r, zz, ttrc, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, f0_red, ee0):

  theta_s, phi_s0, psi_s = angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                   = calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  ff, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0 = hh_ff_2PN_seg_4_2(m1r, m2r, zz, incl, ttrc, theta_s_bar, phi_s_bar, f0_red, ee0)

  Sn_f                   = Sn_f_TQ_NSA(ff)

  signal1_f, signal2_f   = hh_ff_res(theta_s, phi_s0, psi_s, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0)

  SNR1_sq                = inner_product_SNR(signal1_f, signal1_f, Sn_f, ff) 

  SNR2_sq                = inner_product_SNR(signal2_f, signal2_f, Sn_f, ff) 

  SNR_sq                 = SNR1_sq+SNR2_sq

  SNR                    = np.sqrt(SNR_sq)

  return SNR
def hh_ff_2PN_seg_p_4_2(mcr, eta, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
   
  mc_red0        = mcr0*(1+zz0)

  ttc_red0       = ttcr0*(1+zz0)
   
  mc_red         = mcr*(1+zz0)

  ttc_red        = ttcr*(1+zz0)
  
  phic           = -np.pi/4
 
  dL             = lum_dis(zz)

  vv             = np.cos(incl)
  
  chip           = (1+vv**2)/2
  
  chic           = -1j*vv

  freq_red_list0         = freq_segmentate_4_2(mcr0, mtr0, zz0, ttcr0)

  freq_red_i, freq_red_f = freq_red_list0[0][0]+2*f0, freq_red_list0[-1][-1]-2*f0

  
  freq_red_list1 = [];

  for kk in range(len(freq_red_list0)):

    fk_list        = freq_red_list0[kk]

    freq_red_list1 = list(freq_red_list1)+list(fk_list)
  
  
  f_red_m2f0      = np.array(freq_red_list1)-2*f0*1

  f_red_p2f0      = np.array(freq_red_list1)+2*f0*1 


  hp_minus2f0_list = []; hp_plus2f0_list = [];
  
  hc_minus2f0_list = []; hc_plus2f0_list = [];

# hplus(f-2f0) and hcross(f-2f0)

  index                              = list(np.where(f_red_m2f0 < freq_red_i)[0])+list(np.where(f_red_m2f0 > freq_red_f)[0]) 
   
  tt_0PN_fm2f0                       = tt_f_0PN(f_red_m2f0, mc_red0, ttc_red0)
   
  tt_2PN_fm2f0                       = tt_f_2PN(f_red_m2f0, mc_red, ttc_red, eta, f0_red, ee0)

  hp_minus2f0_list, hc_minus2f0_list = hp_hc_fm2f0_4_2(index, tt_0PN_fm2f0, tt_2PN_fm2f0, f_red_m2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1,2, theta_s_bar, phi_s_bar, f0_red, ee0)

# hplus(f+2f0) and hcross(f+2f0)


  index                            = list(np.where(f_red_p2f0 < freq_red_i)[0])+list(np.where(f_red_p2f0 > freq_red_f)[0]) 
   
  tt_0PN_fp2f0                     = tt_f_0PN(f_red_p2f0, mc_red0, ttc_red0)
   
  tt_2PN_fp2f0                     = tt_f_2PN(f_red_p2f0, mc_red, ttc_red, eta, f0_red, ee0)

  hp_plus2f0_list, hc_plus2f0_list = hp_hc_fp2f0_4_2(index, tt_0PN_fp2f0, tt_2PN_fp2f0, f_red_p2f0, mc_red, eta, dL, ttc_red, phic, chip, chic, 0, 1, 2,theta_s_bar, phi_s_bar, f0_red, ee0)

  return np.array(freq_red_list1), np.array(hp_plus2f0_list), np.array(hp_minus2f0_list), np.array(hc_plus2f0_list), np.array(hc_minus2f0_list)
def HH_ff_4_2(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, mcr0, mtr0, zz0, ttcr0):
  
  theta_s, phi_s0, psi_s = angles_in_res_func_fg(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                   = calc_incl(theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar)

  incl                   = incl-del_incl

  ff, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0 = hh_ff_2PN_seg_p_4_2(mcr, eta, zz, incl, ttcr, theta_s_bar, phi_s_bar, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)

  signal1, signal2       = hh_ff_res(theta_s, phi_s0, psi_s, hp_plus_2f0, hp_minus_2f0, hc_plus_2f0, hc_minus_2f0)

  return ff, signal1, signal2
def parameters_estimate_4_2(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  nn           = 8 

  del_pars_mat = np.zeros(shape = ([nn, nn]), dtype = float)

  del_pars     = np.array([del_ln_mcr, del_ln_eta, del_zz, del_ttcr, del_theta_s_bar, del_phi_s_bar, del_incl, del_ln_ee0])

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      if ii == jj:

        del_pars_mat[ii, jj] = del_pars[ii]

      elif ii != jj:

        del_pars_mat[ii, jj] = 0.0

  
  mcr0  = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1/5)

  mtr0  = m1r+m2r

  zz0   = zz

  ttcr0 = ttcr

# 在改变要估计的参数的时候，取固定不变的参数(mcr0, mtr0, ttcr0)的目的是保持频率不随待估计的参数改变而改变
  
  mcr   = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1.0/5)

  eta   = m1r*m2r/(m1r+m2r)**2

  freq0, signal1_0, signal2_0 = HH_ff_4_2(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, 0, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)
  
  signal1_list = []; signal2_list = [];


  for kk in range(len(del_pars)):

    signal1_kk, signal2_kk = HH_ff_4_2(mcr*np.exp(-del_pars_mat[kk,0]), eta*np.exp(-del_pars_mat[kk,1]), zz-del_pars_mat[kk,2], ttcr-del_pars_mat[kk,3], theta_s_bar-del_pars_mat[kk,4], phi_s_bar-del_pars_mat[kk,5], theta_l_bar, phi_l_bar, del_pars_mat[kk,6], f0_red, ee0*np.exp(-del_pars_mat[kk,7]), mcr0, mtr0, zz0, ttcr0)[1:]

    signal1_list.append(signal1_kk)

    signal2_list.append(signal2_kk)


  Sn_f  = Sn_f_TQ_NSA(freq0) 

  del_pars[2] = (lum_dis(zz)-lum_dis(zz-del_zz))/lum_dis(zz) # 把del_zz转化为del_DL/DL

  del_pars[3] = (1+zz)*del_pars[3] # del_pars[3] = del_ttcr, 因为估计的参数为红移之后的量，所以使其乘以(1+zz)

  Gamma = np.zeros(shape = ([len(del_pars), len(del_pars)]), dtype = float)

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      Gamma[ii, jj] = (inner_product((-signal1_list[ii]+signal1_0)/del_pars[ii], (-signal1_list[jj]+signal1_0)/del_pars[jj], Sn_f, freq0)+inner_product((-signal2_list[ii]+signal2_0)/del_pars[ii], (-signal2_list[jj]+signal2_0)/del_pars[jj], Sn_f, freq0)).real


#  return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5], Gamma[6,6], Gamma[7,7] 
# Gamma_McMc, Gamma_etatea, Gamma_DL_DL, Gamma_tctc, Gamma_theta_s_bar_theta_s_bar, Gamma_phi_s_bar_phi_s_bar, Gamma_incl_incl, Gamma_e0_e0

  Sigma         = np.linalg.inv(Gamma.real)

  Del_Omega_bar = 2*np.pi*abs(np.sin(theta_s_bar))*np.sqrt((Sigma[4,4]*Sigma[5,5]-Sigma[4,5]**2))*(180.0/np.pi)**2
 
  dL            = lum_dis(zz)

  Del_V         = 4.0/3*(dL)**3*Del_Omega_bar*3e-4*np.sqrt(Sigma[2,2])

  return np.sqrt(Sigma[3,3]), Del_Omega_bar, np.sqrt(Sigma[0,0]), np.sqrt(Sigma[1,1]), np.sqrt(Sigma[2,2]), Del_V, np.sqrt(Sigma[6,6]), np.sqrt(Sigma[7,7])

  #return Sigma[0,1]

#  return Gamma
def parameters_estimate_test(m1r, m2r, del_ln_mcr, del_ln_eta, zz, del_zz, ttcr, del_ttcr, theta_s_bar, del_theta_s_bar, phi_s_bar, del_phi_s_bar, theta_l_bar, phi_l_bar, del_incl, f0_red, ee0, del_ln_ee0):

  nn           = 8 

  del_pars_mat = np.zeros(shape = ([nn, nn]), dtype = float)

  del_pars     = np.array([del_ln_mcr, del_ln_eta, del_zz, del_ttcr, del_theta_s_bar, del_phi_s_bar, del_incl, del_ln_ee0])

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      if ii == jj:

        del_pars_mat[ii, jj] = del_pars[ii]

      elif ii != jj:

        del_pars_mat[ii, jj] = 0.0

  
  mcr0  = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1/5)

  mtr0  = m1r+m2r

  zz0   = zz

  ttcr0 = ttcr

# 在改变要估计的参数的时候，取固定不变的参数(mcr0, mtr0, ttcr0)的目的是保持频率不随待估计的参数改变而改变
  
  mcr   = (m1r*m2r)**(3.0/5)*(m1r+m2r)**(-1.0/5)

  eta   = m1r*m2r/(m1r+m2r)**2

  freq0, signal1_0, signal2_0 = HH_ff(mcr, eta, zz, ttcr, theta_s_bar, phi_s_bar, theta_l_bar, phi_l_bar, 0, f0_red, ee0, mcr0, mtr0, zz0, ttcr0)
  
  signal1_list = []; signal2_list = [];


  for kk in range(len(del_pars)):

    signal1_kk, signal2_kk = HH_ff(mcr*np.exp(-del_pars_mat[kk,0]), eta*np.exp(-del_pars_mat[kk,1]), zz-del_pars_mat[kk,2], ttcr-del_pars_mat[kk,3], theta_s_bar-del_pars_mat[kk,4], phi_s_bar-del_pars_mat[kk,5], theta_l_bar, phi_l_bar, del_pars_mat[kk,6], f0_red, ee0*np.exp(-del_pars_mat[kk,7]), mcr0, mtr0, zz0, ttcr0)[1:]

    signal1_list.append(signal1_kk)

    signal2_list.append(signal2_kk)


  Sn_f  = Sn_f_TQ_NSA(freq0) 

  del_pars[2] = (lum_dis(zz)-lum_dis(zz-del_zz))/lum_dis(zz) # 把del_zz转化为del_DL/DL

  del_pars[3] = (1+zz)*del_pars[3] # del_pars[3] = del_ttcr, 因为估计的参数为红移之后的量，所以使其乘以(1+zz)

  Gamma = np.zeros(shape = ([len(del_pars), len(del_pars)]), dtype = float)

  for ii in range(len(del_pars)):

    for jj in range(len(del_pars)):

      Gamma[ii, jj] = (inner_product((-signal1_list[ii]+signal1_0)/del_pars[ii], (-signal1_list[jj]+signal1_0)/del_pars[jj], Sn_f, freq0)+inner_product((-signal2_list[ii]+signal2_0)/del_pars[ii], (-signal2_list[jj]+signal2_0)/del_pars[jj], Sn_f, freq0)).real


  return Gamma[0,0], Gamma[1,1], Gamma[2,2], Gamma[3,3], Gamma[4,4], Gamma[5,5], Gamma[6,6], Gamma[7,7] 
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
#5+1


