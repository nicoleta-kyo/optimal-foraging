"""
Optimal foraging.

Create the environment.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
import pickle as pkl

path = 'C:/nikyk/.Mallorca - UIB/Stochastic Simulation Methods/Indiv Project/env/'

# --------------------------------------- ENVIRONMENT


def add_patch(grid, top_left, rad):
    """
    takes a grid and adds a circular patch as 1s to it

    Parameters
    ----------
    grid : numpy array
        the grid to be transformed.
    top_left : tuple
        the coordinates of the top left rectangle of the grid where the circle will be.
    rad : int
        radius of the circle.

    Returns
    -------
    grid : numpy array
        the transformed grid

    """
    
    inds = ((top_left[0], top_left[0]+(2*rad)+1),(top_left[1],top_left[1]+(2*rad)+1))
    x,y = np.ogrid[-rad: rad+1, -rad: rad+1]
    mask = x**2+y**2 <= rad**2
    grid[inds[0][0]:inds[0][1], inds[1][0]:inds[1][1]][mask] = 1
    
    return grid

def save_int_env(path, env_name, env_int):
    
    env_int_name = path+env_name+'.pkl'
    with open(env_int_name, 'wb') as file:   
        pkl.dump(env_int, file)
    # check save is correct
    with open(env_int_name, 'rb') as file: 
        res_test = pkl.load(file)
        print('Object saved is equal to the one created: {}'.format(np.array_equal(env_int, res_test)))

def gen_random_gamma(a, rng, dim):
    
    def gen_x():
    
        u = rng.uniform()
        b = (math.e+a)/math.e
        p = b*u 
        
        if p < 1:                    # 0 < x <= 1
            x = p**(1/a)
            lt_1 = True
        else:                        # x > 1
            x = -1*np.log((b-p)/a)
            lt_1 = False
        
        return x, lt_1
    
    rnums = np.zeros(dim**2)
    
    for i in range(rnums.size):
    
        x, lt_1 = gen_x()
        accept = False
        while accept == False:
            if lt_1 == True:                  # 0 < x <= 1
                if rng.uniform() < np.exp(-x):
                    accept = True
                else:
                    x, lt_1 = gen_x()
            else:                                # x > 1
                if rng.uniform() < x**(a-1):
                    accept = True
                else:
                    x, lt_1 = gen_x()
    
        rnums[i] = x
    
    return rnums.reshape((dim,dim))

# 1 -- noise

    
seed = 123456
rng = np.random.default_rng(seed)

a = 0.2
d = 100
gamma_noise = gen_random_gamma(a, rng, d)
g_scale = 1
plt.imshow(gamma_noise )
plt.axis('off')
plt.savefig(path+'noise.png',format='png',dpi=800)
plt.tight_layout()
plt.show()

# - add blur

b_sigma = 3
gamma_noise_b = ndimage.gaussian_filter(gamma_noise, b_sigma)
plt.imshow(gamma_noise_b )
plt.axis('off')
plt.savefig(path+'noise+blur.png',format='png',dpi=800)
plt.show()

# 2 -- gauss patches

patches = np.zeros((d,d))

rads = [12,8,4]
starts = [(17,53), (42,20), (63,60)]

for rad,start in zip(rads,starts):
    patches = add_patch(patches, start, rad)

plt.imshow(patches,interpolation='none')
plt.axis('off')
plt.savefig(path+'patches.png',format='png',dpi=800)
plt.show()

# add blur
b_sigma = 10
patches = ndimage.gaussian_filter(patches, b_sigma)
plt.imshow(patches,interpolation='none')
plt.axis('off')
plt.savefig(path+'patches+blur.png',format='png',dpi=800)
plt.show()

patches_n = patches/np.sum(patches)
fname = 'food-res_patches-only'
save_int_env(path, fname, patches_n)

# 3 -- add together noise and patches

m1 = np.sum(patches)/np.sum(gamma_noise_b) # scale noise so that it is 50% of the total prob
m2 = np.sum(patches)/2/np.sum(gamma_noise_b) # scale noise so that it is 50% of the patches

for i,m in enumerate([m1,m2]):
    
    noise_min = m*gamma_noise_b
    env = noise_min + patches
    env = env/np.sum(env) #normalise
    
    fname = 'food-res_patch-noise{}to1'.format(i+1)
    
    #save as pkl
    with open(path+fname+'.pkl', 'wb') as file:
          
        pkl.dump(env, file)
    
    # save img
    plt.imshow(env )
    plt.axis('off')
    plt.title('Target Distribution')
    
    plt.savefig(path+fname+'.eps',format='eps')
    plt.savefig(path+fname+'.png',format='png',dpi=1200)
    
    plt.show()
    

# test array saved

with open(path+fname+'.pkl', 'rb') as file:
      
    res_test = pkl.load(file)

np.array_equal(env, res_test)


# # 4 -- interpolate for continuous version


# fname = 'food-res_patch-noise{}to1'
# for i in range(2):
    
#     env_name = fname.format(i+1)
#     with open(path+env_name+'.pkl', 'rb') as file:
#         env = pkl.load(file)
    
#     # crate interpolation
#     x,y = ([np.arange(i) for i in env.shape])    
#     interp = RectBivariateSpline(x, y, z=env, kx=2, ky=2)
#     dx = 0.01
#     x,y = ([np.arange(0, i, dx) for i in env.shape])    
#     env_int = interp(x, y, dx=0, dy=0, grid=True) # values
#     plt.imshow(env_int)
#     plt.show()
    
#     env_int = env_int/np.sum(env_int)# normalise
    
#     env_name = env_name+'-int'
#     save_int_env(path, env_name, env_int) # save env


