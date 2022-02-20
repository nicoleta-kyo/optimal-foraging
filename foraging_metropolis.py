
"""
Optimal foraging.

Run discrete and continuous metropolis.

Results:
    - the distribution of the ants
    - a sample trajectory
    - RMSD
    - cross-entropy between samples and target
    
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle as pkl
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
import pickle
import scipy.stats
from math import log2
import matplotlib.patheffects as pe

def metropolis_MC_entropy(M, pos, u, delta, randgen, ent_p, int_type = 'none', interp_obj = None):
    
    if pos is None:
        pos = randgen.integers(low=0, high=100, size = 2)
    
    samples = np.zeros((M,2))
    
    hs = np.zeros(M)
    p = u.flatten()
    
    for i in tqdm(range(M)):
        pos,_ = metropolis_one_mc(pos, u, delta, randgen, int_type, interp_obj)
        samples[i,:] = pos
        
        # calc the H(p,q) at every 1000 t
        q = np.histogram2d(samples[:i+1,0], samples[:i+1,1], bins=u.shape[0])[0].flatten()
        q = np.where(q == 0, np.nextafter(np.float32(0), np.float32(1)), q)
        # q = q/np.sum(q) # the routine does that
        dkl = scipy.stats.entropy(p, q)
        h = ent_p + dkl
        hs[i] = h
        
    return samples, hs, q


def metropolis_MC_rmsd(N, M, pos, u, delta, randgen, int_type = 'none', interp_obj = None):
    
    if pos is None:
        pos = randgen.integers(low=0, high=100, size = 2)

    samples = np.zeros((M,2))
    samples_std = np.zeros((M,2))
    
    msds = np.zeros(M)
    for ant in tqdm(range(N)):
        
        ant = np.zeros((M,2))
        for i in range(M):
            pos,_ = metropolis_one_mc(pos, u, delta, randgen, int_type, interp_obj)
            ant[i] = pos
            
            samples[i,:] += pos
            samples_std[i,:] += pos**2
            
        msd_ant = msd_straight_forward(ant)
        msds += msd_ant     
        
    samples = samples/N
    samples_std = np.sqrt((samples_std/N-samples**2)/N)
    rmsds = np.sqrt(msds/N)
    
    return samples, samples_std, rmsds


def metropolis_MC(M, pos, u, delta, randgen, int_type = 'none', interp_obj = None):
    
    if pos is None:
        pos = randgen.integers(low=0, high=100, size = 2)
    
    samples = np.zeros((M,2))
    for i in range(M):
        pos,_ = metropolis_one_mc(pos, u, delta, randgen, int_type, interp_obj)
        samples[i,:] = pos
        
    return samples

def metropolis_one_mc(pos, u, delta, randgen, int_type='none', interp_obj=None):
    """
    

    Parameters
    ----------
    pos : numpy array
        the initial position of the ant.
    u : numpy array
        the 2-D environment.
    delta : TYPE
        DESCRIPTION.
    randgen : numpy.default_rng
        the random generator.
    int_type : string, optional
        either none, or the type of interpolator object used: 'lin' or 'spline'. The default is 'none'.
    interp_obj : scipy interpolate object, optional
        the scipy interpolator. The default is None.

    Returns
    -------
    posnew : numpy array
        the coordinates of the new position.
    prob : int
        0 or 1 - whether the proposal was accepted or not.

    """
    
    pos_min = 0
    pos_max = 99
    
    if int_type == 'none':
        
        posnew = pos + randgen.integers(low=0,high=3,size=2)-1 # change is +1,0 or -1
    
        while((posnew < pos_min).any() | (posnew > pos_max).any()): # propose new if outside boundary conditions
            posnew = pos + randgen.integers(low=0,high=3,size=2)-1
    
        uposnew = u[posnew[0], posnew[1]]
        upos = u[pos[0],pos[1]]
        
    else:
        
        posnew = pos + 2*delta*randgen.uniform(size=2)-delta
    
        while((posnew < pos_min).any() | (posnew > pos_max).any()): # propose new if outside boundary conditions
            posnew = pos + 2*delta*randgen.uniform()-delta
        
        pts = np.array([[posnew[0],posnew[1]],[pos[0],pos[1]]])
        uposnew, upos = interp_obj(pts) if int_type == 'lin' else interp_obj.ev(pts[:,0],pts[:,1])
    
    if (uposnew < upos):
        h = uposnew/upos
        if (randgen.uniform() > h):
            prob = 0
            return pos, prob
    prob = 1
    
    return posnew, prob


def msd_straight_forward(r, root=False):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)    

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        
    if root:
        msds = np.sqrt(msds)
        
    return msds

def save_object(path_res, obj_name, obj):

    with open(path_res+obj_name+'.pkl', 'wb') as file:
        pickle.dump(obj, file)
        
def myExpFunc(x, a, b):
    
    # raises 10 to the power of the fitted intercept -  this fixes the
    # unalignment in the log scale
    
    return (10**a) * np.power(x, b)

def fit_exponent(x, y):
    
    logy = np.log10(y)
    logx = np.array([np.log10(i) for i in x]) 
    
    res = np.polyfit(logx, logy, 1, full=True)
    exp, intercept, err_exp = res[0][0], res[0][1], res[1][0]
    
    fitted = myExpFunc(x, intercept, exp)
    
    return fitted, exp, err_exp

def read_obj(obj_path):
    
    with open(obj_path+'.pkl', 'rb') as file:
        
        obj = pickle.load(file)
        
    return obj


#  --- RUN




path_figs = 'C:/nikyk/.Mallorca - UIB/Stochastic Simulation Methods/Indiv Project/figures/'
seed = 123456

# 1 - run discrete

M = 50*(10**3)
pos0 = np.array([60,85])

# the env
path_env = 'C:/nikyk/.Mallorca - UIB/Stochastic Simulation Methods/Indiv Project/env/'
env_name = 'food-res_patch-noise2to1'
env_alias = 'noise2to1'
fname = path_env + env_name+'.pkl'
with open(fname, 'rb') as file: 
    u = pkl.load(file)

# rand gen
ran_gen = np.random.default_rng(seed)

delta = 1
int_n='none'

trajs = metropolis_MC(M, pos0, u, delta, ran_gen)

h = np.histogram2d(trajs[:,0], trajs[:,1], bins=u.shape[0])
plt.imshow(h[0],  interpolation='none')
plt.axis('off')
name_fig = 'delta={:.2f}, interp={}, env={}'.format(delta, int_n, env_alias)
plt.title(name_fig)
plt.savefig(path_figs+name_fig+'.png',format='png',dpi=1200)
plt.show()


# 2 - run continuous (with interpolation)


M = 50*(10**4)
pos0 = np.array([60,85])

# the env
path_env = 'C:/nikyk/.Mallorca - UIB/Stochastic Simulation Methods/Indiv Project/env/'
env_name = 'food-res_patch-noise2to1'
env_alias = 'noise2to1'
fname = path_env + env_name+'.pkl'
with open(fname, 'rb') as file: 
    u = pkl.load(file)
    
# create interpolation object
x = y = np.linspace(0, u.shape[0]-1, u.shape[0])
interp = RectBivariateSpline(x, y, z=u)

int_n = 'spline'

for delta in np.linspace(1.1,2,10):
    
    ran_gen = np.random.default_rng(seed) # init every loop
    trajs = metropolis_MC(M, pos0, u, delta, ran_gen, int_n, interp)
    
    h = np.histogram2d(trajs[:,0], trajs[:,1], bins=u.shape[0])
    plt.imshow(h[0], interpolation='none')
    plt.axis('off')
    name_fig = 'delta={:.2f}, interp={}, env={}'.format(delta, int_n, env_alias)
    plt.title(name_fig)
    # plt.savefig(path_figs+name_fig+'.png',format='png',dpi=1200)
    plt.show()
    





# -------- RMSD

# N trajectories for 600 time steps
N = 100 #100

path_figs = 'C:/nikyk/.Mallorca - UIB/Stochastic Simulation Methods/Indiv Project/figures-2/'
seed = 123456
# rand gen
ran_gen = np.random.default_rng(seed)

pos0 = None #np.array([50,50]) # maybe change to random?

# the env
path_env = 'C:/nikyk/.Mallorca - UIB/Stochastic Simulation Methods/Indiv Project/env/'
env_name = 'food-res_patch-noise2to1'
env_alias = 'noise2to1'
fname = path_env + env_name+ '.pkl'
with open(fname, 'rb') as file: 
    u = pkl.load(file)

# create interpolation object
x = y = np.linspace(0, u.shape[0]-1, u.shape[0])
interp = RectBivariateSpline(x, y, z=u)

int_n = 'spline'


# deltas = np.linspace(1, 10, 4, dtype='int')
deltas = np.linspace(0.1, 1, 4)
delta_M = {}
for d in [0.1,0.4,0.7,1,4,7,10]:
    delta_M[str(d)] = int(600/d)

exps = np.zeros((len(deltas), 2)) # exponent and its error
colours = list(mcolors.TABLEAU_COLORS)

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')

ls = (0, (5, 10)) # loosley dashed

# ------------ LOOP
for i, delta in enumerate(deltas):
    
    print(delta)
    if i == 0:
        max_M = 6000 if deltas.dtype != 'int' else 600
    
    samples, samples_std, rmsds = metropolis_MC_rmsd(N, max_M, pos=pos0, u=u, delta=delta, randgen=ran_gen,
                                                      int_type = int_n, interp_obj = interp)
    
    obj_name = 'rmsds_env={}_N={}_M={}_delta={}'.format(env_alias, N, max_M, delta)
    
    save_object(path_figs, obj_name, rmsds) #save obj
    
    # plot rmsd samples
    y = rmsds[1:]
    x = np.linspace(start=1,stop=max_M-1,num=max_M-1, dtype=int)
    
    # plot the fit on the log of the samples
    M = delta_M[str(int(delta))] if delta>= 1 else delta_M[str(delta)]
        
    fitted, exp, err = fit_exponent(x[:M], y[:M]) # fit
    exps[i] = [exp, err]
    
    ax.plot(x, y, '.', color = colours[i], markersize=15, fillstyle='none',
            alpha = 0.7, label=r'$\delta={},\alpha={:.2f}\pm{:.2f}$'.format(delta, exp, err))
    ax.plot(x[:M], fitted, '--k')
    
plt.xlabel('time steps')
plt.ylabel('r.m.s. displacement')
plt.legend(loc=4)
fig_name = 'rms-disp-fit_deltas={}'.format(deltas)
plt.savefig(path_figs+fig_name+'.png',format='png',dpi=1000)
# plt.savefig(path_figs+fig_name+'-small-deltas-2.png',format='png',dpi=1000)
plt.show()






# --------------------------- entropy


# --- N, M and initial position

N = 100 # ants
M = 5*(10**4)
# M = 1000 #!!!!!! test
pos0 = None


# --- Environment

# substitute 0 probs with close to 0 val
u2 = np.where(u == 0, np.nextafter(np.float32(0), np.float32(1)), u)
u2 = u2/np.sum(u2)
ent_p = scipy.stats.entropy(u2.flatten())
#

# create interpolation object
x = y = np.linspace(0, u.shape[0]-1, u.shape[0])
interp = RectBivariateSpline(x, y, z=u)
int_n = 'spline'

# --- Deltas
deltas = np.linspace(1,10,10,dtype='int')

# --- Run entropies and save

for i,delta in enumerate(deltas):
    
    samples, entropy, _ = metropolis_MC_entropy(M=M, pos=pos0, u=u2, ent_p=ent_p,
                                                delta=delta, randgen=ran_gen,
                                                int_type=int_n, interp_obj=interp
                                              )
    
    # save samples and entropies so I could plot dists later too!
    for name, obj in zip(['samples', 'entropy'],[samples, entropy]):
        obj_name = '{}_env={}_M={}_delta={}'.format(name, env_alias, M, delta)
        save_object(path_figs, obj_name, obj)


# -- Open res by res and plot

deltas = np.linspace(1,10,10,dtype='int')
fig, ax = plt.subplots()
for i,delta in enumerate(deltas):
    
    delta = int(delta)
    obj_name = 'entropy_env={}_M={}_delta={}'.format(env_alias, M, delta)
    data_path = path_figs+obj_name
    entropy = read_obj(data_path)
    
    ax.plot(entropy, color = colours[i], label = r'$\delta = {}$'.format(delta))
    if i == 0:
        plt.axhline(ent_p, linestyle = '--', color = 'k', lw = 1)

ax.set_xticklabels(labels = [0,1,2,3,4,5,6])
plt.xlim([0,M])
plt.xlabel('time step '+r'$(\times 10^{4})$')
plt.ylabel(r'$H(p,q)$')
plt.legend(loc=3, fontsize='x-small')
plt.grid(visible=True)
plt.tight_layout()
plt.savefig(path_figs + 'entropy_big_delta.png',format='png', dpi=1000)
plt.show()
    



# -------------------- plot distributions


deltas = np.hstack([np.linspace(0.1,0.9,9), np.linspace(1,10,10)])
fig, ax = plt.subplots()
for i,delta in enumerate(deltas):
    
    if delta >= 1:
        obj_name = 'samples_env={}_M={}_delta={}'.format(env_alias, M, int(delta))
        title = r'$\delta={}$'.format(int(delta))
    else:
        obj_name = 'samples_env={}_M={}_delta={:.1f}'.format(env_alias, M, delta) 
        title = r'$\delta={:.1f}$'.format(delta)
        
    data_path = path_figs+obj_name
    samples = read_obj(data_path)
    
    q = np.histogram2d(samples[:,0], samples[:,1], bins=u.shape[0])[0]
    plt.imshow(q)
    plt.axis('off')
    plt.title(title)
    plt.savefig(path_figs+'samp-dist'+obj_name+'.png',format='png',dpi=1000)
    plt.show()
    


