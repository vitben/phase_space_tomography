import numpy as np
from scipy import interpolate
import track_help as hp
from scipy import stats
from skimage.transform import iradon, iradon_sart, radon
import matplotlib.pyplot as plt
import odl
import collections



def rotation_matrix(angle):
    """2x2 clockwise rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])

def mean_hist(bins_c, vals):
    vals = 1000*(vals/np.max(vals))
    np.average(bins_c, weights=vals)
    return np.sum(bins_c*vals)/np.sum(vals)

def var_hist(bins_c, vals):
    vals = 1000*(vals/np.max(vals))
    mean = mean_hist(bins_c, vals)
    return np.sqrt(np.average((bins_c - mean)**2, weights=vals))

def get_bin_centers(bin_edges):
    """Return bin centers from bin edges."""
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def get_bin_edges(bin_centers):
    """Return bin edges from bin centers."""
    w = 0.5 * np.diff(bin_centers)[0]
    return np.hstack([bin_centers - w, [bin_centers[-1] + w]])


def cov2corr(cov_mat):
    """Correlation matrix from covariance matrix."""
    D = np.sqrt(np.diag(cov_mat.diagonal()))
    Dinv = np.linalg.inv(D)
    corr_mat = np.linalg.multi_dot([Dinv, cov_mat, Dinv])
    return corr_mat


def symmetrize(M):
    """Return a symmetrized version of M.
    
    M : A square upper or lower triangular matrix.
    """
    return M + M.T - np.diag(M.diagonal())


def rand_rows(X, n):
    """Return n random elements of X."""
    Xsamp = np.copy(X)
    if n < len(X):
        idx = np.random.choice(Xsamp.shape[0], n, replace=False)
        Xsamp = Xsamp[idx]
    return Xsamp


def apply(M, X):
    """Apply M to each row of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def project(array, axis=0):
    """Project array onto one or more axes."""
    if type(axis) is int:
        axis = [axis]
    axis_sum = tuple([i for i in range(array.ndim) if i not in axis])
    proj = np.sum(array, axis=axis_sum)
    # Handle out of order projection. Right now it just handles 2D.
    if proj.ndim == 2 and axis[0] > axis[1]:
        proj = np.moveaxis(proj, 0, 1)
    return proj


def make_slice(n, axis=0, ind=0):
    """Return a slice index."""
    if type(axis) is int:
        axis = [axis]
    if type(ind) is int:
        ind = [ind]
    idx = n * [slice(None)]
    for k, i in zip(axis, ind):
        if i is None:
            continue
        idx[k] = slice(i[0], i[1]) if type(i) in [tuple, list, np.ndarray] else i
    return tuple(idx)


def hist(X, bins="auto"):
    edges = [np.histogram_bin_edges(X[:, i], bins) for i in range(X.shape[1])]
    return np.histogramdd(X, edges)

import pandas as pd


def repopulate_dist(X_true, n_part, threshold):
    """Generate point distribution from histogram"""
    X_t_hist, xedges, yedges = np.histogram2d(X_true[:,0], X_true[:,1], bins=(71, 71))
    X_t_hist[X_t_hist<threshold*np.max(X_t_hist)] = 0

    xg = xedges[:-1]+(xedges[1]-xedges[0])/2
    yg = yedges[:-1]+(yedges[1]-yedges[0])/2
    X_f = interpolate.interp2d(xg, yg, X_t_hist.T)
    pop_size = n_part
    X_ = X_f(xg,yg)*1000
    binx = (xg[1]-xg[0])/2
    biny = (yg[1]-yg[0])/2
    X, PX = np.meshgrid(xg, yg)
    X_out = []
    for xi, px in zip(X.flatten(), PX.flatten()):
        
        d = X_f(xi,px)[0]*100
        if len(X_out) == 0:
            X_out =  np.random.multivariate_normal(mean = (xi,px), cov = [[binx**2,0],[0,biny**2]], size=int(d))
        else:
            X_out = np.vstack((X_out, np.random.multivariate_normal(mean = (xi,px), cov = [[binx**2,0],[0,biny**2]], size=int(d))))
    return X_true

def repopulate_dist_fast(data, n_particles, n_bins, threshold):
    """Faster implementation of repopulate_dist"""
    # Calculate 2D histogram
    hist, x_edges, y_edges = np.histogram2d(data[:,0], data[:,1], bins=n_bins)
    hist[hist<threshold*np.max(hist)] = 0
    # Normalize histogram
    hist = (hist / hist.sum())

    n_points = n_particles
    # Generate new dataset from histogram
    new_data = []
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            num_points = int(hist[i,j] * n_points) # Generate 100 points in total
            x_coord = np.random.normal(x_edges[i], (x_edges[i+1]-x_edges[i])/2, num_points)
            y_coord = np.random.normal(y_edges[j], (y_edges[j+1]-y_edges[j])/2, num_points)
            new_data.append(np.column_stack((x_coord, y_coord)))
    new_data = np.concatenate(new_data)

    # new_data
    num_samples = n_points
    indices = np.random.choice(np.arange(new_data.shape[0]), size=num_samples, replace=True)
    final_data = new_data[indices]
    return final_data

def load_beam(filename, n_part, alpha, beta, eps, fast = True):
    """Open beam distribution file and apply scale and rotation to match the required Twiss paramters"""
    d = pd.read_csv(filename, header = None)
    d.drop(columns = [6, 7], inplace = True)
    d.columns = ['x', 'y', 't', 'px', 'py', 'pt']
    d['px'] = d['px']/19000000
    d['py'] = d['py']/19000000
    d['pt'] = d['pt']/19000000
    d['pt'] = d['pt']-np.mean(d['pt'])
    d['t'] = d['t']

    beam_m = d.to_dict('list')
    for k in beam_m.keys():
        beam_m[k] = np.array(beam_m[k])
    goal_twiss = {'alfx':alpha, 'betx':beta, 'alfy':alpha, 'bety':beta, 'epsx':eps, 'epsy':eps, 'dpp':0}
    n_part_s = 9999

    beam_out = pd.DataFrame(hp.shift_twiss(beam_m, goal_twiss))
    beam_out = pd.DataFrame(hp.shift_twiss(beam_out, goal_twiss))
    beam_out = pd.DataFrame(hp.shift_twiss(beam_out, goal_twiss))
    beam_out = pd.DataFrame(hp.shift_twiss(beam_out, goal_twiss))
    # beam_out = pd.DataFrame(beam_m)
    beam_filt = beam_out.sample(n_part_s)
    # pt = np.random.multivariate_normal((0,0), [[np.std(d['t'])**2,0],[0,env.dpp**2]], (n_part, n_part))
    # beam_filt['t'] = pt[:,0]
    # beam_filt['pt'] = pt[:,1]
    beam_filt = beam_filt.to_dict('list')
    for n in beam_filt.keys():
        beam_filt[n] = np.array(beam_filt[n])

    X_true = np.zeros((len(beam_filt['x']), 2))
    X_true[:,0] = beam_filt['x']
    X_true[:,1] = beam_filt['px']

    if n_part<10000:
        X_n = X_true[:n_part,:]
    else:
        X_n = np.zeros((n_part, 2))
        if fast:
            X_n = repopulate_dist_fast(X_true, n_part, threshold=0, n_bins=64)
        else:
            X_n = repopulate_dist(X_true, n_part, threshold=0)
    return X_n


def get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)
def M_quad(k_quad, l_quad,plane):
    """Transport matrix for quadrupole and drift"""
    s = 0.831693
    Md = np.array([[1, s],[0, 1]])
    l = l_quad
    if plane == 'y':
        k_quad = -k_quad
    
    if k_quad>0:
        k = np.sqrt(k_quad)
        Mq = np.array([[np.cos(k*l), (1/k)*np.sin(k*l)],
            [(-k)*np.sin(k*l), np.cos(k*l)]])
    elif k_quad<0:
        k = np.sqrt(abs(k_quad))
        Mq = np.array([[np.cosh(k*l), (1/k)*np.sinh(k*l)],
            [(k)*np.sinh(k*l), np.cosh(k*l)]]) 
    else:
        Mq = np.array([[1, l], [0, 1]]) 
    
    M = np.dot(Md,Mq)
    return M

def quad(k_quad, l_quad, plane):
    """Transport matrix for quadrupoles"""

    l = l_quad
    if plane == 'y':
        k_quad = -k_quad
    
    if k_quad>0:
        k = np.sqrt(abs(k_quad))
        M = np.array([[np.cos(k*l), (1/k)*np.sin(k*l)],
            [(-k)*np.sin(k*l), np.cos(k*l)]])
    elif k_quad<0:
        k = np.sqrt(abs(k_quad))
        M = np.array([[np.cosh(k*l), (1/k)*np.sinh(k*l)],
            [(k)*np.sinh(k*l), np.cosh(k*l)]]) 
    else:
        M = np.array([[1, l], [0, 1]]) 
    
    M
    return M

def drift(s):
    """Transport matrix for drift"""
    Md = np.array([[1, s],[0, 1]])
    return Md


def kde(X_true, nbins, limits_n):
    "Generate Kernel Density Estimation for a given 2D distribution"

    m1 = X_true[:,0]
    m2 = X_true[:,1]


    xmin = np.min(m1)
    xmax = np.max(m1)
    ymin = np.min(m2)
    ymax = np.max(m2)



    xmin = limits_n[0,0]
    xmax = limits_n[0,1]
    ymin = limits_n[1,0]
    ymax = limits_n[1,1]

    x_in = np.linspace(xmin, xmax, nbins)
    y_in = np.linspace(ymin, ymax, nbins)

    np.hstack((m1, limits_n[0]))
    np.hstack((m2, limits_n[0]))

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)

    X_in, Y_in = np.meshgrid(x_in, y_in)

    input_pos = np.vstack([X_in.T.ravel(), Y_in.T.ravel()])
    Z = np.reshape(kernel(input_pos), X_in.shape)
    stx = x_in[1]-x_in[0]
    sty = y_in[1]-y_in[0]
    x_in = np.linspace(xmin, xmax, nbins+1)
    y_in = np.linspace(ymin, ymax, nbins+1)

    return Z.T, np.vstack((x_in, y_in))

def gen_kde_dist(n_bins, X_true, roi, fit = 'kde'):
    """Generate KDE or histogram"""
    maxs_n = np.array([
        roi,
        roi,
    ])
    
    limits_n = np.array([(-m, m) for m in maxs_n])
    if fit == 'hist':
        f_true, edges_rec = np.histogramdd(X_true, n_bins, limits_n)
        f_true = f_true.T
    elif fit == 'kde':
        f_true, edges_rec = kde(X_true, n_bins, limits_n)

    grid_rec = [
        get_bin_centers(edges_rec[0]),
        get_bin_centers(edges_rec[1]),
    ]

    xmin = np.min(edges_rec[0])
    xmax = np.max(edges_rec[0])
    ymin = np.min(edges_rec[1])
    ymax = np.max(edges_rec[1])



    f_true = (f_true/np.sum(f_true))*1e3
    return edges_rec, f_true

def create_beam(filename, eps, alpha, beta, n_particles, norm_space, fast=True):
    """create particle beam and normalization matrix"""
    V = norm_matrix(alpha, beta)  # normalization matrix
    A = np.sqrt(np.diag([eps, eps]))  # scale by emittances

    X_true = load_beam(filename, n_particles, alpha, beta, eps, fast)

    V = norm_matrix(alpha, beta)  # normalization matrix
    A = np.sqrt(np.diag([eps, eps]))  # scale by emittances
    if norm_space:
        X_true = apply(np.linalg.inv(V), X_true)
    # X_true = utils.apply(V, X_true_n)
    
    return X_true, V

def mse_err(x, image):
    """Mean Squared Error"""
    x = x
    image = image
    return np.sum((x-image)**2)/len(x)

def norm_image(image):
    """Normalize image to max"""
    return image/np.max(image)


def calc_k(theta):
    """Calculate quadrupole strenght correponsding to a given rotation angle"""
    phase_advances_calc = []
    k_range = np.linspace(-100,100, 100)
    for k in k_range:
        M = M_quad(k, 0.0708, 'x')
        ph = np.arctan2(M[0,1], M[0,0])
        phase_advances_calc.append(np.degrees(ph))
    f = interpolate.interp1d(phase_advances_calc, k_range)
    return float(f(theta))

def calc_k_norm(theta, V):
    """Calculate quadrupole strenght correponsding to a given rotation angle and a normalized beam"""
    phase_advances_calc = []
    k_range = np.linspace(-100,100, 100)
    for k in k_range:
        M = M_quad(k, 0.0708, 'x')
        M = np.linalg.multi_dot([M,V])
        ph = np.arctan2(M[0,1], M[0,0])
        phase_advances_calc.append(np.degrees(ph))
    f = interpolate.interp1d(phase_advances_calc, k_range)
    return float(f(theta))

def get_theta_range(k_lims = [-40, 40]):
    """Calculate rotation angle range given quadrupoles strengths limits"""
    ph_lims = []
    for k in k_lims:
        M = M_quad(k, 0.0708, 'x')
        ph = np.arctan2(M[0,1], M[0,0])
        ph_lims.append(np.degrees(ph))
    return np.ceil(ph_lims[0]), np.floor(ph_lims[1])

def get_theta_range_norm(V, k_lims = [-40, 40]):
    """Calculate rotation angle range given quadrupoles strengths limits and normalization matrix"""
    ph_lims = []
    for k in k_lims:
        M = M_quad(k, 0.0708, 'x')
        M = np.linalg.multi_dot([M,V])
        ph = np.arctan2(M[0,1], M[0,0])
        ph_lims.append(np.degrees(ph))
    return np.ceil(ph_lims[0]), np.floor(ph_lims[1])

def get_theta_norm(V, k, plane):
    """Calculate rotation angle range given quadrupoles strengths limits and normalization matrix"""
    M = M_quad(k, 0.0708, plane)
    M = np.dot(M,V)
    ph = np.arctan2(M[0,1], M[0,0])
    
    return ph*180/np.pi

def get_scaling_norm(V, k, plane):
    """Calculate rotation angle range given quadrupoles strengths limits and normalization matrix"""
    M = M_quad(k, 0.0708, plane)
    M = np.linalg.multi_dot([M,V])
    ph = np.arctan2(M[0,1], M[0,0])
    sc = np.sqrt(M[0,1]**2+ M[0,0]**2)
    
    return sc

def calc_twiss_end(alpha, beta, M):
    """Calculates Twiss beta after applying a tranport matrix M"""
    gamma = (1+alpha**2)/beta
    beta_e =   beta*M[0,0]**2       -alpha*2*M[0,0]*M[0,1] +gamma*M[0,1]**2
    # alpha_e = -beta*M[0,0]*M[1,0]   +alpha*2*M[0,1]*M[1,0] +gamma*M[0,1]*M[1,1]
    return  beta_e

def moving_average(x, w):
    """Convolution function"""
    return np.convolve(x, np.ones(w), 'same') / w

def gen_beam_dset(alpha, beta, eps, kind, n_particles):
    """Generate a set of different beam distributions to create the dataset for NN training"""
    X_true, _ = create_beam('ref_beam.csv', eps, alpha, beta, n_particles, norm_space = False)

# 2. Gaussian
    if kind == 'gauss': 
        cov = eps*np.array([[beta, -alpha], [-alpha, (1+alpha**2)/beta]])
        gaussian_beam = np.random.multivariate_normal([0, 0], cov, size=n_particles)
        X_true = gaussian_beam
# 3. flipped
    elif kind == 'flip':
        sim_beam_flip = X_true.copy()
        sim_beam_flip[:,0] = -X_true[:,0]
        X_true = sim_beam_flip
# 4. cut_beam
    elif kind == 'cut':
        cut_beam = X_true.copy()
        cut_beam = repopulate_dist_fast(X_true, n_particles, 64, 0.1)
        X_true = cut_beam
# 5. flip cut beam
    elif kind == 'cut_flip':
        flip_cut_beam = X_true.copy()
        flip_cut_beam = repopulate_dist_fast(X_true, n_particles, 64, 0.1)
        flip_cut_beam[:,0] = -flip_cut_beam[:,0]
        X_true = flip_cut_beam
    elif kind == 'sim':
        X_true = X_true
    else:
        raise Exception('Please specify a valid kind of distribution')

    return X_true


def interp_sinogram(projections,meas_range,phase_advances):
    """Interpolate sinogram"""

    x = phase_advances
    y = meas_range
    z = projections
    f = interpolate.interp2d(x, y, z, kind='linear')

    steps_ph = int(np.max(phase_advances)-np.min(phase_advances))


    phase_advances_interp = np.linspace(min(phase_advances), max(phase_advances), steps_ph)
    meas_range_interp = y
    projections_interp = f(phase_advances_interp, meas_range_interp)
    projections_interp /= np.max(projections_interp)
    return projections_interp, phase_advances_interp

def evaluate_rec(f_true, edges_rec,  xs,plot = True):
    f_true_n = norm_image(f_true)
    f_rec_n = norm_image(xs)
    # Apply threshold
    f_rec_n[f_rec_n<0.005] = 0
    f_true_n[f_true_n<0.005] = 0
    # Calculate MSE
    mse_e = mse_err(f_true_n, f_rec_n)
    print("The Meas Squared Error between the two images is: {:.1e}".format(mse_e))
    if plot:
        plt.figure()
        mplt.plot_reconstr_diff(f_true_n, f_rec_n, edges_rec)
    return mse_e


def select_window(X, ind, sh):
    # Calculate the ending index of the window
    end_ind = ind + sh
    
    # Check if the ending index is greater than the shape of X along axis 0
    if end_ind > X.shape[0]:
        # Calculate the number of missing rows at the end of X
        missing_rows = end_ind - X.shape[0]
        
        # Slice the window from the end of X and from the beginning of X to fill the missing rows
        window = np.concatenate((X[ind:], X[:missing_rows]), axis=0)
    else:
        # Slice the window normally
        window = X[ind:end_ind]
    
    return window



def norm_matrix(alpha, beta):
    """2x2 normalization matrix for x-x' or y-y'."""
    return np.array([[1/np.sqrt(beta), 0], [alpha/np.sqrt(beta), np.sqrt(beta)]])

def unnorm_matrix(alpha, beta):
    """2x2 normalization matrix for x-x' or y-y'."""
    return np.linalg.inv(np.array([[1/np.sqrt(beta), 0], [alpha/np.sqrt(beta), np.sqrt(beta)]]))


def track(X_true_n, thetas, plane, x_new, Vi, n_bins):
    projections = []
    for th in (thetas):
    # for k in ks:
        
        k = calc_k_norm(th, Vi)
        # ph = utils.get_theta_norm(Vi, k, 'x')
        
        # print(k)
        if plane == 'x':
            P =M_quad(k, 0.0708, plane)
        elif plane == 'y':
            P =M_quad(-k, 0.0708, plane)
        # M = utils.apply(Vi,P) 
        M=P
        # print(ph)
        sc = np.sqrt(M[0,0]**2+M[0,1]**2)
        # 
      
        X_meas= np.zeros(np.shape(X_true_n))
        X_meas[:,0] = X_true_n[:,0]*M[0,0]+X_true_n[:,1]*M[0,1]
        X_meas[:,1] = X_true_n[:,0]*M[1,0]+X_true_n[:,1]*M[1,1]
        X_meas[:,0] = X_meas[:,0]
        projection, edges = np.histogram(X_meas[:, 0], range=(x_new[0], x_new[-1]), bins=n_bins)
        projection = moving_average(projection, 1)
        projection = projection/np.sum(projection)
        # projection[(projection<0.8*roi) | (projection>0.8*roi)] == 0
        projections.append(projection)

    projections = np.array(projections).T
    projections = projections/np.max(projections)
    return projections, edges

def gen_dist(hist, n_points, x):    
    n_points = 100000
    x_edges = x
    y_edges = x

    new_data = []
    for i in range(hist.shape[0]-1):
        for j in range(hist.shape[1]-1):
            num_points = int(hist[i,j] * n_points) # Generate 100 points in total
           
            y_coord = np.random.normal(x_edges[i], (x_edges[i+1]-x_edges[i])/2, num_points)
            x_coord = np.random.normal(y_edges[j], (y_edges[j+1]-y_edges[j])/2, num_points)
            new_data.append(np.column_stack((x_coord, y_coord)))
    new_data = np.concatenate(new_data)

    # new_data
    num_samples = n_points
    indices = np.random.choice(np.arange(new_data.shape[0]), size=num_samples, replace=True)
    final_data_x = new_data[indices]
    
    return final_data_x
