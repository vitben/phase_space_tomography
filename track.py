import numpy as np
import utils

class Lattice():
    def __init__(self):
        self.lattice = []
        self.normalized_coordinates = None
    

    def add_drift(self, length):
        element = np.array([[1, length, 0, 0], 
                            [0, 1, 0, 0],
                            [0, 0, 1, length],
                            [0, 0, 0, 1]])
        self.lattice.append(element)

    def add_quadrupole(self, length, strength):
        l = length 
        k_quad = strength       
        if k_quad>0:
            k = np.sqrt(abs(k_quad))
            element = np.array([[np.cos(k*l), (1/k)*np.sin(k*l), 0, 0],
                [(-k)*np.sin(k*l), np.cos(k*l), 0, 0],
                [0, 0, np.cosh(k*l), (1/k)*np.sinh(k*l)],
                [0, 0 ,(k)*np.sinh(k*l), np.cosh(k*l)]])
        elif k_quad<0:
            k = np.sqrt(abs(k_quad))
            element = np.array([[np.cosh(k*l), (1/k)*np.sinh(k*l), 0, 0],
                [(k)*np.sinh(k*l), np.cosh(k*l), 0, 0],
                [0, 0, np.cos(k*l), (1/k)*np.sin(k*l)],
                [0, 0 ,(-k)*np.sin(k*l), np.cos(k*l)]])
 
        else:
            element = np.array([[1, length, 0, 0], 
                            [0, 1, 0, 0],
                            [0, 0, 1, length],
                            [0, 0, 0, 1]])
        self.lattice.append(element)
        
    def create_transport(self):
        transport = np.eye(4)
        for element in self.lattice:
            transport = element @ transport
        self.transport = transport
        return transport
    
    def create_lattice(self, sequence):
        for key in sequence.keys():
            value = utils.get_iterable(sequence[key])
            if key == 'quad':
                self.add_quadrupole(value[0], value[1])
            elif key == 'drift':
                self.add_drift(value[0])
        
        return self.create_transport()

# def norm_beam(self):
#     np.dot(M,V)
#     self.norm_sequence = 
class Model():
    def __init__(self,
                 sequence,
                 ks,
                 norm, 
                 plane,
                 ):
        
        self.sequence = sequence
        self.ks= ks
        self.norm = norm
        self.plane = plane
        if norm is None:
            self.V = np.array([[0,1],[1,0]])
            self.Vi = np.array([[0,1],[1,0]]).T

        else:
            self.V = utils.norm_matrix(norm[0], norm[1])
            self.Vi = utils.unnorm_matrix(norm[0], norm[1])
    
    def gen_transport(self):
        lattice = Lattice()
        if self.plane == 'x':
            self.M = lattice.create_lattice(self.sequence)[:2,:2]
        elif self.plane == 'y':
            self.M = lattice.create_lattice(self.sequence)[2:,2:]

    def get_theta_norm(self):
        """Calculate rotation angle range given quadrupoles strengths limits and normalization matrix"""
        M = np.dot(self.M,self.Vi)
        ph = np.arctan2(M[0,1], M[0,0])
        
        return ph*180/np.pi

    def get_scaling_norm(self):
        """Calculate rotation angle range given quadrupoles strengths limits and normalization matrix"""
        M = np.linalg.multi_dot([self.M,self.Vi])
        ph = np.arctan2(M[0,1], M[0,0])
        sc = np.sqrt(M[0,1]**2+ M[0,0]**2)
        return sc
    
    def get_tomo_params(self):
        self.scaling = []
        self.thetas = []
        for k in self.ks:
            self.sequence['quad'][1] = k
            self.gen_transport()
            self.scaling.append(self.get_scaling_norm())
            self.thetas.append(self.get_theta_norm())
        self.thetas = np.array(self.thetas)
        self.scaling = np.array(self.scaling)



class Track():
    def __init__(self,
                 sequence,
                 dist = None,
                 twiss=None):
        
    
        self.sequence = sequence
        self.dist = dist  
         
    
    def track(self):
        self.n_part = self.dist.shape[0]
        lattice = Lattice()
        M = lattice.create_lattice(self.sequence)
        self.dist_out = self.dist.copy()
        for p in range(self.n_part):
            self.dist_out[p,:] = np.dot(M, self.dist[p,:])


  

    # def track(X_true_n, thetas, plane, x_new, Vi, n_bins):
    #     projections = []
    #     for th in (thetas):
    #     # for k in ks:
            
    #         k = calc_k_norm(th, Vi)
    #         # ph = utils.get_theta_norm(Vi, k, 'x')
            
    #         # print(k)
    #         if plane == 'x':
    #             P =M_quad(k, 0.0708, plane)
    #         elif plane == 'y':
    #             P =M_quad(-k, 0.0708, plane)
    #         # M = utils.apply(Vi,P) 
    #         M=P
    #         # print(ph)
    #         sc = np.sqrt(M[0,0]**2+M[0,1]**2)
    #         # 
    #         print(k)
    #         X_meas= np.zeros(np.shape(X_true_n))
    #         X_meas[:,0] = X_true_n[:,0]*M[0,0]+X_true_n[:,1]*M[0,1]
    #         X_meas[:,1] = X_true_n[:,0]*M[1,0]+X_true_n[:,1]*M[1,1]
    #         X_meas[:,0] = X_meas[:,0]
    #         projection, edges = np.histogram(X_meas[:, 0], range=(x_new[0], x_new[-1]), bins=n_bins)
    #         projection = moving_average(projection, 1)
    #         projection = projection/np.sum(projection)
    #         # projection[(projection<0.8*roi) | (projection>0.8*roi)] == 0
    #         projections.append(projection)

    #     projections = np.array(projections).T
    #     projections = projections/np.max(projections)
    #     return projections, edges