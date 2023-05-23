import numpy as np

class Track():
    def __init__(self):
        self.lattice = []
        self.normalized_coordinates = None
        self.sequence = []
    

    def add_drift(self, length):
        element = np.array([[1, length], [0, 1]])
        self.lattice.append(element)

    def add_quadrupole(self, length, strength, plane):
        l = length 
        if plane == 'y':
            k_quad = -strength
        else:
            k_quad = strength
        
        if k_quad>0:
            k = np.sqrt(k_quad)
            element = np.array([[np.cos(k*l), (1/k)*np.sin(k*l)],
                [(-k)*np.sin(k*l), np.cos(k*l)]])
        elif k_quad<0:
            k = np.sqrt(abs(k_quad))
            element = np.array([[np.cosh(k*l), (1/k)*np.sinh(k*l)],
                [(k)*np.sinh(k*l), np.cosh(k*l)]]) 
        else:
            element = np.array([[1, l], [0, 1]]) 
        self.lattice.append(element)

# def add_dipole(self, length, bending_angle):
#     L = length
#     theta = bending_angle
#     element = np.array([[np.cos(theta), L * np.sin(theta)], [-np.sin(theta) / L, np.cos(theta)]])
#     self.lattice.append(element)
        
    def create_sequence(self):
        sequence = np.eye(2)
        for element in self.lattice:
            sequence = element @ sequence
        self.sequence = sequence
        return sequence

# def norm_beam(self):
#     np.dot(M,V)
#     self.norm_sequence = 
    
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
            print(k)
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