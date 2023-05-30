import numpy as np
from skimage.transform import radon, iradon, iradon_sart
import utils
from scipy.optimize import curve_fit
from track import Lattice

class Reconstruct():
    def __init__(self,
                 process):
        self.process = process
        self.projections = process.projections.T
        self.thetas = process.thetas
        self.ks = process.ks



    def MLEM(self, iterations):
            n_iters=iterations
            mlem_rec = np.ones((self.projections.shape[0], self.projections.shape[0]))
            sino_ones = np.ones(self.projections.shape)
            sens_image = iradon(sino_ones, theta=-self.thetas, circle=True, filter_name = None)

            for p in range(n_iters):

                fp = radon(mlem_rec, -self.thetas, circle=True)
                ratio = self.projections/(fp+0.000001)
                correction = iradon(ratio, -self.thetas, circle=True, filter_name=None)/(sens_image+0.000001)
                mlem_rec = mlem_rec*correction
            xs = mlem_rec
            return xs
    
    def SART(self, iterations):
        xs = np.zeros((len(self.projections),len(self.projections)))
        for p in range(iterations):
            xs = iradon_sart(self.projections, theta = -self.thetas, image = xs)
        return xs
    
    def FBP(self):
        xs = iradon(self.projections, theta = -self.thetas)
        return xs
    


    def fit_profiles(self):
        sigma= []
        for k in range(len(self.ks)):
            X = self.process.x_new
            Y = self.process.unscaled_projections[k,:]
            coeffs,_ = utils.gaussian_profile_fit(X, Y)
            sigma.append(coeffs[2])
        self.sigmas  = np.array(sigma)


    def quad_rec_matrix(self):
        A = []
        d = self.process.sequence['drift'][0]
        l = self.process.sequence['quad'][0]
        for k in self.ks:
            seq = {'quad':[l,k], 'drift':[d]}
            lattice = Lattice()
            M = lattice.create_lattice(seq)
            if len(A)==0:
                A = np.array([M[0,0]**2, +2*M[0,0]*M[0,1], M[0,1]**2])
            else:
                A = np.vstack((A, np.array([M[0,0]**2, +2*M[0,0]*M[0,1], M[0,1]**2])))
        self.A_i = np.linalg.pinv(A)


    def QuadScanRec(self, sigmas = None):
        self.fit_profiles()
        if sigmas is None:
            sigmas = self.sigmas
        self.quad_rec_matrix()
        s11i, s12i, s22i = np.dot(self.A_i,sigmas.T**2)
        eps = np.sqrt(abs(s11i*s22i-s12i**2))
        alf = -s12i/eps
        bet = s11i/eps
        gam = s22i/eps
        return eps, alf, bet, gam
        


    


    #     pass

