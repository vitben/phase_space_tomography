import numpy as np
from skimage.transform import radon, iradon, iradon_sart
import utils
from scipy.optimize import curve_fit
from track import Lattice
from preprocess import Preprocess
import tensorflow as tf
from neural_net import FeedBack

class Reconstruct():
    def __init__(self,
                 process):
        self.process = process
        self.projections = process.projections.T
        self.thetas = process.thetas
        self.ks = process.ks



    def MLEM(self, iterations):
            """
            The MLEM function takes in the number of iterations and returns a reconstructed image.
            The MLEM function uses the Radon transform to project an image onto a sinogram, then backproject it using 
            the inverse Radon transform. The ratio between the original projection and this new projection is used to 
            correct for errors in each iteration.
            
            :param self: Bind the method to the object
            :param iterations: Determine how many times the algorithm will run
            :return: A 2d array
            :doc-author: Trelent
            """
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
        """
        The SART function takes the projections and angles of a given object,
        and returns an image of that object. The function uses the iradon_sart
        method from scikit-image to reconstruct the image.
        
        :param self: Bind the method to an object
        :param iterations: Determine how many times the sart algorithm will run
        :return: The reconstructed image
        :doc-author: Trelent
        """
        xs = np.zeros((len(self.projections),len(self.projections)))
        for p in range(iterations):
            xs = iradon_sart(self.projections, theta = -self.thetas, image = xs)
        xs[xs<0] = 0
        return xs
    
    def FBP(self):
        """
        The FBP function takes the projections and thetas from a given object,
        and uses them to reconstruct an image using filtered backprojection.
        
        
        :param self: Bind the method to an object
        :return: The reconstructed image
        :doc-author: Trelent
        """
        xs = iradon(self.projections, theta = -self.thetas)
        xs[xs<0]=0
        return xs
    


    def fit_profiles(self):
        """
        The fit_profiles function fits a Gaussian profile to the unscaled projections of the process.
        The function returns an array of sigmas, which are used in the calculation of kurtosis.
        
        :param self: Refer to the instance of a class
        :return: The sigmas array, which is the standard deviation of each projection
        :doc-author: Trelent
        """
        sigma= []
        for k in range(len(self.ks)):
            X = self.process.x_new
            Y = self.process.unscaled_projections[k,:]
            coeffs,_ = utils.gaussian_profile_fit(X, Y)
            sigma.append(coeffs[2])
        self.sigmas  = np.array(sigma)


    def quad_rec_matrix(self):
        """
        The quad_rec_matrix function creates a matrix of the form:
        
        :param self: Bind the method to an object
        :return: The pseudo-inverse of the matrix a
        :doc-author: Trelent
        """
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
        """
        The QuadScanRec function takes the sigmas from a quad scan and returns the reconstructed epsilon, alpha, beta, gamma values.
        
        :param self: Bind the method to an object
        :param sigmas: Calculate the inverse of the a matrix
        :return: The epsilon, alpha, beta and gamma parameters of the quadrupole
        :doc-author: Trelent
        """
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
 

