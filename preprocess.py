# Preprocessing of raw sinograms

import numpy as np
# import track
import utils
from track import Model
from scipy.interpolate import RegularGridInterpolator


class Preprocess(Model):
    def __init__(self,
                 raw_sinogram,
                 ks_quad,
                 sequence,
                 norm = [0.4, 0.6],
                 plane = 'x'
                 ):
        """
        Preprocess class is used to trnslate the measured sinogrmas into inputs for tomographic recontruction (see reconstruction.py). 
        
        :param self: Refer to the current object
        :param raw_sinogram: Store the original sinogram
        :param ks_quad: Define the quadrupole strengths
        :param sequence: Defines the transport sequence
        :param norm: Normalization parameters alpha, beta 
        :param plane: Determine the phase space to be reconstructed
        :doc-author: Trelent
        """
      
        super().__init__(sequence,
                 ks_quad,
                 norm, 
                 plane,)
        
        self.projections_raw = raw_sinogram
        self.projections = raw_sinogram
        self.ks_quad = ks_quad
        self.plane = plane


        

        self.n_bins = 128
        self.pixel_size = 0.01
        self.get_tomo_params()
        self.align_sinogram()
        self.projections = self.center(self.projections)
        self.cent = self.projections.shape[1]//2
        self.range_var = 4*self.get_sino_range()
        self.cut([self.cent-self.range_var//2, self.cent+self.range_var//2])
        self.norm_image()
        
        self.apply_scaling()
    
    def func(self, ks, a, b):
        """
        The func function takes in three arguments:
            1. ks - a list of parameters to be optimized over
            2. a - the reference array (the one that will not move)
            3. b - the shifted array (the one that will move)
        
        :param self: Allow an instance of the class to access its own attributes and methods
        :param ks: Pass the value of k to the function
        :param a: Set the reference array
        :param b: Shift the array
        :return: The sum of the absolute difference between two arrays
        :doc-author: Trelent
        """
        ref = a.copy()
        shift = np.roll(b, np.round(ks[0]))
    
        goal = np.sum(abs(ref-shift))
        return goal

    def align_sinogram(self):
        """
        The align_sinogram function aligns the sinogram by shifting each projection in the sinogram
        by a certain amount. The shift is determined by minimizing the difference between two consecutive
        projections. This function is called after every iteration of reconstruction to ensure that 
        the projections are aligned.
        
        :param self: Refer to the object itself
        :return: The projections_aligned array
        :doc-author: Trelent
        """
        projections_aligned = self.projections.copy()
        for c in range(projections_aligned.shape[0]-1):

            
            pl = []
            for p in range(-50, 50):
                pl.append(self.func(np.array([p]),projections_aligned[c,:], projections_aligned[c+1,:]))
            # plt.plot(range(-50,50),pl)
            ind_min = np.argmin(np.array(pl))
            projections_aligned[c+1,:] = np.roll(projections_aligned[c+1], ind_min-50)
        self.projections = projections_aligned

    def cut(self, range_cut):
        """
        The cut function takes a range of pixels and cuts the projections to that size.
            The function will not cut if the range is outside of the projection's pixel bounds.
        
        :param self: Represent the instance of the class
        :param range_cut: Specify the range of pixels to be cut
        :return: The projections between the range_cut[0] and range_cut[-]
        :doc-author: Trelent
        """
        if (range_cut[0]<0) or (range_cut[1]>self.projections.shape[1]):
            pass
        else:
            self.projections = self.projections[:,range_cut[0]: range_cut[1]]
    
    def center(self, roi):
        """
        The center function takes a 2D array and shifts the maximum value to the center of the array.
        
        :param self: Represent the instance of the class
        :param roi: Pass in the array that is to be centered
        :return: The shifted array
        :doc-author: Trelent
        """
        
        
        max_index = np.argmax(roi)

        # find the center of the array
        center_x, center_y = roi.shape[1]//2, roi.shape[0]//2

        # calculate the shift needed to center the maximum value
        shift_x= center_x - max_index % roi.shape[1]

        # shift the array
        roi_shifted = np.roll(roi, shift_x, axis=1)
        # find the index of the maximum value
        return roi_shifted



    def get_sino_range(self):
        """
        The get_sino_range function is used to determine the range of the sinogram.
        The function takes in a sinogram and returns an integer value that represents 
        the range of the sinogram. The function does this by looping through each column 
        of the sinogram, calculating its variance, and then comparing it to a variable called var_r. 
        If var_r is less than or equal to 0, then it will be set equal to var (which is calculated from utils). If not, then nothing happens.
        
        :param self: Access the class attributes
        :return: The variance of the histogram of each projection
        :doc-author: Trelent
        """
        var_r = 0
        for c in range(self.projections.shape[0]):
            bins = np.arange(self.projections.shape[1])*self.pixel_size-np.mean(np.arange(self.projections.shape[1])*self.pixel_size)
            var = utils.var_hist(bins, self.projections[c,:])
            if var>var_r:
                var_r=var
        return int(var/self.pixel_size)

    def get_angles(self):
        """
        The get_angles function takes the k-space coordinates of a quadrant and returns the angles between each point in that quadrant and the center of k-space.
        
        :param self: Represent the instance of the class
        :return: The angles of the k vectors in the plane
        :doc-author: Trelent
        """
        thetas = []
        for k in self.ks_quad:
            th = utils.get_theta_norm(self.Vi, k, self.plane)
            thetas.append(th)
        # Get scaling
        self.thetas = np.array(thetas)
        
        

    def get_scalings(self):
        """
        The get_scalings function calculates the scaling of each k-point in the Brillouin zone.
        The scaling is calculated by taking the norm of a vector, which is defined as:
            V = (V_x, V_y) = (kx * Vi[0], ky * Vi[0])
        where: 
            - kx and ky are components of a given wavevector in reciprocal space. 
            - Vi is an array containing lattice vectors for a given plane. The first element corresponds to x-direction and second to y-direction.
        
        :param self: Bind the method to an object
        :return: The scaling factor for each k-point
        :doc-author: Trelent
        """
        scaling = []
        for k in self.ks_quad:
        # Get scaling
            sc = utils.get_scaling_norm(self.Vi, k, self.plane)
            scaling.append(sc)
        self.scaling = np.array(scaling)

        

    def apply_scaling(self):
        """
        The apply_scaling function takes the projections and applies the scaling to them.
        It does this by interpolating each projection onto a new x-axis, which is defined as 
        the original x-axis multiplied by the scaling factor for that particular projection. 
        The function then returns two arrays: one with unscaled projections and one with scaled ones.
        
        :param self: Refer to the object itself
        :return: The unscaled_projections and the projections
        :doc-author: Trelent
        """
        projs_cut = self.projections.copy()
        projs_new = []
        projs_new_sc = []
        for p in range(projs_cut.shape[0]):
            x_old = np.arange(len(projs_cut[p,:]))*self.pixel_size - np.mean(np.arange(len(projs_cut[p,:]))*self.pixel_size)

            self.x_new = np.linspace(x_old.min(), x_old.max(), self.n_bins)
            projs_new.append(np.interp(self.x_new, x_old, projs_cut[p,:]))
            x_sc = np.arange(len(projs_cut[p,:]))*self.pixel_size*self.scaling[p]- np.mean(np.arange(len(projs_cut[p,:]))*self.pixel_size*self.scaling[p])
            
            self.x_new_sc = np.linspace(x_sc.min(), x_sc.max(), self.n_bins) 
            proj_n_sc = np.interp(self.x_new_sc, x_old, projs_cut[p,:], left = 0, right = 0)
            proj_n_sc = proj_n_sc*self.scaling[p]
            projs_new_sc.append(proj_n_sc/np.sum(proj_n_sc))
            
        projs_new = np.array(projs_new)
        self.unscaled_projections = projs_new
        projs_new_sc = self.center(np.array(projs_new_sc))
        self.projections = projs_new_sc
        
    def norm_image(self):
        self.projections = self.projections/np.max(self.projections)

    def reset_projections(self):
        self.projections = self.projections_raw



 
class nn_predict(Preprocess):
    def __init__(self,
                 projectionsx, 
                 ksx, 
                 sequence, 
                 plane = 'x', 
                 norm = [0.7, 1.5]):
        """
        nn_predict is used to interpolate the scaled sinogram and use autoregressive LSTM model (FeedBack)
        to predict the missing profiles.

        :param self: Refer to the object itself
        :param projectionsx: measured projections
        :param ksx: Quadrupole strengths
        :param sequence: Determine the number of projections
        :param plane: Select the plane of the projection
        :param norm: Normalize the data
        :param 1.5]: Normalize the data
        :return: The following:
        :doc-author: Trelent
        """

        self.ang_step = 5
        self.n_bins=65
        self.process = Preprocess(projectionsx, 
                ksx, 
                sequence, 
                plane=plane, 
                norm=norm)
        self.process.x_new = np.linspace(self.process.x_new.min(),self.process.x_new.max(),self.n_bins)
   


    def interp_sinogram(self):
        """
        The interp_sinogram function interpolates the sinogram to a new grid.
        The function takes in the following parameters:
            - self: The class instance of TomoPyReconstructor.
            
        The function performs the following steps:
            1) Create meshgrids for x and y values from process object's x_new_sc and 
               thetas attributes, respectively. Store these as variables named xg and yg, 
               respectively. Also store data from process object's projections attribute as variable named data.
        
        :param self: Access the attributes of the class
        :return: The projections and thetas
        :doc-author: Trelent
        """
        x, y = self.process.x_new_sc, self.process.thetas
        xg, yg = np.meshgrid(x, y)
        data = self.process.projections.T
        interp = RegularGridInterpolator((x, y), data,
                                        bounds_error=False, fill_value=None)
        x_n, y_n = np.linspace(x.min(), x.max(), self.n_bins), np.arange(y.min(), y.max(), self.ang_step)
        X, Y = np.meshgrid(x_n, y_n, indexing='ij')
        self.process.projections = interp((X,Y)).T

        missing_angles_range = 180-(y.max()-y.min())
        missing_angles = int(np.floor(missing_angles_range/self.ang_step))
        self.missing_angles = missing_angles
        self.thetas_interp = y_n
        self.process.thetas = np.concatenate((y_n, np.array([y_n[-1]+n*self.ang_step for n in range(1,self.missing_angles+1)])))
        self.process.align_sinogram()
    
    
    def nn_fit(self):
        """
        The nn_fit function takes the sinogram and interpolates it to fill in missing angles.
        It then uses a trained LSTM model to predict the missing projections. 
        The predicted projections are added back into the original sinogram, which is then returned.
        
        :param self: Access the attributes and methods of the class
        :return: The concatenation of the original sinogram and the predicted sinogram
        :doc-author: Trelent
        """
        self.interp_sinogram()
        OUT_STEPS = int(self.missing_angles)
        projections_temp = utils.norm_image(self.process.projections).copy()
        feedback_model = FeedBack(units=128, out_steps=OUT_STEPS, num_features = self.n_bins)

        feedback_model.load_weights('./models/{}'.format('multi_ar_lstm'))
        example_inputs = tf.reshape(tf.constant(projections_temp, dtype=tf.float32), shape = (1, len(self.thetas_interp),self.n_bins))

        out_r = feedback_model(example_inputs)
        out = out_r.numpy()[0]
        out[out<0]=0
        self.process.projections = np.concatenate((projections_temp, out), axis=0)
    

    #     pass



        

    

    