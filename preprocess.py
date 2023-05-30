# Preprocessing of raw sinograms

import numpy as np
# import track
import utils
from track import Model


class Preprocess(Model):
    def __init__(self,
                 raw_sinogram,
                 ks_quad,
                 sequence,
                 norm = [0.4, 0.6],
                 plane = 'x'
                 ):
      
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
        ref = a.copy()
        shift = np.roll(b, np.round(ks[0]))
    
        goal = np.sum(abs(ref-shift))
        return goal

    def align_sinogram(self):
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
        if (range_cut[0]<0) or (range_cut[1]>self.projections.shape[1]):
            pass
        else:
            self.projections = self.projections[:,range_cut[0]: range_cut[1]]
    
    def center(self, roi):
        
        
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
        var_r = 0
        for c in range(self.projections.shape[0]):
            bins = np.arange(self.projections.shape[1])*self.pixel_size-np.mean(np.arange(self.projections.shape[1])*self.pixel_size)
            var = utils.var_hist(bins, self.projections[c,:])
            if var>var_r:
                var_r=var
        return int(var/self.pixel_size)

    def get_angles(self):
        thetas = []
        for k in self.ks_quad:
            th = utils.get_theta_norm(self.Vi, k, self.plane)
            thetas.append(th)
        # Get scaling
        self.thetas = np.array(thetas)
        
        pass

    def get_scalings(self):
        scaling = []
        for k in self.ks_quad:
        # Get scaling
            sc = utils.get_scaling_norm(self.Vi, k, self.plane)
            scaling.append(sc)
        self.scaling = np.array(scaling)

        pass

    def apply_scaling(self):
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

    def interpolate_sinogram(self):
        pass
    
