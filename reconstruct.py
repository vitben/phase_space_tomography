import numpy as np
from skimage.transform import radon, iradon, iradon_sart

class Reconstruct():
    def __init__(self,
                 process):
        self.projections = process.projections.T
        self.thetas = process.thetas


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

