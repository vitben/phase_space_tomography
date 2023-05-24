
import matplotlib.pyplot as plt

class Plot():

    def __init__(self, processx, processy, recx=None, recy=None):
        self.processx = processx
        self.processy = processy
        self.recx = recx 
        self.recy = recy

   
    def plot_sino(self):
        fig, axs = plt.subplots(1,2, figsize = (7, 8))
        axs[0].imshow(self.processx.unscaled_projections, 
                      extent=[self.processx.x_new[0], self.processx.x_new[-1], self.processx.thetas[0], self.processx.thetas[-1]],
                      aspect = 'auto',
                      origin = 'lower')
        axs[0].set_title('x plane')
        axs[0].set_xlabel('r (mmm)')
        axs[0].set_ylabel('$\\theta$ (deg)')
        axs[1].imshow(self.processy.unscaled_projections, 
                      extent=[self.processy.x_new[0], self.processy.x_new[-1], self.processy.thetas[0], self.processy.thetas[-1]],
                      aspect = 'auto',
                      origin = 'lower')
        axs[1].set_xlabel('r (mmm)')
        axs[1].set_ylabel('$\\theta$ (deg)')
        axs[1].set_title('y plane')
        plt.tight_layout()
        plt.show()
        return fig, axs

    def plot_sino_scale(self):
        try:
            fig, axs = plt.subplots(1,2, figsize = (7, 8))
            axs[0].imshow(self.processx.projections, 
                        extent=[self.processx.x_new_sc[0], self.processx.x_new_sc[-1], self.processx.thetas[0], self.processx.thetas[-1]],
                        aspect = 'auto',
                        origin = 'lower')
            axs[0].set_title('x plane')
            axs[0].set_xlabel('r (mmm)')
            axs[0].set_ylabel('$\\theta$ (deg)')
            axs[1].imshow(self.processy.projections, 
                        extent=[self.processy.x_new_sc[0], self.processy.x_new_sc[-1], self.processy.thetas[0], self.processy.thetas[-1]],
                        aspect = 'auto',
                        origin = 'lower')
            axs[1].set_title('y plane')
            axs[1].set_xlabel('r (mmm)')
            axs[1].set_ylabel('$\\theta$ (deg)')
        except:
            print('Scalded sinogram do not exist. Use "Preprocess.apply_scaling" function to generate scaled sinogram and produce the plot.')
        plt.tight_layout()
        plt.show()
        return fig, axs

   
    def plot_reconstructed(self):
        if (self.recx is None) or (self.recy is None):
            print('Input a reconstructed image.')
        else:
            fig, axs = plt.subplots(1,2, figsize = (7, 8))
            axs[0].imshow(self.recx, origin='lower', extent=(self.processx.x_new[0],self.processx.x_new[-1],self.processx.x_new[0],self.processx.x_new[-1]))
            axs[0].set_xlabel('x (mm)')
            axs[0].set_ylabel('x\' (mrad)')
            axs[0].set_title('x plane')
            axs[1].imshow(self.recy, origin='lower', extent=(self.processy.x_new[0],self.processy.x_new[-1],self.processy.x_new[0],self.processy.x_new[-1]))
            axs[1].set_xlabel('y (mm)')
            axs[1].set_ylabel('y\' (mrad)')
            axs[1].set_title('y plane')
        
        plt.tight_layout()
        plt.show()
        return fig, axs

    
    