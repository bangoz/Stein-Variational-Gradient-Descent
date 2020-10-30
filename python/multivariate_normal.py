import numpy as np
import numpy.matlib as nm
from time import time
import matplotlib.pyplot as plt

from svgd import SVGD

class MVN:
    """
    Multivariate Normal
    """
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)

class GMM:
    """
    Gaussian Mixture Model
    """
    def __init__(self, mu, A, gmprob):
        # mu: LxM with L peaks, M dims
        # A: LxMxM
        # gmprob: Lx1
        assert mu.shape[0]==A.shape[0]==gmprob.shape[0] and gmprob.sum()==1.0
        self.mu = mu
        self.A = A
        self.gmprob = gmprob
    
    def dlnprob(self, theta):
        dlnp1, dlnp2 = 0, 0
        for l in range(self.gmprob.shape[0]):
            tmp1 = theta-nm.repmat(self.mu[l], theta.shape[0], 1)
            tmp2 = np.matmul(tmp1, self.A[l])
            dlnp1 += -1*np.exp(-0.5*(tmp1*tmp2).sum(axis=1)).reshape((theta.shape[0],1))*tmp2*self.gmprob[l]
            dlnp2 += 1*np.exp(-0.5*(tmp1*tmp2).sum(axis=1))*self.gmprob[l]
        return dlnp1/dlnp2.reshape((theta.shape[0],1))


if __name__ == '__main__':
    # A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    # mu = np.array([-0.6871,0.8010])
    A = np.array([[[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]]])
    mu = np.array([[-5.0,-5.0], [5.0,5.0]])
    gmprob = np.array([0.2, 0.8])
    
    # model = MVN(mu, A)
    model = GMM(mu, A, gmprob)
    
    tik = time()
    # x0 = np.random.normal(0, 1, [1000, 2])
    x0 = np.random.uniform(-4.0, 4.0, [1000, 2])
    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01)
    tok = time()
    
    print("ground truth: ", mu)
    print("svgd: ", np.mean(theta,axis=0))
    print("time: ", tok-tik)

    plt.scatter(theta.T[0], theta.T[1], s=1)
    plt.show()
