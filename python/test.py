# %%
import numpy as np
import numpy.matlib as nm
from time import time
import matplotlib.pyplot as plt

from svgd import SVGD
from multivariate_normal import GMM


def GMM_test(stepsize=0.01):
    A = np.array([[[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]]])
    mu = np.array([[-5.0,-5.0], [5.0,5.0]])
    gmprob = np.array([0.2, 0.8])
    
    model = GMM(mu, A, gmprob)
    
    tik = time()
    x0 = np.random.uniform(-4.0, 4.0, [1000, 2])
    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=stepsize)
    tok = time()
    
    print("ground truth: ", mu)
    print("svgd: ", np.mean(theta,axis=0))
    print("time: ", tok-tik)

    plt.scatter(theta.T[0], theta.T[1], s=1)
    plt.show()

# %%
if __name__ == "__main__":
    stepsizes = [0.01,0.05,0.1,0.5,1.0,2.0,4.0,8.0,16.0]
    for stepsize in stepsizes:
        print("### result for stepsize =", stepsize, "###")
        GMM_test(stepsize)
# %%
