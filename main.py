import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.stattools import acf as autocorr
import arviz
#multivariate_normal.pdf(x, mean=2.5, cov=0.5)

rng = np.random.default_rng()
def logp(a, b, cov, target_f):#proposals
    #TODO mogoče treba zamenjat a in b
    #test = -n * log(x[2]) - (sum_y2 - 2 * nybar * x[1] + n * x[1] ** 2) / (2 * x[2] ** 2) - log(1 + x[2] ** 2)
    l1 = np.log(target_f(a))
    l2 = np.log(multivariate_normal.pdf(b, mean=np.zeros(b.shape[0]), cov=cov, allow_singular=True))
    return l1 + l2
def metropolis_hastings(start, func, steps=1000, burnin=True):
    size = start.shape[0]
    S = np.eye(size)
    current_theta = start.copy()
    B = 10
    samples = np.zeros((steps - (steps%B), size))
    a_rate = np.zeros(B)
    M = steps // B
    i = 0
    if burnin:
        for b in range(B):
            for m in range(M):
                i = b * M + m
                cov = (2.4**2)*S/size
                proposed_theta = rng.multivariate_normal(mean=np.zeros(size), cov=cov)
                #cov = (2.4**2)*np.eye(size)/size
                log_r = logp(proposed_theta, current_theta, cov, func) - logp(current_theta, proposed_theta, cov, func)
                if np.log(rng.random()) < log_r:
                    current_theta = proposed_theta
                samples[i, :] = current_theta

            a_rate[b] = len(np.unique(samples[:i, 0])) / len(samples[:i, 0])
            S = np.cov(samples[:i, :].T)
        print(a_rate)
    cov = (2.4 ** 2) * S / 2
    samples = np.zeros((steps, size))
    for i in range(steps):
        proposed_theta = rng.multivariate_normal(mean=np.zeros(size), cov=cov)
        log_r = logp(proposed_theta, current_theta, cov, func) - logp(current_theta, proposed_theta, cov, func)
        if np.log(rng.random()) < log_r:
            current_theta = proposed_theta
        samples[i, :] = current_theta
    #print(len(np.unique(samples[:,1]))/len(samples[:,1]) )
    #plt.hist(samples.squeeze(), bins=200)
    return samples

B = 0.05
def minus_logf(x):
    return -(-(x[0]**2)/200- 0.5 * (x[1]+ B * x[0]**2 - 100*B)**2)


def minus_logf_grad(x):
  g1 = -(x[0])/100- 1.0 * (2* B * x[0]) * (x[1]+ B * x[0]**2 - 100*B)
  g2 = - 1.0 * (x[1]+ B * x[0]**2 - 100*B)
  return np.array((-g1, -g2))


def HMC(U, grad_U, epsilon, L, current_q):
    q = current_q
    p = np.random.normal(size=len(q))  # independent standard normal variates
    current_p = p

    traj = []
    traj.append([p, q, U(q) + sum(p ** 2) / 2])

    # Make a half step for momentum at the beginning
    p = p - epsilon * grad_U(q) / 2
    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q=q+epsilon * p

        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            p=p-epsilon * grad_U(q)
        traj.append([p, q, U(q)+sum(p ** 2) / 2])

    # Make a half step for momentum at the end.
    p=p-epsilon * grad_U(q) / 2
    # Negate momentum at end of trajectory to make the proposal symmetric
    p=-p
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = sum(current_p ** 2) / 2
    proposed_U = U(q)
    proposed_K = sum(p ** 2) / 2
    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position

    if random.random() < np.exp(current_U-proposed_U+current_K-proposed_K):
        return q, traj # accept
    else:
        return current_q, traj  # reject


def hmc_f(f, grad, n):
    if n == 2:
        img = np.zeros((300, 500))
        for i in range(500):
            for j in range(300):
                img[299 - j, i] = np.exp(-f((i/10 - 25 ,j/10 - 20.0)))
    ## HMC
    L = 27
    epsilon = 0.1#0.6
    current_q = np.array((0, 0))
    m = 100
    samples = np.zeros((m, 2))
    #pdf(paste("trajectories-ex02.pdf", sep=""), width=9, height=3)
    for i in range(m):
        #print(i)
        res = HMC(f, grad, epsilon, L, current_q)
        samples[i, :] = np.array([res[0][0], res[0][1]])
        current_q = res[0]
        if (i > 10):
            print(m * arviz.ess(samples[:, :2].T) / i)  # monitor effective size of first 3 components

        # plot trajectory
        if i % 20 == 19:
            traj = res[1]
            #plt.figure(figsize=(15,10))
            if n == 2: plt.imshow(img, extent=[-25,25,-20,10])
            plt.plot([traj[0][1][0], traj[1][1][0]],[traj[0][1][1], traj[1][1][1]])
            for h in range(2, len(traj)):
                plt.plot([traj[h - 1][1][0], traj[h][1][0]],[traj[h- 1][1][1], traj[h][1][1]])
                #plt.scatter([h[0][0]], [h[0][1]])
                #print(h)
            #plt.ylim(-2, 2)
            #plt.xlim(-2, 2)
            #plt.imshow(img)
            #plt.savefig()
            plt.show()
            # g1 = ggplot(res$traj, aes(x=X1, y=X2))  + coord_cartesian(ylim=c(-2, 2), xlim=c(-2, 2))+ geom_point() +
            # geom_path() + theme_bw() + xlab("p1") + ylab("p2") +
            # geom_point(data=res$traj[1, ], colour = "red", aes(x=X1, y=X2))
            #
            # x < - seq(-25, 25, 0.2)
            # x0 < - expand.grid(x, x)
            # y < - apply(x0, 1, minus_logf)
            # df < - data.frame(x0, y = exp(-y))
            #
            #
            #
            # g2 = ggplot(res$traj, aes(x=X1.1, y=X2.1)) + geom_point() +
            # geom_path() + theme_bw() + xlab("q1")  + coord_cartesian(xlim=c(-25, 25), ylim=c(-20, 10)) + ylab("q2") +
            # geom_point(data=res$traj[1, ], colour = "red", aes(x=X1.1, y=X2.1)) +
            # geom_contour(data = df, mapping =  aes(Var1, Var2, z = y), alpha = 0.2, colour="black")
            #
            # g3 = ggplot(res$traj, aes(x=1:nrow(res$traj), y = H)) + geom_point() +
            #                                                       geom_path() + theme_bw() + ylab("H") + xlab("step")
            # multiplot(g1, g2, g3, cols=3)


def gradf(f, x0):
    gs = np.zeros(x0.shape)
    eps = 1e-12
    for i in range(len(x0)):
        base = np.zeros(x0.shape)
        base[i] = eps
        fplus = f(x0 + base)
        fminus = f(x0 - base)
        gs[i] = ((fplus - fminus) / (2 * eps))
    return gs

def logi_fun(xs):
    ...

if __name__ == "__main__":
    #n = neff(np.array((1,2,3,1,2,3,1,2,3,1,2,3)))
    #h = arviz.ess(np.array((1,2,3,1,2,3,1,2,3,1,2,3)), method="identity")
    dataset = pd.read_csv("datset.csv").to_numpy()
    #print(dataset)
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(dataset[:, :2], dataset[:,-1])
    logistic_1 = lambda x: clf._predict_proba_lr(np.atleast_2d(x))[:,0]
    clf2 = LogisticRegression(fit_intercept=False)
    clf2.fit(dataset[:,:-1], dataset[:,-1])
    logistic_2 = lambda x: clf2.predict_proba(np.atleast_2d(x))[:,0]
    bivariate = lambda x:multivariate_normal.pdf(x, mean=np.zeros(2), cov=np.array([[2,-1],[-1,2]]))
    banana = lambda x: np.exp(-minus_logf(x))

    funt = logistic_1
    fun = lambda x: -np.log(funt(x))
    grad = lambda x:gradf(fun, x)

    hmc_f(fun, grad, 2)
    quit()
    samples = metropolis_hastings(np.ones(2), logistic_1)


    print(samples)
    plt.scatter(samples[:,0], samples[:,1])
    plt.show()

