import random
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.stattools import acf as autocorr
import arviz
#multivariate_normal.pdf(x, mean=2.5, cov=0.5)

rng = np.random.default_rng()
def logp(b, a, cov, target_f):#proposals
    #TODO mogoče treba zamenjat a in b
    #test = -n * log(x[2]) - (sum_y2 - 2 * nybar * x[1] + n * x[1] ** 2) / (2 * x[2] ** 2) - log(1 + x[2] ** 2)
    l1 = np.log(target_f(a))
    l2 = np.log(multivariate_normal.pdf(b, mean=np.zeros(b.shape[0]), cov=cov, allow_singular=True))
    return l1 + l2
def metropolis_hastings(start, func, steps=10000, burnin=False, cov=None, burn=10):
    dim = start.shape[0]
    def proposal(dimension, cov):
        return rng.multivariate_normal(mean=np.zeros(dimension), cov=cov)
    def proposal_prob(x, cov):
        return multivariate_normal.pdf(x, mean=np.zeros(x.shape[0]), cov=cov, allow_singular=True)
    def proposal_logp(x, cov):
        return -np.log(proposal_prob(x, cov))
    def get_samples(start, cov):
        current_theta = start
        samples = []
        for i in range(steps):
            proposed_theta = proposal(dim, cov)
            proposed_logp = proposal_logp(proposed_theta, cov)
            target_logp = np.log(func(proposed_theta))
            proposed0_logp = proposal_logp(current_theta, cov)
            target0_logp = np.log(func(current_theta))

            logratio = target_logp - proposed_logp - target0_logp + proposed0_logp

            if np.log(random.random()) < -logratio:
                current_theta = proposed_theta
            samples.append(current_theta)
        return np.array(samples)

    sigma = np.eye(dim)
    if cov is None:
        #TODO make this a parameter
        cov = (2.4 ** 2) * sigma / dim
    samples = np.zeros((burn * burn * (steps//burn),dim))
    print(cov)
    start_time = time.time()
    if burnin:
        for i in range(burn):
            ... #just do manual tuning... q.q
            cov = (2.4 ** 2) * sigma / dim
            print(cov)
            current_theta = start
            for j in range(burn):
                for k in range(steps//burn):
                    proposed_theta = proposal(dim, cov)
                    proposed_logp = proposal_logp(proposed_theta, cov)
                    target_logp = np.log(func(proposed_theta))
                    proposed0_logp = proposal_logp(current_theta, cov)
                    target0_logp = np.log(func(current_theta))

                    logratio = target_logp - proposed_logp - target0_logp + proposed0_logp

                    if np.log(random.random()) < -logratio:
                        current_theta = proposed_theta
                    samples[i * burn * (steps//burn) + j * (steps//burn) + k] = current_theta
            sigma = np.cov(samples[:(i+1) * burn * (steps//burn), :].T)
            #if (sigma < 1).all():
            #    sigma /= np.max(np.abs(sigma))
            #if (np.abs(sigma) > 100).all():
            #    sigma /= np.max(np.abs(sigma))
    #cov = np.array([[3,0.3],[0.11,3]])
    samples = get_samples(start, cov)

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
    p = np.random.normal(loc=0, scale=1, size=len(q))  # independent standard normal variates
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
    img = None
    factor = 1
    print("building image...")
    if n == 2:
        hi_res = False
        if hi_res:
            img = np.zeros((300, 500))
            for i in range(500):
                print(i)
                if i % 50 == 0:
                    print(f"{i // 50} % done")
                for j in range(300):
                    img[299 - j, i] = np.exp(-f((i/10 - 25 ,j/10 - 20.0)))
                    #img[299 - j, i] = -f((i/10 - 25 ,j/10 - 20.0))
        else:
            img = np.zeros((300, 500))
            for i in range(50):
                print(i)
                for j in range(30):
                    #img[299 - j, i] = -f((i/10 - 25 ,j/10 - 20.0))
                    #factor = 2
                    img[299-((10*j)+9):299-(10*j) + 1, (10 * i):(10 * i + 10)] = np.exp(-f(((i * factor) + 0.5 * factor - (25 * factor), ((j * factor) + 0.5 * factor) - (20.0 * factor))))

    print(np.sum(np.sum(1-img)))
    ## HMC
    L = 200
    epsilon = 0.11#0.6
    current_q = np.array((0, 0))
    m = 100
    samples = np.zeros((m, 2))
    #pdf(paste("trajectories-ex02.pdf", sep=""), width=9, height=3)
    for i in range(m):
        res = HMC(f, grad, epsilon, L, current_q)
        samples[i, :] = np.array([res[0][0], res[0][1]])
        current_q = res[0]
        if (i > 10):
            print("ess:", m * arviz.ess(samples[:, :2].T) / i)  # monitor effective size of first 3 components

        # plot trajectory
        if i % 20 == 19:
            traj = res[1]
            #plt.figure(figsize=(15,10))
            if n == 2: plt.imshow(img, extent=[-25 * factor,25 * factor,-20 * factor,10 * factor])
            plt.scatter([traj[0][1][0], traj[1][1][0]],[traj[0][1][1], traj[1][1][1]], alpha=0.1)
            for h in range(2, len(traj)):
                ...
                plt.scatter([traj[h - 1][1][0], traj[h][1][0]],[traj[h- 1][1][1], traj[h][1][1]], alpha=0.1)
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
    return -gs

def logi_fun(xs):
    ...

if __name__ == "__main__":
    #n = neff(np.array((1,2,3,1,2,3,1,2,3,1,2,3)))
    #h = arviz.ess(np.array((1,2,3,1,2,3,1,2,3,1,2,3)), method="identity")
    dataset = pd.read_csv("datset.csv").to_numpy()
    logreg1 = LogisticRegression(penalty="none",fit_intercept=False)
    logreg1.fit(dataset[:, :11], dataset[:, -1])
    logreg2 = LogisticRegression()
    logreg2.fit(dataset[:, :11], dataset[:, -1])
    #print(dataset)

    def logistic_1(x):#log prob
        x = np.atleast_2d(x).T
        z = dataset[:, :2] @ x
        return np.log(np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z))

    def logi1_grad(x):
        z = np.exp(-(dataset[:, :2] @ np.atleast_2d(x).T))
        tz = z / (1 + z)
        return -(dataset[:,:2].T @ tz - dataset[:,:2].T @ (1 - dataset[:,-1:])).squeeze()


    def logistic_2(x):  # log prob
        x = np.atleast_2d(x).T
        z = dataset[:, :11] @ x
        return np.log(np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z))


    def logi2_grad(x):
        z = np.exp(-(dataset[:, :11] @ np.atleast_2d(x).T))
        tz = z / (1 + z)
        return -(dataset[:, :11].T @ tz - dataset[:, :11].T @ (1 - dataset[:, -1:])).squeeze()

    logbivariate = lambda x:-np.log(multivariate_normal.pdf(x, mean=np.zeros(2), cov=np.array([[2,-1],[-1,2]])))
    def logbivar_grad(x):
        return (np.linalg.inv(np.array([[2,-1],[-1,2]])) @ np.atleast_2d(x).T).T.squeeze()
    #bivar.prob = function(x)
    #{
    #return (dmvnorm(x=c(x), mean=rep(0, 2), sigma=covmat))
    #}

#    bivar.prob.log = function(x)
#    {
#    return (log(dmvnorm(x=c(x), mean=rep(0, 2), sigma=covmat)))
#}

#bivar.log.grad = function(x)
#{
#return (-solve(covmat) % * % x)
#}




    banana = lambda x: minus_logf(x)
    bananagrad = minus_logf_grad

    #fun = lambda x: -np.log(funt(x))
    fun = logistic_1
    #fun = banana
    #fun = logbivariate
    #grad = lambda x:gradf(logistic_1, x)
    #grad = logi1_grad
    grad = bananagrad
    #grad = logbivar_grad
    #hmc_f(fun, grad, 2) #TESTED
    samples = metropolis_hastings(np.ones(2), fun, burnin=True, burn=10, steps=1000)
    plt.scatter(samples[:,0], samples[:,1])
    plt.show()

