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

def rejection_sampling(func, dim, n=1000, bounds=None, maximum=1.0):
    if bounds is None:
        bounds = np.ones((dim, 2)) * 10
        bounds[:,0] *= -1
    def sample(bnds):
        upper = bnds[1]
        lower = bnds[0]
        return random.random() * (upper - lower) + lower
    num_samps = 0
    samples = np.zeros((n, dim))
    while num_samps < n:
        point = np.array([sample(bounds[x]) for x in range(dim)])
        if np.exp(-func(point)) > random.random() * maximum:
            samples[num_samps, :] = point
            num_samps += 1
    return samples





def metropolis_hastings(start, func, steps=10000, burnin=False, cov=None, burn=10, burn_steps=1000, step_size=1):
    dim = start.shape[0]
    def proposal(dimension, cov):
        return rng.multivariate_normal(mean=np.zeros(dimension), cov=cov)
    def proposal_prob(x, cov):
        return multivariate_normal.pdf(x, mean=np.zeros(x.shape[0]), cov=cov, allow_singular=True)
    def proposal_logp(x, cov):
        return np.log(proposal_prob(x, cov))
    def get_samples(start, cov):
        current_theta = start
        samples = np.zeros((steps, dim))
        for i in range(steps):
            proposed_theta = current_theta + proposal(dim, cov) * step_size
            #proposed_theta = proposal(dim, cov)
            #proposed_logp = proposal_logp(proposed_theta, cov)
            #target_logp = -np.log(func(proposed_theta))
            target_logp = -func(proposed_theta)
            #proposed0_logp = proposal_logp(current_theta, cov)
            #target0_logp = -np.log(func(current_theta))
            target0_logp = -func(current_theta)
            logratio = (target_logp - target0_logp)# + (proposed0_logp - proposed_logp)
            #logratio = target_logp - proposed_logp - target0_logp + proposed0_logp

            #if np.log(random.random()) < -logratio:
            if random.random() < np.exp(logratio):
                current_theta = proposed_theta
            samples[i, :] = current_theta
        return samples

    sigma = np.eye(dim)
    if cov is None:
        cov = (2.4 ** 2) * sigma / dim
    samples = np.zeros((burn * burn * burn_steps, dim))
    print(cov)
    if burnin:
        for i in range(burn):
            ... #just do manual tuning... q.q
            print(cov)
            acceptance = 0
            current_theta = start
            for j in range(burn):
                for k in range(burn_steps):
                    proposed_theta = current_theta + proposal(dim, cov) * step_size
                    #proposed_theta = proposal(dim, cov)
                    #proposed_logp = proposal_logp(proposed_theta, cov)
                    target_logp = -func(proposed_theta)
                    #proposed0_logp = proposal_logp(current_theta, cov)
                    target0_logp = -func(current_theta)
                    logratio = (target_logp - target0_logp)# + (proposed0_logp - proposed_logp)
                    #logratio = target_logp - proposed_logp - target0_logp + proposed0_logp

                    # if np.log(random.random()) < -logratio:
                    if random.random() < np.exp(logratio):
                        current_theta = proposed_theta
                        acceptance += 1
                    #proposed_theta = proposal(dim, cov)
                    #proposed_logp = proposal_logp(proposed_theta, cov)
                    #target_logp = func(proposed_theta)
                    #proposed0_logp = proposal_logp(current_theta, cov)
                    #target0_logp = func(current_theta)

                    #logratio = target_logp - proposed_logp - target0_logp + proposed0_logp
                    #rand = random.random()
                    #if np.log(rand) < -logratio:
                    #    if not np.isnan(proposed_theta).any():
                    #        current_theta = proposed_theta
                    samples[i * burn * burn_steps + j * burn_steps + k] = current_theta
            sigma = np.cov(samples[:(i+1) * burn * burn_steps, :].T)
            cov = (2.4 ** 2) * sigma / dim

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

def hamilton_MC(f, grad, num=1000, epsilon=0.11, dim=2, L=27, current_q=None):
    samples = np.zeros((num,dim))
    if current_q is None:
        current_q = np.array((0, 0))
    for i in range(num):
        res = HMC(f, grad, epsilon, L, current_q)
        samples[i, :] = np.array([res[0][0], res[0][1]])
        current_q = res[0]
    return samples

def hmc_f(f, grad, n, L=500, epsilon=0.11):
    img = None
    factor = 1
    print("building image...")
    if n == 13:
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

    ## HMC
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
            #if n == 2: plt.imshow(img, extent=[-25 * factor,25 * factor,-20 * factor,10 * factor])
            #plt.scatter([traj[0][1][0], traj[1][1][0]],[traj[0][1][1], traj[1][1][1]], alpha=0.1)
            for h in range(2, len(traj)):
                ...
                #plt.scatter([traj[h - 1][1][0], traj[h][1][0]],[traj[h- 1][1][1], traj[h][1][1]], alpha=0.1)
                #plt.scatter([h[0][0]], [h[0][1]])
                #print(h)
            #plt.ylim(-2, 2)
            #plt.xlim(-2, 2)
            #plt.imshow(img)
            #plt.savefig()


            #plt.show()

dataset = pd.read_csv("datset.csv").to_numpy()
#print(dataset)
reg1 = LogisticRegression(fit_intercept=False)
reg1.fit(dataset[:,:2], dataset[:, -1])
reg2 = LogisticRegression(fit_intercept=False)
reg2.fit(dataset[:, :11], dataset[:, -1])


def s(x):
    h = np.exp(-x)
    return h / (1 + h)
def logistic_1(x):

    #x = np.atleast_2d(x)
    #p = s(-x.dot(dataset[:, :2].T)).T
    #loss = np.multiply(dataset[:, -1:],np.log(p + 1e-15)) + \
    #       np.multiply((1 - dataset[:, -1:]),np.log(1 - p + 1e-15))
    #return np.log(-np.sum(loss))

    x = np.atleast_2d(x).T
    z = dataset[:, :2] @ x
    temp = np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z) #loglikelihood
    return np.log(temp) #additional log, since the values are so small
def logi1_grad(x):#Derivative of log-loglikelihood, since values were too small to handle otherwise
    z = np.exp(-(dataset[:, :2] @ np.atleast_2d(x).T))
    tz = z / (1 + z)
    temp = -(dataset[:,:2].T @ tz - dataset[:,:2].T @ (1 - dataset[:,-1:])).squeeze()
    return temp# * 1/logistic_1(x)

def logistic_2(x):  # log prob
    #x = np.atleast_2d(x).T
    #z = dataset[:, :11] @ x
    #return np.log(np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z))
    x = np.atleast_2d(x).T
    z = dataset[:, :11] @ x
    temp = np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z) #loglikelihood
    return np.log(temp) #additional log, since the values are so small


def logi2_grad(x):
    #z = np.exp(-(dataset[:, :11] @ np.atleast_2d(x).T))
    #tz = z / (1 + z)
    #return -(dataset[:, :11].T @ tz - dataset[:, :11].T @ (1 - dataset[:, -1:])).squeeze()

    z = np.exp(-(dataset[:, :11] @ np.atleast_2d(x).T))
    tz = z / (1 + z)
    temp = -(dataset[:,:11].T @ tz - dataset[:,:11].T @ (1 - dataset[:,-1:])).squeeze()
    return temp * 1/logistic_2(x) #grad of log

def logbivariate(x):
    return -np.log(multivariate_normal.pdf(x, mean=np.zeros(2), cov=np.eye(2)))
def logbivar_grad(x):
    return (np.linalg.inv(np.eye(2)) @ np.atleast_2d(x).T).T.squeeze()

if __name__ == "__main__":
    banana = lambda x: minus_logf(x)
    bananagrad = minus_logf_grad
    file = open("output_ess.txt", "a")
    print("-------------------------------------", file=file)
    #fun = logistic_1
    #fun = banana
    #fun = logbivariate
    #grad = logi1_grad
    #grad = bananagrad
    #grad = logbivar_grad
    #hmc_f(fun, grad, 2, epsilon=0.11) #TESTED1
    #ZAKAJ AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA?
    #samples = rejection_sampling(fun, bounds=np.array([[-25,25],[-20,10]]), dim=2, )#tested
    cov1 = np.array([[1, 0],[0,1]])
    #cov2 = np.array([[100, 30],[30,70]])
    for funct, grad, cov2 in [(logbivariate, logbivar_grad, np.array([[2.88, 0],[0,2.88]])),
                              (banana,       bananagrad,    np.array([[270, -10],[-10,100]])),
                              (logistic_1,   logi1_grad,    np.array([[700,-100],[-100,20]]))
                              ]:
        if funct == logbivariate:
            h = 0
        elif funct == banana:
            h = 1
        elif funct == logistic_1:
            h = 2
        for algo in [hamilton_MC,
                     metropolis_hastings,
                     rejection_sampling
                     ]:
            if algo == hamilton_MC:
                g = 0
            elif algo == metropolis_hastings:
                g = 1
            elif algo == rejection_sampling:
                g = 2
            for tune in [False, True]:
                if tune:
                    cov = cov2
                else:
                    cov = cov1

                print("making image")

                #plt.figure()

                img = np.zeros((300, 500))
                factor = 1
                fun = funct
                #fun = logistic_1
                for i in range(50):
                    # print(i)
                    for j in range(30):
                        # img[299 - j, i] = -f((i/10 - 25 ,j/10 - 20.0))
                        # factor = 2
                        img[299 - ((10 * j) + 9):299 - (10 * j) + 1, (10 * i):(10 * i + 10)] = np.exp(
                            -fun(
                                ((i * factor) + 0.5 * factor - (25 * factor), ((j * factor) + 0.5 * factor) - (20.0 * factor))))

                #axes[0, 0].imshow(img, extent=[-25 * factor, 25 * factor, -20 * factor, 10 * factor])
                #axes[0, 1].imshow(img, extent=[-25 * factor, 25 * factor, -20 * factor, 10 * factor])

                #plt.imshow(img, extent=[-25 * factor, 25 * factor, -20 * factor, 10 * factor])

                #
                chains = np.zeros((5, 1000, 2))
                start, end, ess = 0,0,0
                #cov = np.array([[700, -350],[-250,700]])
                for i, color in enumerate(["r", "g", "b", "cyan", "yellow"]):
                    print(f"running chain {i}")
                    if algo == hamilton_MC:
                        start += time.time()
                        if tune:
                            #hmc_f(fun, grad, n=200, L=15, epsilon=0.11)
                            samples = hamilton_MC(fun, grad, epsilon=0.11, L=15, num=1000)
                        else:
                            samples = hamilton_MC(fun, grad, epsilon=0.1, L=10, num=1000)

                    elif algo == metropolis_hastings:
                        start += time.time()
                        samples = metropolis_hastings(np.zeros(2), fun, cov=cov, burnin=False, burn=10, steps=1000, burn_steps=1000)
                    elif algo == rejection_sampling:
                        maxi=1
                        if fun == banana:
                            maxi = 1
                        elif fun == logbivariate:
                            maxi = 0.1
                        elif fun == logistic_1:
                            maxi = 0.001
                        start += time.time()
                        samples = rejection_sampling(fun, dim=2, n=1000, bounds=np.array([[-25,25],[-20,10]]), maximum=maxi)
                    end += time.time()
                    #unique_samples = np.unique(samples, axis=0)np.array([[200, -20],[-20, 70]])
                    #unique_samples = samples
                    chains[i, :, :] = samples
                    unique_samples = np.unique(samples, axis=0)
                    plt.scatter(unique_samples[:, 0], unique_samples[:, 1], alpha=0.1, color=color)
                avg_time = (end - start)/5 #in seconds
                plt.tight_layout()
                plt.savefig(f"{h}_{g}_tuning{tune}_samples.png")
                #axes = np.atleast_2d(axes)
                idata = arviz.InferenceData()
                idata.add_groups(
                    {"posterior": {"x1": chains[:, :, 0], "x2": chains[:, :, 1]}},
                    dims={"obs": None}
                )
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                avg_ess = arviz.ess(idata, var_names=["x1", "x2"])
                avg_ess = avg_ess["x1"].to_numpy() + avg_ess["x2"].to_numpy()
                avg_ess = avg_ess / 2
                print(f"{algo}\n{funct}\nTuning:{tune}\n", file=file)
                print(avg_ess, "Average effective sample size", file=file)
                print(avg_ess/avg_time, "Average ESS/s", file=file)
                print("-------------------------",file=file, flush=True)
                arviz.plot_trace(idata, var_names=["x1","x2"], kind="trace", compact=True, axes=axes, legend=True)
                fig.tight_layout()
                plt.savefig(f"{h}_{g}_tune{tune}.png")
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                arviz.plot_autocorr(idata, var_names=["x1"], ax=axes[0])
                arviz.plot_autocorr(idata, var_names=["x2"], ax=axes[1])
                fig.tight_layout()
                plt.savefig(f"{h}_{g}_tune{tune}_autocorr.png")
                #plt.show()
                #arviz.plot_trace(chains, compact=True, kind="trace",figsize=(15,10))
                #plt.tight_layout()