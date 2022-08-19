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
    #print(cov)
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
        current_q = np.zeros(dim)
    for i in range(num):
        res = HMC(f, grad, epsilon, L, current_q)
        samples[i, :] = np.array([res[0][g] for g in range(dim)])
        current_q = res[0]
    return samples

def hmc_f(f, grad, n, L=500, epsilon=0.11):
    #plt.clf()
    #make_img(f)
    #plt.show()
    #plt.clf()
    #make_img(lambda x:np.sum(grad(x) ), exp=False)
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
            ...
            #print("ess:", m * arviz.ess(samples[:, :2].T) / i)  # monitor effective size of first 3 components

        # plot trajectory
        if i % 20 == 19:
            traj = res[1]
            plt.figure(figsize=(15,10))
            #if n != 32: plt.imshow(img, extent=[-25 * factor,25 * factor,-20 * factor,10 * factor])
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

dataset = pd.read_csv("datset.csv").to_numpy()
#print(dataset)
reg1 = LogisticRegression(fit_intercept=False, penalty="none")
reg1.fit(dataset[:,:2], dataset[:, -1])
reg2 = LogisticRegression(fit_intercept=False, penalty="none")
reg2.fit(dataset[:, :11], dataset[:, -1])

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
def s(x):
    h = np.exp(-x)
    return 1 / (1 + h)
# def logi12(x):
#
#     x = np.atleast_2d(x)
#     p = s(dataset[:, :2].dot(x.T))
#     loss = np.multiply(dataset[:, -1:],np.log(p + 1e-15)) + \
#            np.multiply((1 - dataset[:, -1:]),np.log(1 - p + 1e-15))
#     return np.log(-np.sum(loss))
def logistic_1(x):
    x = np.atleast_2d(x).T
    z = dataset[:, :2] @ x
    temp = np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z) #loglikelihood
    return np.log(temp) #additional log, since the values are so small
def logi1_grad(x):#Derivative of log-loglikelihood, since values were too small to handle otherwise
    z = np.exp(-dataset[:, :2] @ np.atleast_2d(x).T)
    tz = z / (1 + z)
    temp = -(dataset[:,:2].T @ tz - dataset[:,:2].T @ (1 - dataset[:,-1:])).squeeze()
    return temp / np.exp(logistic_1(x)) #grad of log
# def logi12_g(x):
#     x = np.atleast_2d(x)
#     p = s(dataset[:, :2] @ x.T)
#     temp = dataset[:, :2].T @ (dataset[:, -1:] - p)
#     return temp
#def logistic_1(x):#log prob
#    x = np.atleast_2d(x).T
#    z = dataset[:, :2] @ x
#    return np.log(np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z))
    #z = dataset[:, :2] @ x
    #return np.log(np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z))

#def logi1_grad(x):
#    z = np.exp(-(dataset[:, :2] @ np.atleast_2d(x).T))
#    tz = z / (1 + z)
#    return -(dataset[:,:2].T @ tz - dataset[:,:2].T @ (1 - dataset[:,-1:])).squeeze()
    #z = np.exp(-(dataset[:,:2].dot(x)))

    #tz = z / (1 + z)
    #return -(dataset[:, :2].T @ tz - dataset[:, :2].T @ (1 - dataset[:, -1]))
def logistic_2(x):
    x = np.atleast_2d(x).T
    z = dataset[:, :11] @ x
    temp = np.sum(np.log(1 + np.exp(-z)) + (1 - dataset[:, -1:]) * z) #loglikelihood
    return np.log(temp) #additional log, since the values are so small
def logi2_grad(x):#Derivative of log-loglikelihood, since values were too small to handle otherwise
    z = np.exp(-dataset[:, :11] @ np.atleast_2d(x).T)
    tz = z / (1 + z)
    temp = -(dataset[:,:11].T @ tz - dataset[:,:11].T @ (1 - dataset[:,-1:])).squeeze()
    return temp / np.exp(logistic_2(x)) #grad of log

def logbivariate(x):
    return -np.log(multivariate_normal.pdf(x, mean=np.zeros(2), cov=np.eye(2)))
def logbivar_grad(x):
    return (np.linalg.inv(np.eye(2)) @ np.atleast_2d(x).T).T.squeeze()
def make_img(fun, exp=True):
    factor = 1
    img = np.zeros((300, 500))
    for i in range(500):
        print(i)
        if i % 50 == 0:
            print(f"{i // 50} % done")
        for j in range(300):
            if exp:
                img[299 - j, i] = np.exp(-fun((i / 10 - 25, j / 10 - 20.0)))
            else:
                img[299 - j, i] = fun((i / 10 - 25, j / 10 - 20.0))

            # img[299 - j, i] = -f((i/10 - 25 ,j/10 - 20.0))
    plt.imshow(img, extent=[-25 * factor, 25 * factor, -20 * factor, 10 * factor])

def bananamean(fun):
    factor = 1
    a, b = 2000, 1200
    img = np.zeros((b, a))
    xs = 0
    ys = 0
    for i in range(a):
        print(xs, ys)
        if i % 50 == 0:
            print(f"{i // 5} % done")
        for j in range(b):
            x = i / 10 - (a / 20)
            y = j / 10 - (b / 15)
            img[b - 1 - j, i] = np.exp(-fun((x, y)))
            xs += x * img[b - 1 - j, i]
            ys += y * img[b - 1 - j, i]
    print(xs, ys)
    print(xs / a, ys / b)

    plt.imshow(img, extent=[-25 * factor, 25 * factor, -20 * factor, 10 * factor])
    plt.show()

def elevenDim():#AAAAAAAAAAAA
    fun = logistic_2
    chains = np.zeros((5,1000,11))
    grad = logi2_grad
    cov =  np.array([[ 2.17716554e+09,  3.38517214e+08,  2.74833155e+08,  9.98393490e+07,
                       1.26402760e+07, -3.79417918e+08,  4.36967307e+07,  9.68642574e+07,
                       2.13187176e+08,  5.40818200e+07,  1.78196854e+08,],
                     [ 3.38517214e+08,  5.29968514e+07,  4.29087188e+07,  1.54761715e+07,
                       2.07978134e+06, -5.88151439e+07,  6.31168260e+06,  1.49766048e+07,
                       3.35060229e+07,  8.50565103e+06,  2.75860710e+07,],
                     [ 2.74833155e+08,  4.29087188e+07,  3.50405728e+07,  1.26207468e+07,
                       1.72701866e+06, -4.77618112e+07,  5.27548001e+06,  1.21923485e+07,
                       2.72590628e+07,  7.14864633e+06,  2.24947652e+07,],
                     [ 9.98393490e+07,  1.54761715e+07,  1.26207468e+07,  4.59213821e+06,
                       5.74681295e+05, -1.74135193e+07,  2.06473271e+06,  4.45386870e+06,
                       9.75600365e+06,  2.50892095e+06,  8.19687945e+06,],
                     [ 1.26402760e+07,  2.07978134e+06,  1.72701866e+06,  5.74681295e+05,
                       1.42366201e+05, -2.14403081e+06,  1.10178989e+05,  5.40325209e+05,
                       1.40057243e+06,  4.31578354e+05,  1.01596951e+06,],
                     [-3.79417918e+08, -5.88151439e+07, -4.77618112e+07, -1.74135193e+07,
                      -2.14403081e+06,  6.62548446e+07, -7.87644031e+06, -1.69335743e+07,
                      -3.69653186e+07, -9.32917536e+06, -3.10941292e+07,],
                     [ 4.36967307e+07,  6.31168260e+06,  5.27548001e+06,  2.06473271e+06,
                       1.10178989e+05, -7.87644031e+06,  1.53600804e+06,  2.06332653e+06,
                       3.80560606e+06,  9.54897192e+05,  3.73399400e+06,],
                     [ 9.68642574e+07,  1.49766048e+07,  1.21923485e+07,  4.45386870e+06,
                       5.40325209e+05, -1.69335743e+07,  2.06332653e+06,  4.33391953e+06,
                       9.41316892e+06,  2.38927571e+06,  7.95389200e+06,],
                     [ 2.13187176e+08,  3.35060229e+07,  2.72590628e+07,  9.75600365e+06,
                       1.40057243e+06, -3.69653186e+07,  3.80560606e+06,  9.41316892e+06,
                       2.13639786e+07,  5.56713225e+06,  1.73569178e+07,],
                     [ 5.40818200e+07,  8.50565103e+06,  7.14864633e+06,  2.50892095e+06,
                       4.31578354e+05, -9.32917536e+06,  9.54897192e+05,  2.38927571e+06,
                       5.56713225e+06,  1.66466341e+06,  4.46128355e+06,],
                     [ 1.78196854e+08,  2.75860710e+07,  2.24947652e+07,  8.19687945e+06,
                       1.01596951e+06, -3.10941292e+07,  3.73399400e+06,  7.95389200e+06,
                       1.73569178e+07,  4.46128355e+06,  1.46462958e+07,]])
    for Ls in [10,20,50,100]:
        for eps in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            ...
    Ls = 27
    eps = 0.6
    for algo in [#hamilton_MC,# 20 0.5
                 metropolis_hastings
                 ]:
        for tune in [True, False]:#, False]:

            start, stop = 0, 0
            for i, color in enumerate(["r", "g", "b", "cyan", "yellow"]):
                #print(f"running chain {i}")
                if algo == hamilton_MC:
                    if tune:
                        start += time.time()
                        samples = hamilton_MC(fun, grad, epsilon=eps, L=Ls, num=1000, dim=11)
                        stop += time.time()
                    else:
                        start += time.time()
                        samples = hamilton_MC(fun, grad, epsilon=1.0, L=10, num=1000, dim=11)
                        stop += time.time()
                    print(np.unique(samples, axis=0).shape)

                elif algo == metropolis_hastings:
                    if tune:
                        start += time.time()
                        # hmc_f(fun, grad, n=200, L=15, epsilon=0.11)
                        samples = metropolis_hastings(np.zeros(11), fun, cov=cov/1e8, burnin=False, burn=10, steps=1000, burn_steps=1000)
                        stop += time.time()
                    else:
                        start += time.time()
                        samples = metropolis_hastings(np.zeros(11), fun, cov=np.eye(11), burnin=False, burn=10, steps=1000, burn_steps=1000)
                        stop += time.time()
                chains[i, :, :] = samples
            chns = chains.reshape((5000, 11))
            # reg3.fit(chns, reg1.predict(chns))
            print(reg2.coef_)
            print("&".join([np.format_float_positional(x, precision=2) for x in chns.mean(axis=0)]))
            idata = arviz.InferenceData()
            idata.add_groups(
                {"posterior": {"x1": chains[:, :, 0], "x2": chains[:, :, 1]}},
                dims={"obs": None}
            )
            avg_time = (stop - start) / 5
            avg_ess = arviz.ess(idata, var_names=["x1", "x2"])
            avg_ess = avg_ess["x1"].to_numpy() + avg_ess["x2"].to_numpy()
            avg_ess = avg_ess / 2
            print(avg_ess, "Effective sample size")
            print(avg_ess/avg_time, "ESS/s")



reg3 = LogisticRegression(fit_intercept=False, penalty="none")
if __name__ == "__main__":
    elevenDim()
    quit()
    banana = lambda x: minus_logf(x)
    #bananamean(banana)
    #quit()
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
                              (logistic_1,   logi1_grad,    np.array([[9,-2],[-2,3]]))
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
            for tune in [True, False]:
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
                plt.figure(figsize=(6,6))
                #cov = np.array([[700, -350],[-250,700]])
                for i, color in enumerate(["r", "g", "b", "cyan", "yellow"]):
                    print(f"running chain {i}")
                    if algo == hamilton_MC:
                        if fun == logistic_1:
                            start += time.time()
                            if tune:
                                #hmc_f(fun, grad, n=200, L=15, epsilon=0.11)
                                samples = hamilton_MC(fun, grad, epsilon=0.11, L=15, num=1000)
                            else:
                                samples = hamilton_MC(fun, grad, epsilon=0.1, L=10, num=1000)
                        elif fun == banana:
                            start += time.time()
                            if tune:
                                # hmc_f(fun, grad, n=200, L=15, epsilon=0.11)
                                samples = hamilton_MC(fun, grad, epsilon=0.6, L=27, num=1000)
                            else:
                                samples = hamilton_MC(fun, grad, epsilon=0.1, L=10, num=1000)
                        else:
                            start += time.time()
                            if tune:
                                # hmc_f(fun, grad, n=200, L=15, epsilon=0.11)
                                samples = hamilton_MC(fun, grad, epsilon=0.3, L=50, num=1000)
                            else:
                                samples = hamilton_MC(fun, grad, epsilon=0.1, L=10, num=1000)

                    elif algo == metropolis_hastings:
                        start += time.time()
                        samples = metropolis_hastings(np.zeros(2), fun, cov=cov, burnin=False, burn=10, steps=1000, burn_steps=1000)
                    elif algo == rejection_sampling:
                        maxi=1
                        bounds = None
                        if fun == banana:
                            maxi = 1
                            bounds = np.array([[-25, 25], [-20, 10]])
                        elif fun == logbivariate:
                            maxi = 0.1
                            bounds = np.array([[-5, 5], [-5, 5]])
                        elif fun == logistic_1:
                            maxi = 0.01
                            #I x2'd this for larger sample after making images
                            bounds = np.array([[-50, 50], [-50, 50]])
                        if not tune:
                            maxi = 1
                        start += time.time()
                        samples = rejection_sampling(fun, dim=2, n=1000, bounds=bounds, maximum=maxi)
                    end += time.time()
                    #unique_samples = np.unique(samples, axis=0)np.array([[200, -20],[-20, 70]])
                    #unique_samples = samples
                    chains[i, :, :] = samples
                    unique_samples = np.unique(samples, axis=0)
                    plt.scatter(unique_samples[:, 0], unique_samples[:, 1], alpha=0.1, color=color)
                    plt.plot(samples[:, 0], samples[:, 1], alpha=0.2, color=color, linewidth=0.4)
                avg_time = (end - start)/5 #in seconds
                plt.tight_layout()
                chns = chains.reshape((5000,2))
                #reg3.fit(chns, reg1.predict(chns))
                print(reg1.coef_)
                print(chns.mean(axis=0))
                #plt.savefig(f"{h}_{g}_tuning{tune}_samples.png")
                #axes = np.atleast_2d(axes)
                idata = arviz.InferenceData()
                idata.add_groups(
                    {"posterior": {"x1": chains[:, :, 0], "x2": chains[:, :, 1]}},
                    dims={"obs": None}
                )
                plt.clf()
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
                #plt.savefig(f"{h}_{g}_tune{tune}.png")
                fig.clf()
                plt.clf()
                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                arviz.plot_autocorr(idata, var_names=["x1"], ax=axes[0])
                arviz.plot_autocorr(idata, var_names=["x2"], ax=axes[1])

                fig.tight_layout()
                #plt.savefig(f"{h}_{g}_tune{tune}_autocorr.png")
                plt.clf()
                fig.clf()
                #plt.show()
                #arviz.plot_trace(chains, compact=True, kind="trace",figsize=(15,10))
                #plt.tight_layout()
