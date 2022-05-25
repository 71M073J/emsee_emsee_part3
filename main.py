import numpy as np

from scipy.stats import multivariate_normal


#multivariate_normal.pdf(x, mean=2.5, cov=0.5)

def logp(a, b):
    cov1 = np.eye(a.shape[0]) * a
    cov2 = np.eye(b.shape[0]) * b
    return np.log(multivariate_normal.pdf(a, cov1)) + np.log(multivariate_normal.pdf(b, cov2))

def metropolis_hastings(start, func, cov, size=1, steps=1000):
    rng = np.random.default_rng()
    mean = np.zeros(start.shape[0])
    #target_distribution = func
    #theta 0
    #theta_t = func(start)
    cov_m = np.eye(start.shape[0]) * start
    cov = cov_m * (2.4**2) / 2
    current_theta = start.copy()

    tuning = 10
    samples = np.zeros((tuning * (steps//tuning),start.shape[0]))
    rate = np.zeros(tuning)
    for j in range(tuning):
        for i in range(steps//tuning):

            proposed_theta = rng.multivariate_normal(mean=np.zeros(size), cov=cov)
            log_r = logp(proposed_theta, current_theta) - logp(current_theta, proposed_theta)
            if np.log(rng.uniform(0,1)) < log_r:
                current_theta = proposed_theta
            samples[j * steps//tuning + i,:] = current_theta
        rate[j] = np.unique(samples[:(j + 1) * (steps//tuning), 0]).shape[0] / ((j + 1) * (steps//tuning))
        cov = samples[:(j + 1) * (steps//tuning),:].var()
        print(cov)





if __name__ == "__main__":
    metropolis_hastings(np.ones(1), None, None)


