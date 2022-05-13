library(ggplot2)

### RULES ----------------------------------------------------------------------

# uniform
rule_A <- function(x) {
  x_new <- runif(1, x - 0.1, x + 0.1)
  if (x_new > 1) x_new <- 2 - x_new
  if (x_new < 0) x_new <- 0 - x_new
  x_new
}

# triangular
rule_B <- function(x) {
  x_new <- runif(1, x - 0.1, x + 0.1)
  if (x_new > 1) x_new <- x_new - 1 # teleport
  if (x_new < 0) x_new <- 0 - x_new
  x_new
}

# half-gaussian-like
rule_C <- function(x) {
  x_new <- runif(1, x - 0.13, x + 0.1)
  if (x_new > 1) x_new <- 2 - x_new
  if (x_new < 0) x_new <- 0 - x_new
  x_new
}

# beta density M-H
rule_D <- function(x, alpha, beta, delta = 0.1) {
  x_new <- runif(1, x - delta, x + delta)
  if (x_new > 1 | x_new < 0) return (x)
  p <- exp(
    dbeta(x_new, alpha, beta, log = T) - 
             dbeta(x, alpha, beta, log = T))
  if (runif(1) > p) x_new <- x # reject
  x_new
}

# gamma density M-H
rule_E <- function(x, alpha, beta, delta = 1) {
  x_new <- rnorm(1, x, delta)
  if (x_new < 0) return (x)
  p <- exp(
    dgamma(x_new, alpha, beta, log = T) - 
      dgamma(x, alpha, beta, log = T))
  if (runif(1) > p) 
  { 
    x_new <- x # reject
  }
  x_new
}

### SIM ------------------------------------------------------------------------

set.seed(0)

# setup
m <- 1000
x <- array(0, dim = m + 1)

# parameters
my_rule <- rule_E  # local rule
x[1] <- 0        # starting position

# main
count <- 0
for (i in 1:m) {
  x[i + 1] <- my_rule(x[i], alpha = 5, beta = 1)
  if (x[i+1]==x[i]) {
    count <- count + 1
  }
}

print(count / m)

# plot
df <- data.frame(ID = 1:(m+1), x = x)
g1 <- ggplot(df, aes(x = x)) + 
  geom_histogram(center = 0.5, bins = 100)
plot(g1)

# DIAGNOSTICS:

# autocorrelation
acf(x, lag.max = 100)

# traceplot
g1 <- ggplot(df[1:1000,], aes(x = ID, y = x)) + 
  geom_line()
plot(g1)

# ESS (let's say we are estimating the mean)
library(mcmcse)
mcse(x)
ess(x)
m * (var(x) / m) / (mcse(x)$se)^2 # diy ESS: sample size times ratio of MC and MCMC error