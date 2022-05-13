# - This is an example of Hamiltonian trajectories (not HMC, just Hamiltonian Dynamics),
# - we use the banana function as an example,
# - each next trajectory starts from end of previous,
# - momentum is resampled for every new trajectory,
# - observe how the dynamics depend on starting total energy!

hamiltonian_dynamics <- function (U, grad_U, current_q, current_p, epsilon, L)
{
  traj <- NULL

  q = current_q
  p = current_p
  
  traj <- rbind(traj, data.frame(t(p), t(q)))
  
  p <- p - epsilon * grad_U(q) / 2
  
  for (i in 1:L)
  {
    q <- q + epsilon * p
    if (i != L) p <- p - epsilon * grad_U(q)
    traj <- rbind(traj, data.frame(t(p), t(q)))
  }

  p <- p - epsilon * grad_U(q) / 2
  traj <- rbind(traj, data.frame(t(p), t(q)))
  list(next_q = q, traj = traj)
}

library(ggplot2)


minus_logf <- function(x, B = 0.05) {
  -(-(x[1]^2)/200- 0.5 * (x[2]+ B * x[1]^2 - 100*B)^2 )
}

minus_logf_grad <- function(x, B = 0.05) {
  g1 <- -(x[1])/100- 1.0 * (2* B * x[1]) * (x[2]+ B * x[1]^2 - 100*B)
  g2 <- - 1.0 * (x[2]+ B * x[1]^2 - 100*B)
  -c(g1,g2)
}

current_q <- c(0, 0)
pdf(paste("trajectories.pdf",sep=""), width = 3, height = 3)
set.seed(0)
for (i in 1:10) {
  print(i)
  current_p <- c(rnorm(1), rnorm(1))
  res <- hamiltonian_dynamics(minus_logf, minus_logf_grad, 
                             current_q = current_q, 
                             current_p = current_p, epsilon = 0.5, L = 100)

  # plot trajectory
    x <- seq(-25,25,0.2)
    x0 <- expand.grid(x,x)
    y <- apply(x0,1,minus_logf)
    df <- data.frame(x0,y = exp(-y))
    H <- minus_logf(current_q) + sum(current_p^2) / 2
    g2 <- ggplot(res$traj,aes(x=X1.1,y=X2.1)) + geom_point(size = 0.5) + 
      geom_path() + theme_bw() + xlab("q1")  + coord_cartesian(xlim=c(-25, 25), ylim=c(-20,10)) + ylab("q2") +
      geom_point(data=res$traj[1,], colour = "red", aes(x=X1.1,y=X2.1)) +
      geom_contour(data = df, mapping =  aes(Var1, Var2, z = y), alpha = 0.2, colour="black") +
      ggtitle(sprintf("Potential = %.3f\nMomentum = %.3f, %.3f\nEnergy = %.3f\n", 
                      minus_logf(current_q), current_p[1], current_p[2], H))
    plot(g2)
    
    current_q = res$next_q
    
  
  }
  
dev.off()
