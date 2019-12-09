# Variational Inference: A Review for Statisticians

By 

David M. Blei - Department of Computer Science and StatisticsColumbia University

Alp Kucukelbir - Department of Computer ScienceColumbia University

Jon D. McAuliffe - Department of StatisticsUniversity of California, Berkeley



## Introduction

Core problem of modern stat is to approximate probability density (that is difficult).

very important in a Bayesian stat that is based on posterior not easy to compute 

this doc is about **Variational Inference (VI)** a method from machine learning that approximate posterior probability densities. The **VI** is an alternative to **MCMC** (markov chain Monte Carlo Sampling), that can be scaled to a large set of data (easy and fast).

GENERAL PROBLEM :  $p(z,x) = p(x)p(x|z)$

where, $z = z_{1:m}$ is the latent variable and $x = x_{1:n}$ the observation

Bayesian models : 

- Latent var help to govern the distribution of data. 
- draws the latent var from variables from a prior density $p(z)$ and then relates them to the observations through the likelihood $p(x|z)$

MCMC models :

* first construct a ergodic Markov chain on $z$
* whose stationary distribution is $p(x|z)$
* hen, we sample from the chain to collect samples from the stationary distribution. 
* Finally, we approximate the posterior with an empirical estimate constructed from (a subset of) the collected samples

**VI** IDEAS :

* use optimization : First, we posit a *family* of approximate densities $D$. This is a set of densities over the latent variables. 

* Then, we try to find the member of that family that minimizes the Kullback-Leibler(KL) divergence to the exact posterior,

  $q∗(z)=arg min_{q(z)∈D} KL(q(z)‖p(z|x))$

* Finally, we approximate the posterior with the optimized member of the family $q∗(·)$.

**VI** turn an interference problem into an optimization problem.

*We emphasize that MCMC and variational inference are different approaches to solving the same problem. MCMC algorithms sample a Markov chain; variational algorithms solve an optimization problem. MCMC algorithms approximate the posterior with samples from the chain; variational algorithms approximate the posterior with the result of the optimization*

## Variational inference

the goal of variational inference is to approximate a conditional density of latent variables given observed variables. The key idea is to solve this problem with optimization. We use a family of densities over the latent variables, parameterized by free “variational parameters”.

### The problem of approximate inference

here we note : 

- $x = x_{1:n}$ the observed variables
- $z = z_{1:m}$ the latent variables
- with join density $p(z,x)$

The inference problem is to compute the conditional density of the latent variable given the observations $p(z|x)$. 

conditional density : $p(z|x)=\frac{p(x,z)}{p(x)}$

with the joint density : $p(x)= \int p(z,x)dz$

For many models, this evidence integral is unavailable in closed form or requires exponential time to compute. The evidence is what we need to compute the conditional from the joint;this is why inference in such models is hard