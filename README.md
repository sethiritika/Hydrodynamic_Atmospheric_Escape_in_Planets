# Hydrodynamic Atmospheric Escape in Planets

This repository provides a simple notebook interface for estimating planetary atmospheric mass-loss rates. Users only need to supply the basic system parameters, such as stellar mass, planet mass, planet radius, orbital separation, gas temperature, and mean molecular weight, and the notebook evaluates mass-loss rates using commonly used models as well as our ''Mixture Model".

## Model Overview
The example notebook computes mass-loss rates using commonly used analytic models, including the Parker wind model (E. N. Parker 1960) and the L1/L2 nozzle model, as well as their hydrodynamics-calibrated extensions- The Modified spherical model, and the Modified tidal two-tail model, respectively.

The Parker wind model and Modified spherical model assume approximately spherical, isotropic outflow. These models are most appropriate for planets that are not strongly Roche-lobe filling, where the escaping gas is not primarily channeled through the Lagrange points.

The nozzle model and Modified tidal two-tail model represent the opposite limit, where mass loss is strongly shaped by the Roche potential and escapes through narrow streams near the L1 and L2 Lagrange points. These models are most appropriate for more Roche-lobe-filling planets with cooler, more tidally confined winds.

Our Mixture model provides a smooth interpolation between these spherical and L1/L2-directed anisotropic limits. Its weighting function is calibrated against 3D hydrodynamic simulations, allowing it to reproduce the simulated mass-loss rates across the explored parameter space. The calibration set spans a broad range of physically plausible hot-planet escape regimes.

The model implementation and calibration are described in detail in Sethi et al. 2026 (in prep.)


