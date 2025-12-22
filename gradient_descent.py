# -*- coding: utf-8 -*-
"""
Gradient Descent Module for Presolar Grain Simulations

This module contains all functions and utilities related to gradient descent
optimization using the Adam-Nesterov algorithm.

Extracted from:
- PG_simulations_func.py: GD_AdamNesperov function
- PG_NanoSIMS_Simulations_v3.0.py: Parameter initialization, update logic,
  constraint application, and cost computation

Functions:
- GD_AdamNesperov: Core gradient descent computation
- initialize_gradient_descent_parameters: Initialize optimizer parameters
- update_gradient_descent_parameters: Update Adam-Nesterov parameters
- apply_simulation_constraints: Apply physical constraints to parameters
- compute_cost: Calculate cost function from normalized norms
- save_norm_to_summary: Save cost function evolution to DataFrame

@author: Maximilien Verdier-Paoletti
"""

import numpy as np
import pandas as pd


def GD_AdamNesperov(target, measured_simulations, initial_simulations, learning_rate):
    """
    Gradient Descent function using Adam-Nesterov optimizer.

    Computes the normalized 3D norm (cost function) and gradient for gradient descent
    optimization. The cost function is based on normalized absolute differences
    between target and measured values.

    Parameters
    ----------
    target : array-like
        Target values [size, ratio1, ratio2] to match
    measured_simulations : array-like
        Measured simulation results with shape (3, n_grains) where:
        - measured_simulations[0] = measured sizes
        - measured_simulations[1] = measured ratio1 values
        - measured_simulations[2] = measured ratio2 values
    initial_simulations : array-like
        Initial simulation parameters with shape (3, n_grains) where:
        - initial_simulations[0] = initial sizes
        - initial_simulations[1] = initial ratio1 values
        - initial_simulations[2] = initial ratio2 values
    learning_rate : array-like
        Learning rate matrix with shape (n_grains, 3) or compatible shape

    Returns
    -------
    new_simu : array
        Updated simulation parameters after gradient descent step
        Shape: (n_grains, 3) where columns are [size, ratio1, ratio2]
    norm3D_norm : array
        Normalized 3D norm (cost function) for each grain
    grad : array
        Gradient matrix with shape (3, n_grains) where rows are gradients
        for [size, ratio1, ratio2]
    """
    X = measured_simulations[0]
    Y = measured_simulations[1]
    Z = measured_simulations[2]

    # Compute normalized 3D norm (cost function)
    # Using normalized absolute differences
    norm3D_norm = np.asarray(
        (
            np.abs((X - target[0]) / target[0])
            + np.abs((Y - target[1]) / target[1])
            + np.abs((Z - target[2]) / target[2])
        )
        ** 0.5
    )

    # Compute gradient
    grad = np.array(
        [
            (X - target[0]) / (norm3D_norm * target[0] ** 2),
            (Y - target[1]) / (norm3D_norm * target[1] ** 2),
            (Z - target[2]) / (norm3D_norm * target[2] ** 2),
        ]
    )

    # Update parameters: new = old - learning_rate
    new_simu = initial_simulations.T - learning_rate

    return new_simu, norm3D_norm, grad


def initialize_gradient_descent_parameters(PG_size, PG_delta, Nb_PG, n_ratios=2):
    """
    Initialize gradient descent parameters for Adam-Nesterov optimizer.

    Parameters
    ----------
    PG_size : array-like
        Initial grain sizes
    PG_delta : array-like
        Initial grain delta values (composition)
    Nb_PG : int
        Number of presolar grains
    n_ratios : int, optional
        Number of isotopic ratios (default: 2)

    Returns
    -------
    eta : array
        Initial learning rate (base learning rate)
    learning_rate : array
        Current learning rate (initially same as eta)
    eps : float
        Small epsilon value for numerical stability (default: 1e-8)
    beta_decay : float
        Decay rate for second moment estimate (default: 0.9)
    beta_momentum : float
        Momentum coefficient (default: 0.6)
    decay_mat : array
        Unbiased decay matrix (second moment estimate)
    decay_adam : array
        Bias-corrected decay matrix
    momentum_mat : array
        Unbiased momentum matrix (first moment estimate)
    momentum_adam : array
        Bias-corrected momentum matrix
    """
    # Calculate initial learning rate (eta) based on parameter magnitudes
    eta = (
        10
        ** np.round(
            np.log10(
                np.abs(
                    np.concatenate(
                        (
                            PG_size.T,
                            np.array(PG_delta).reshape(Nb_PG, n_ratios),
                        ),
                        axis=1,
                    )
                )
            )
        )
        / 10
    )

    learning_rate = eta
    eps = 1e-8
    beta_decay = 0.9
    beta_momentum = 0.6

    # Initialize matrices: (n_parameters, n_grains)
    # Parameters: [size, ratio1, ratio2, ...]
    n_params = 1 + n_ratios  # size + ratios
    decay_mat = np.zeros((n_params, Nb_PG))
    decay_adam = np.zeros((n_params, Nb_PG))
    momentum_mat = np.zeros((n_params, Nb_PG))
    momentum_adam = np.zeros((n_params, Nb_PG))

    return (
        eta,
        learning_rate,
        eps,
        beta_decay,
        beta_momentum,
        decay_mat,
        decay_adam,
        momentum_mat,
        momentum_adam,
    )


def update_gradient_descent_parameters(
    grad,
    decay_mat,
    momentum_mat,
    eta,
    beta_decay,
    beta_momentum,
    eps,
    iteration,
):
    """
    Update gradient descent parameters using Adam-Nesterov algorithm.

    This function implements the Adam optimizer with Nesterov momentum.
    It updates the decay (second moment) and momentum (first moment) estimates,
    computes bias-corrected estimates, and calculates the adaptive learning rate.

    Parameters
    ----------
    grad : array
        Gradient matrix with shape (n_params, n_grains)
    decay_mat : array
        Current decay matrix (second moment estimate)
    momentum_mat : array
        Current momentum matrix (first moment estimate)
    eta : array
        Base learning rate
    beta_decay : float
        Decay rate for second moment (typically 0.9)
    beta_momentum : float
        Momentum coefficient (typically 0.6)
    eps : float
        Small epsilon for numerical stability
    iteration : int
        Current iteration number (0-indexed)

    Returns
    -------
    decay_mat : array
        Updated decay matrix
    decay_adam : array
        Bias-corrected decay matrix
    momentum_mat : array
        Updated momentum matrix
    momentum_adam : array
        Bias-corrected momentum matrix
    momentum_nesperov_adam : array
        Nesterov momentum (lookahead momentum)
    learning_rate : array
        Adaptive learning rate for next iteration
    """
    # Update decay (second moment estimate)
    decay_mat = decay_mat * beta_decay + (1 - beta_decay) * grad**2

    # Bias correction for decay
    decay_adam = decay_mat / (1 - beta_decay ** (iteration + 1))

    # Update momentum (first moment estimate)
    momentum_mat = beta_momentum * momentum_mat + (1 - beta_momentum) * grad

    # Bias correction for momentum
    momentum_adam = momentum_mat / (1 - beta_momentum ** (iteration + 1))

    # Nesterov momentum (lookahead)
    momentum_nesperov_adam = beta_momentum * momentum_adam + (1 - beta_momentum) * grad

    # Adaptive learning rate for Adam-Nesterov
    learning_rate = (eta.T * momentum_nesperov_adam / (decay_adam + eps) ** 0.5).T

    return (
        decay_mat,
        decay_adam,
        momentum_mat,
        momentum_adam,
        momentum_nesperov_adam,
        learning_rate,
    )


def apply_simulation_constraints(
    new_simu, min_size=50, default_size=100, min_delta=-1000, default_delta=-999
):
    """
    Apply physical constraints to updated simulation parameters.

    Ensures that simulation parameters remain within physically reasonable bounds:
    - Grain sizes must be above a minimum threshold
    - Delta values must be above a minimum threshold

    Parameters
    ----------
    new_simu : array
        Updated simulation parameters with shape (n_grains, n_params)
        where columns are [size, ratio1, ratio2, ...]
    min_size : float, optional
        Minimum allowed grain size (default: 50)
    default_size : float, optional
        Default size to use if below minimum (default: 100)
    min_delta : float, optional
        Minimum allowed delta value (default: -1000)
    default_delta : float, optional
        Default delta to use if below minimum (default: -999)

    Returns
    -------
    new_simu : array
        Constrained simulation parameters
    """
    # Constrain size (first column)
    new_simu.T[0] = np.where(new_simu.T[0] < min_size, default_size, new_simu.T[0])

    # Constrain delta values (remaining columns)
    new_simu.T[1::] = np.where(
        new_simu.T[1::] <= min_delta, default_delta, new_simu.T[1::]
    )

    return new_simu


def compute_cost(norm_summary, sim_selgrain, k, nb_closest_match=3):
    """
    Compute the cost function from normalized norms.

    The cost is the mean of the nb_closest_match smallest norms from
    the current outer iteration.

    Parameters
    ----------
    norm_summary : pandas.DataFrame
        DataFrame containing normalized norms with columns ['Norm']
    sim_selgrain : pandas.DataFrame
        DataFrame containing simulation results for the current grain
    k : int
        Current outer iteration number
    nb_closest_match : int, optional
        Number of closest matches to use for cost calculation (default: 3)

    Returns
    -------
    cost : float
        Mean cost (normalized norm) of the nb_closest_match best matches
    """
    norm_selgrain = norm_summary.iloc[sim_selgrain.index]
    norm = norm_selgrain.loc[
        sim_selgrain.loc[sim_selgrain["Outer Iteration"] == k].index
    ]
    cost = (
        norm.sort_values(by="Norm", ascending=True)[0:nb_closest_match][
            0:nb_closest_match
        ]
        .mean()
        .item()
    )
    return cost


def save_norm_to_summary(norm3D, norm_summary=None):
    """
    Save normalized norms to summary DataFrame.

    Parameters
    ----------
    norm3D : array
        Array of normalized norms for current iteration
    norm_summary : pandas.DataFrame, optional
        Existing summary DataFrame to append to. If None, creates new one.

    Returns
    -------
    norm_summary : pandas.DataFrame
        Updated summary DataFrame with new norms appended
    """
    if norm_summary is None:
        norm_summary = pd.DataFrame(data=norm3D, columns=["Norm"])
    else:
        norm_summary = pd.concat(
            [norm_summary, pd.DataFrame(norm3D, columns=["Norm"])],
            axis=0,
            ignore_index=True,
        )
    return norm_summary
