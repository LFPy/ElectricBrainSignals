#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Demonstrate usage of LFPy.Network with network of ball-and-stick type
morphologies with active HH channels inserted in the somas and passive-leak
channels distributed throughout the apical dendrite. The corresponding
morphology and template specifications are in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

Execution (w. MPI):

    mpirun -np 2 python example_network.py

Copyright (C) 2017 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""
# import modules:
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import h5py
import json
import hashlib
from parameters import ParameterSet, ParameterSpace
from plotting import draw_lineplot
import example_network_parameters as params
import example_network_methods as methods


# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED)


class ISynApprox(object):
    '''Class for computing linear spike-to-isyn filter kernels resulting
    from presynaptic spiking activity and resulting postsynaptic currents

    Parameters
    ----------
    X: list of str
        presynaptic populations
    Y: str
        postsynaptic population
    N_X: array of int
        presynaptic population sizes
    N_Y: int
        postsynaptic population size
    C_YX: array of float
        pairwise connection probabilities betwen populations X and Y
    multapseParameters: list of dict
        kwargs for np.random.normal which will be treated as a one-sided
        truncated normal distribution for determining number of synapse
        instances per connection between populations X and Y
    delayFunction: callable
        scipy.stats callable with pdf method
    delayParameters: dict
        kwargs for delayFunction
    synapseParameters: list of dict
        kwargs for LFPy.Synapse, assuming conductance based synapse which
        will be linearized to current based synapse for connections between
        populations X and Y
    nu_X: dict of floats
        presynaptic population rates (1/s)
    '''
    def __init__(
            self,
            X=['E'],
            Y='E',
            N_X=np.array([1024]),
            N_Y=1024,
            C_YX=np.array([0.1]),
            multapseParameters=[dict(loc=2, scale=5)],
            delayFunction=st.truncnorm,
            delayParameters={'a': -4.0, 'b': np.inf, 'loc': 1.5, 'scale': 0.3},
            synapseParameters=[dict(weight=0.001, syntype='Exp2Syn',
                                    tau1=0.2, tau2=1.8, e=0.)],
            nu_X=dict(E=1.)
    ):
        # set attributes
        self.X = X
        self.Y = Y
        self.N_X = N_X
        self.N_Y = N_Y
        self.C_YX = C_YX
        self.multapseParameters = multapseParameters
        self.delayFunction = delayFunction
        self.delayParameters = delayParameters
        self.synapseParameters = synapseParameters
        self.nu_X = nu_X

    def get_delay(self, dt, tau):
        '''Get normalized transfer function for conduction delay distribution
        for connections between population X and Y

        Parameters
        ----------
        dt: float
        tau: float

        Returns
        -------
        h_delta: ndarray
            shape (2 * tau // dt + 1) array with transfer function for delay
            distribution
        '''
        t = np.linspace(-tau, tau, int(2 * tau // dt + 1))
        h_delay = self.delayFunction(**self.delayParameters).pdf(t)
        return h_delay / h_delay.sum()

    def get_kernel(self, Vrest=-65, dt=2**-4,
                   X='E', tau=50,
                   ):
        '''Compute linear spike-to-signal filter kernel mapping presynaptic
        population firing rates/spike trains to signal measurement, e.g., LFP.

        Parameters
        ----------
        Vrest: float
            Mean/Expectation value of postsynaptic membrane voltage used
            for linearization of synapse conductances
        dt: float
            temporal resolution (ms)
        X: str
            presynaptic population for kernel, must be element in
            `<ISynApprox instance>.X`
        tau: float
            half-duration of filter kernel -- full duration is (2 * tau + dt)

        Returns
        -------
        H_YX: dict of ndarray
            shape (2 * tau // dt + 1) linear response kernel
        '''

        # assess index of presynaptic population in X
        (X_i, ) = np.where(np.array(self.X) == X)[0]

        # estimate number of connections as in Potjans&Diesmann2014
        # K_YX = np.log(1. - C_YX) / np.log(1. - 1. / (N_X * N_Y))
        # accurate for small K/(N_X N_Y):
        K_YX = self.C_YX * self.N_X * self.N_Y

        # account for one-sided truncated distribution of synapses per
        # connection with expectation 'loc' and standard deviation 'scale'
        a = 1  # lower bound for distribution (# of synapses >= 1!)
        for i in range(len(self.X)):
            kwargs = dict(
                a=((a - self.multapseParameters[i]['loc'])
                    / self.multapseParameters[i]['scale']),
                b=np.inf,
                loc=self.multapseParameters[i]['loc'],
                scale=self.multapseParameters[i]['scale'],
            )
            multapsedist = st.truncnorm(**kwargs)

            # total number of connections
            K_YX[i] = K_YX[i] * multapsedist.mean()

        def f_exp2(t, tau1, tau2):
            '''2-exponential temporal kernel function

            Parameters
            ----------

            '''
            t_peak = (tau1 * tau2) / (tau2 - tau1) * np.log(tau2 / tau1)
            factor = np.exp(- t_peak / tau2) - np.exp(- t_peak / tau1)
            f_beta = (np.exp(-t / tau2) - np.exp(-t / tau1)) / factor
            return f_beta

        # iterate over all presynaptic populations in order to offset g_L
        # correctly
        for iii in range(len(self.X)):
            if iii == X_i:
                # modify synapse parameters to account for current-based
                # synapses linearized around Vrest and
                # scaled by total # synapses
                d = self.synapseParameters[iii].copy()
                weight = - d['weight'] * (Vrest - d['e']) * K_YX[iii]
                del d['e']  # no longer needed
                del d['syntype']

                # time vector
                t = np.arange(tau // dt + 1) * dt

                h_isyn = np.r_[np.zeros(int(tau // dt)),
                               f_exp2(t, d['tau1'], d['tau2'])]
                h_isyn *= weight

        # get conduction delay transfer function for connections from X to Y
        h_delta = self.get_delay(dt, tau)

        # filter kernel by normalized delay distribution function
        H_YX = np.convolve(h_isyn, h_delta, 'same')

        return H_YX


if __name__ == '__main__':
    ##########################################################################
    # Main simulation
    ##########################################################################

    #######################################
    # Capture command line values
    #######################################

    # fetch parameter values
    weight_EE = float(sys.argv[1])
    weight_EI = float(sys.argv[2])
    weight_IE = float(sys.argv[3])
    weight_II = float(sys.argv[4])
    tau = 50  # time lag around t_X
    dt = params.networkParameters['dt']
    TRANSIENT = 200

    ##########################################################################
    # Set up shared and population-specific parameters
    ##########################################################################
    # relative path for simulation output:
    PS0 = ParameterSpace('PS0.txt')
    pset_0 = ParameterSet(dict(weightE=weightE,
                               weightI=weightI,
                               n_ext=PS0['n_ext']))
    js_0 = json.dumps(pset_0, sort_keys=True).encode()
    md5_0 = hashlib.md5(js_0).hexdigest()
    OUTPUTPATH_REAL = os.path.join('output', md5_0)

    ##########################################################################
    # Parameters dependent on command line input
    ##########################################################################
    # synapse max. conductance (function, mean, st.dev., min.):
    weights_YX = [[weight_EE, weight_EI],
                  [weight_IE, weight_II]]

    # Compute average firing rate of presynaptic populations X
    nu_X = methods.compute_nu_X(params, OUTPUTPATH_REAL,
                                TRANSIENT=TRANSIENT)

    # conduction delay function
    if params.delayFunction == np.random.normal:
        delayFunction = st.truncnorm
    else:
        raise NotImplementedError

    ##########################################################################
    # Compute kernels
    ##########################################################################
    # kernel container
    H_YX = dict()

    # iterate over pre and postsynaptic units
    for i, (X, N_X) in enumerate(zip(params.population_names,
                                     params.population_sizes)):

        for j, (Y, N_Y) in enumerate(zip(params.population_names,
                                         params.population_sizes)):
            # Extract median soma voltages from actual network simulation and
            # assume this value corresponds to Vrest.
            with h5py.File(os.path.join(OUTPUTPATH_REAL, 'somav.h5'
                                        ), 'r') as f:
                Vrest = np.median(f[Y][()][:, 200:])

            # some inputs must be lists
            multapseParameters = [
                dict(loc=params.multapseArguments[ii][j]['loc'],
                     scale=params.multapseArguments[ii][j]['scale'])
                for ii in range(len(params.population_names))]
            synapseParameters = [
                dict(weight=weights_YX[ii][j],
                     syntype='Exp2Syn',
                     **params.synapseParameters[ii][j])
                for ii in range(len(params.population_names))]

            # Create kernel approximator object
            kernel = ISynApprox(
                X=params.population_names,
                Y=Y,
                N_X=np.array(params.population_sizes),
                N_Y=N_Y,
                C_YX=np.array(params.connectionProbability[i]),
                multapseParameters=multapseParameters,
                delayFunction=delayFunction,
                delayParameters=dict(
                    a=((params.mindelay - params.delayArguments[i][j]['loc'])
                       / params.delayArguments[i][j]['scale']),
                    b=np.inf,
                    **params.delayArguments[i][j]),
                synapseParameters=synapseParameters,
                nu_X=nu_X,
            )

            H_YX['{}:{}'.format(Y, X)] = kernel.get_kernel(
                Vrest=Vrest, dt=dt, X=X, tau=tau,
            )

    ##########################################################################
    # plot kernels
    ##########################################################################
    t = np.linspace(-tau, tau, int(2 * tau // dt + 1))

    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(16, 9))
    for i, X in enumerate(params.population_names):
        for j, Y in enumerate(params.population_names):
            unit = 'nA'
            title = r'$\hat{H}_\mathrm{%s %s}(\mathbf{r}, \tau)$' % (Y, X)

            ax = axes[i * 2 + j]

            draw_lineplot(
                ax,
                H_YX['{}:{}'.format(Y, X)][int(tau // dt):].reshape((1, -1)),
                dt=dt,
                T=(0, tau),
                scaling_factor=1.,
                label='ISyn',
                scalebar=True,
                unit=unit,
                ylabels=False,
                color='k',
                ztransform=False
            )
            ax.set_title(title)
            if (i * 2 + j) == 0:
                ax.set_ylabel('ISyn')
            else:
                ax.set_ylabel('')
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_xlabel(r'$\tau$ (ms)')
    axes[0].legend(loc=1)

    plt.show()
