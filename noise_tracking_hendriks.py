# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg
import scipy.io

class NoiseTracking:

    class State:

        def __init__(self, frame : int = 512):

            self._frame_counter = 0

            self._noise_power = np.zeros((int(frame/2+1),), dtype = np.float32) 

            self._ph1_mean = np.ones((int(frame/2+1),), dtype = np.float32) * .5

    """
    Noise estimation routine. Algorithm taken from the papers:

    "Unbiased MMSE-Based Noise Power Estimation with Low Complexity and Low Tracking Delay", IEEE TASL, 2012 
    "Noise Power Estimation Based on the Probability of Speech Presence", Timo Gerkmann and Richard Hendriks, WASPAA 2011

    :param frame: - frame length in samples (typically 512 for 16kHz sampling frequency)

    Author: Marcin Kuropatwiński
    Created: 02.04.2014, Modified 08.11.2016

    """

    def __init__(self, frame : int = 512):

        self.state = NoiseTracking.State(frame = frame)

    def noisePowRunning(self, noisy_psd):

        """
        Real time function for computing noise power spectral density (psd) given instantaneous noisy speech psd.

        Input:

        noisyPer - noisy signal frame psd, vector of length frame/2+1 (for 16kHz sampling rate it should be 257
                   elements)


        Author: Marcin Kuropatwiński
        Created: 02.04.2014

        """


        #initialize for 0th frame

        state = self.state

        if state._frame_counter == 0:

            noise_power = noisy_psd

            self.state._frame_counter += 1

            self.state._noise_power = noisy_psd

            self.noise_power = noisy_psd

            #

            self.alphaPH1mean = 0.9

            self.alphaPSD = 0.8

            #constants for a posteriori SPP

            self.q = 0.5  # a-priori probability of speech presence

            self.priorFact = self.q / (1 - self.q)

            self.xiOptDb = 15.  # optimal fixed a priori SNR for SPP estimation

            self.xiOpt = 10 ** (self.xiOptDb / 10)

            self.logGLRFact = np.log(1 / (1 + self.xiOpt))

            self.GLRexp = self.xiOpt / (1 + self.xiOpt)

            return

        #initialize for first 5 frames

        if state._frame_counter < 5:  # first frame has frmCnt equal zero

            noise_power = .5 * (self.state._noise_power + noisy_psd)

            self.state._noise_power = noise_power

            self.state._frame_counter += 1

            self.noise_power = noise_power

            return

        ph1_mean = state._ph1_mean

        snrPost1 = noisy_psd / np.maximum(state._noise_power,1.e-5)  # a posteriori SNR based on old noise power estimate

        #noise power estimation

        GLR = self.priorFact * np.exp(np.minimum(self.logGLRFact + self.GLRexp * snrPost1, 50))

        PH1 = GLR / (1 + GLR) # a posteriori speech presence probability

        ph1_mean = self.alphaPH1mean * ph1_mean + (1 - self.alphaPH1mean) * PH1

        stuckInd = np.zeros_like(ph1_mean)

        stuckInd = ph1_mean > 0.99

        PH1[stuckInd] = np.minimum(PH1[stuckInd], 0.99)

        estimate = PH1 * state._noise_power + (1 - PH1) * noisy_psd

        noise_power = self.alphaPSD * state._noise_power+ (1 - self.alphaPSD) * estimate

        self.state._frame_counter += 1
        self.state._noise_power = noise_power
        self.state._ph1_mean = ph1_mean

        self.noise_power = noise_power


    def get_noise_psd(self):

        """

        Noise psd getter.

        :return: instantaneous noise power density spectrum

        """

        return self.noise_power

    def get_nsepow_state(self):

        return self.state

if __name__ == "__main__":

    mats = scipy.io.loadmat('./test-data/stencil.mat')

    nst = NoiseTracking(256)

    result = np.zeros_like(mats['noisy_psd'])

    for i in range(mats['noisy_psd'].shape[1]):

        nst.noisePowRunning(mats["noisy_psd"][:,i])

        result[:,i] = nst.get_noise_psd()

    if numpy.linalg.norm(mats['noise_psd'] - result, ord='fro') < 1.e-7:
        print("Test passed ...")
    else:
        print("Test failed ...")
