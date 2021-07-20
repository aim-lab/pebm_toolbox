
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt
from scipy.spatial import cKDTree
import os
import wfdb
from wfdb import processing
import tempfile


class Preprocessing:


    def __init__(self, signal, fs):
        """
        The purpose of the Preprocessing class is to prepare the ECG signal for the analysis.
        The class contains functions to filter the ECG signal from artifacts,
        the function to estimate the quality of the ECG signal -bsqi,
        and epltd peak detector.

        :param signal: the ECG signal as a ndarray.
        :param fs: the frequency of the signal.
        """
        self.signal = signal
        self.fs = fs
        self.notch_freq = None #can be 60 or 50 HZ



    def notch(self, notch_freq):

        """
        The notch function applies a notch filter in order to remove the power line artifacts.
        :param notch_freq: The frequency of the power line in the country where the signal was captured,
        usually the frequency is 50Hz (EUR) or 60Hz (US).
        :return: the filtered ECG signal
        """

        signal = self.signal
        fs = self.fs
        self.notch_freq = notch_freq
        # notch_freq have to be 50 or 60 HZ (make that condition)
        fsig = mne.filter.notch_filter(signal.astype(np.float), fs, freqs=notch_freq)

        #plot:

        self.signal =fsig
        return fsig


    def bpfilt(self):
        """
        The bpfilt function applies a bandpass filter between [0.67, 100] Hz,
        this function uses a zero-phase Butterwarth filter with 75 coefficients.
        :return:
        """
        signal = self.signal
        fs = self.fs
        filter_order = 75 #??
        low_cut = 0.67
        high_cut = 100

        nyquist_freq = 0.5 * fs
        low = low_cut / nyquist_freq
        high = high_cut / nyquist_freq
        if fs <= high_cut*2:
            sos = butter(filter_order, low, btype="high", output='sos', analog=False)
        else:
            sos = butter(filter_order, [low, high], btype="band", output='sos', analog=False)
        fsig = sosfiltfilt(sos, signal)
        self.signal = fsig
        return fsig


    def epltd(self):
        """
        This function calculates the indexes of the R-peaks with epltd peak detector.
        :return: indexes of the R-peaks in the ECG signal.
        """
        cwd = os.getcwd()
        #pad peaks:
        five_sec = 5*self.fs
        pad = self.signal[0:five_sec]
        signal_pad = np.concatenate((pad,self.signal))

        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            wfdb.wrsamp(record_name= 'temp', fs=np.asscalar(self.fs), units=['mV'], sig_name=['V5'], p_signal=signal_pad.reshape(-1, 1), fmt = ['16'] )
            prog_dir = '/home/sheina/pebm_toolbox/src/pebm_pkg/c_files/epltd_all' # take it from const
            ecg_dir = tmpdirname
            command = ';'.join(['EPLTD_PROG_DIR=' + prog_dir,
                                'ECG_DIR=' + ecg_dir,
                                'cd $ECG_DIR',
                                'command=\"$EPLTD_PROG_DIR -r ' + str('temp') + '\"',
                                'eval $command'])
            if os.name == 'nt':
                command = 'wsl ' + command
            os.system(command)
            peaks= wfdb.rdann('temp', 'epltd0').sample -five_sec
        os.chdir(cwd)

        return peaks

    def xqrs(self):
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            wfdb.wrsamp(record_name= 'temp', fs=np.asscalar(self.fs), units=['mV'], sig_name=['V5'], p_signal=self.signal.reshape(-1, 1), fmt = ['16'] )
            record = wfdb.rdrecord(tmpdirname+'/temp')
            fs = self.fs
            ecg = record.p_signal[:, 0]
            xqrs = processing.XQRS(ecg, fs)

            xqrs.detect()
            peaks = xqrs.qrs_inds
        os.chdir(cwd)
        return peaks

    def bsqi(self, peaks = None):

        """
        This function is based on the following paper:
            Li, Qiao, Roger G. Mark, and Gari D. Clifford.
            "Robust heart rate estimation from multiple asynchronous noisy sources
            using signal quality indices and a Kalman filter."
            Physiological measurement 29.1 (2007): 15.

        The implementation itself is based on:
            Behar, J., Oster, J., Li, Q., & Clifford, G. D. (2013).
            ECG signal quality during arrhythmia and its application to false alarm reduction.
            IEEE transactions on biomedical engineering, 60(6), 1660-1666.

        :param peaks:  Annotation of the reference peak detector (Indices of the peaks). If peaks are not given,
         the peaks are calculated with epltd detector, the test peaks are calculated with xqrs detector.
        :returns F1:    The 'bsqi' score, between 0 and 1.
        """

        fs = self.fs
        agw = 0.05 #in seconds
        if peaks is None:
            refqrs = self.epltd()
        else:
            refqrs = peaks
        testqrs = self.xqrs()
        agw *= fs
        if len(refqrs) > 0 and len(testqrs) > 0:
            NB_REF = len(refqrs)
            NB_TEST = len(testqrs)

            tree = cKDTree(refqrs.reshape(-1, 1))
            Dist, IndMatch = tree.query(testqrs.reshape(-1, 1))
            IndMatchInWindow = IndMatch[Dist < agw]
            NB_MATCH_UNIQUE = len(np.unique(IndMatchInWindow))
            TP = NB_MATCH_UNIQUE
            FN = NB_REF - TP
            FP = NB_TEST - TP
            Se = TP / (TP + FN)
            PPV = TP / (FP + TP)
            if (Se + PPV) > 0:
                F1 = 2 * Se * PPV / (Se + PPV)
                _, ind_plop = np.unique(IndMatchInWindow, return_index=True)
                Dist_thres = np.where(Dist < agw)[0]
                meanDist = np.mean(Dist[Dist_thres[ind_plop]]) / fs
            else:
                return 0

        else:
            F1 = 0
            IndMatch = []
            meanDist = fs
        return F1