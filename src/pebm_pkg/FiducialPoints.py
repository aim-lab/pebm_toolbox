from pebm_pkg.Preprocessing  import *
import tempfile
import platform
import scipy.io as spio

class FiducialPoints:

    def __init__(self, signal, fs, peaks= None):
        """
        The purpose of the FiducialPoints class is to calculate the fiducial pointes of the ECG signal.
        :param signal: the ECG signal as a ndarray.
        :param fs: the frequency of the signal.
        :param peaks:the indexes of the R- points of the ECG signal – optional input
        """
        self.signal = signal
        self.fs = fs
        if peaks is None:
            pre = Preprocessing(signal, fs)
            peaks = pre.epltd()
        self.peaks = peaks

    def wavedet(self, matlab_pat = None):
        """
        The wavedat function uses the matlab algorithm wavedet, compiled for python.
        The algorithm is described in the following paper:
        Martinze at el (2004),
        A wavelet-based ECG delineator: evaluation on standard databases.
        IEEE Transactions on Biomedical Engineering, 51(4), 570-581.

        :param matlab_pat: optional input- needed to use a linux machine
        :return: dictionary that includes indexes for each fiducial point
        """

        signal = self.signal
        fs = self.fs
        peaks = self.peaks
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
             # take it from const
            ecg_kit = '/home/sheina/pebm_toolbox/src/pebm_pkg/wavedet_exe/ecg-kit-master'
            #matlab_pat = '/usr/local/MATLAB/MATLAB_Runtime'

            np.savetxt("peaks.txt", peaks)
            np.savetxt("signal.txt", signal)
            if platform.system() == 'Linux':
                set_command = ''.join(['export MATLAB_RUNTIME=', matlab_pat])
                wavedet_dir = '/home/sheina/pebm_toolbox/src/pebm_pkg/wavedet_exe/run_peak_det.sh'
                command = ' '.join([wavedet_dir, ecg_kit, '"signal.txt" "peaks.txt" "200" '] )
                all_command =';'.join([set_command, command])
                os.system(all_command)
            if platform.system() == 'Windows':
                wavedet_dir = '/home/sheina/pebm_toolbox/src/pebm_pkg/wavedet_exe/peak_det.exe'
                os.system('wavedet_dir "signal.txt" "peaks.txt" "200"')
            fiducials_mat = spio.loadmat('output.mat')
        os.chdir(cwd)
        keys = ["Pon", "P", "Poff", "QRSon", "Q", "qrs", "S", "QRSoff", "Ton", "T", "Toff", "Ttipo", "Ttipoon",
                "Ttipooff"]
        position = fiducials_mat['output'][0, 0]
        all_keys = fiducials_mat['output'].dtype.names
        position_values = []
        position_keys = []
        for i, key in enumerate(all_keys):
            ret_val = position[i].squeeze()
            if (keys.__contains__(key)):
                ret_val[np.isnan(ret_val)] = -1
                ret_val = np.asarray(ret_val, dtype=np.int64)
                position_values.append(ret_val.astype(int))
                position_keys.append(key)
        # -----------------------------------

        fiducials = dict(zip(position_keys, position_values))

        return fiducials

