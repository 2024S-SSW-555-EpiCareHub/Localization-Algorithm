
import mne
from mne.datasets import sample
from mne.beamformer import make_lcmv, apply_lcmv
import scipy.io as sio
import numpy as np


def get_stc():
    data_path = sample.data_path()
    subjects_dir = data_path / 'subjects'
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'

    # Read the raw data
    raw = mne.io.read_raw_fif(raw_fname)
    raw.info['bads'] = ['MEG 2443']  # bad MEG channel

    # Set up the epoching
    event_id = 1  # those are the trials with left-ear auditory stimuli
    tmin, tmax = -0.2, 0.5
    events = mne.find_events(raw)

    # pick relevant channels
    raw.pick(['meg', 'eog'])  # pick channels of interest

    # Create epochs
    proj = False  # already applied
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        baseline=(None, 0), preload=True, proj=proj,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

    # for speed purposes, cut to a window of interest
    evoked = epochs.average().crop(0.05, 0.15)

    # Visualize averaged sensor space data
    # evoked.plot_joint()

    del raw  # save memory

    data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.25,
                                      method='empirical')
    noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0,
                                       method='empirical')
    # data_cov.plot(epochs.info)
    del epochs

    # Read forward model
    fwd_fname = './data/meg-fwd.fif'
    forward = mne.read_forward_solution(fwd_fname)

    filters = make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='unit-noise-gain', rank=None)

    filters_vec = make_lcmv(evoked.info, forward, data_cov, reg=0.05,
                            noise_cov=noise_cov, pick_ori='vector',
                            weight_norm='unit-noise-gain-invariant', rank=None)

    # save a bit of memory
    src = forward['src']
    del forward

    stc = apply_lcmv(evoked, filters)
    stc_vec = apply_lcmv(evoked, filters_vec)
    del filters, filters_vec

    return stc


def load_result(task, result_path):
    if not isinstance(task, str):
        raise Exception(
            "Oops! That was not a valid type for task. Try type a 'str'!")

    if not task in ['LA', 'RA', 'LV', 'RV']:
        raise Exception(
            "Oops! That was not a valid task. Try use 'LA', 'RA', 'LV' or 'RV'!")
    else:
        print("load result for task: {}".format(task))
        fname = result_path + \
            '/sample/Test_result_evoked_' + str(task) + '.mat'
        dataset = sio.loadmat(fname)
        s_pred = dataset['s_pred']
        s_pred = np.absolute(s_pred)
        if s_pred.shape[0] != 1984:
            s_pred = s_pred.T

    return s_pred
