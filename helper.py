
import mne
from mne.beamformer import make_lcmv, apply_lcmv
import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import h5py
from pyvistaqt import BackgroundPlotter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


event_dict = {
    "LA": 1,
    "RA": 2,
    "LV": 3,
    "RV": 4,
}


class RandomDatasetSingle(Dataset):
    def __init__(self, x_data, y_data, length):
        self.x_data = x_data.reshape(
            (x_data.shape[0], 1, x_data.shape[1], x_data.shape[2]))
        self.y_data = y_data
        self.len = length

    def __getitem__(self, index):
        x_batch = torch.Tensor(self.x_data[index, :, :, :]).float()
        y_batch = torch.Tensor(self.y_data[index, :]).float()
        return x_batch, y_batch, index

    def __len__(self):
        return self.len


class ConvDipSingleCatAtt(torch.nn.Module):
    def __init__(self, channel_num=1, output_num=1984, conv1_num=8, fc1_num=792, fc2_num=500):
        super(ConvDipSingleCatAtt, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(channel_num, conv1_num, kernel_size=3,
                            stride=1, dilation=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        # Squeeze-and-Excitation module
        self.SE = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 8),
            torch.nn.Sigmoid(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(fc1_num, fc2_num, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(fc2_num, output_num, bias=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), out.size(1), -1)
        squeeze = torch.mean(out, dim=2)
        excitation = self.SE(squeeze)
        excitation = excitation.unsqueeze(2)
        scale_out = torch.mul(out, excitation)
        flat_out = out.view(scale_out.size(0), -1)
        final_out = self.classifier(flat_out)
        return final_out


def get_stc(file):
    # Read the raw data
    raw = mne.io.read_raw_fif(file)
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

        fname = os.path.join(
            result_path, 'Test_EEG_' + str(task) + '.mat')
        dataset = sio.loadmat(fname)
        s_pred = dataset['s_pred']
        s_pred = np.absolute(s_pred)
        if s_pred.shape[0] != 1984:
            s_pred = s_pred.T

    return s_pred


def max_min_normalize(data):
    data_min = np.min(data, axis=1)
    data_max = np.max(data, axis=1)
    data_min = np.expand_dims(data_min, axis=1)
    data_max = np.expand_dims(data_max, axis=1)

    data_min = np.tile(data_min, (1, data.shape[1]))
    data_max = np.tile(data_max, (1, data.shape[1]))

    # data_normalized = (data - data_min) / (data_max - data_min)
    data_normalized = np.divide(data - data_min, data_max - data_min)
    return data_normalized


def input_reshape(data, fname):
    """
    change vector of EEG/MEG data to matrix.
    for ConvDip input
    """
    # load map...................
    matfile = h5py.File(fname)
    maptable = matfile['maptable'][()].T
    x = max(maptable[:, 0]) + 1
    y = max(maptable[:, 1]) + 1
    data_num = data.shape[0]
    temp_matrix = np.zeros((data_num, int(x), int(y)))
    for index in range(maptable.shape[0]):
        i = maptable[index, 0]
        j = maptable[index, 1]
        value = maptable[index, 2]
        temp_matrix[:, int(i), int(j)] = data[:, int(value)]
    return temp_matrix


def ConvDip_ESI(task_id, path):
    """
    EEG source imaging with ConvDip framework
    task_id: str or list ['LA', 'RA', 'LV', 'RV']
    result_path: path to model output
    """
    data_name = 'sample'
    map_dir = './data/eeg_maptable.mat'

    model_flag = 'real_model'
    model_dir = './model/' + data_name + '/' + model_flag

    test_data = os.path.join(path, "data")

    result_path = os.path.join(path, "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # data: EEG/MEG & source
    # size: (dim, nsample)

    if isinstance(task_id, str):
        task_set = [task_id]
    elif isinstance(task_id, list):
        task_set = task_id
    else:
        raise Exception(
            "Oops! That was not a valid type for task id. Try use 'str' or 'list'!")

    for run in task_set:
        print('processing task:', str(run))

        if not run in ['LA', 'RA', 'LV', 'RV']:
            raise Exception(
                "Oops! That was not a valid task id. Try use 'LA', 'RA', 'LV' or 'RV'!")
        else:
            data_mat = test_data + '/EEG_' + str(run) + '.mat'
            result_mat = result_path + '/Test_EEG_' + str(run) + '.mat'

            # load the real dataset
            dataset = sio.loadmat(data_mat)
            test_input = dataset['eeg'].T

            test_input = max_min_normalize(test_input)

            # change to [timepoint, 12, 14]:
            test_input_matrix = input_reshape(test_input, map_dir)

            # get the number of samples:
            ntest = test_input_matrix.shape[0]
            test_output = test_input

            # change to [timepoint, 1, 12, 14]:
            RandomDataset_test = RandomDatasetSingle(
                test_input_matrix, test_output, ntest)
            rand_loader_test = DataLoader(dataset=RandomDataset_test, batch_size=ntest, num_workers=0,
                                          shuffle=False)

            # Build model: Model_EEG
            model = ConvDipSingleCatAtt()

            model = model.to(device)
            # load pretrained model...
            model.load_state_dict(torch.load(
                '%s/net_params_best.pkl' % (model_dir)))
            # model prediction...
            model.eval()
            with torch.no_grad():
                for data in rand_loader_test:
                    batch_X_eeg, _, _ = data
                    batch_X_eeg = batch_X_eeg.to(device)

                    X_eeg = Variable(batch_X_eeg)

                    # forward propagation
                    Y_pred = model(X_eeg)
                    Y_pred = Y_pred.cpu().detach().numpy().T

            # ================== save test result: ==================
            sio.savemat(result_mat, {'s_pred': Y_pred})
            Y_pred = np.absolute(Y_pred)
            if Y_pred.shape[0] != 1984:
                Y_pred = Y_pred.T
            return Y_pred
            # print("======== test finished!! ========")


def data_preprocessing(file):
    raw = mne.io.read_raw_fif(file, preload=True)
    l_freq, h_freq = 1, 30
    raw.filter(l_freq, h_freq, method='fir', fir_design='firwin')
    sfreq_resample = 480
    raw = raw.resample(sfreq_resample)
    events = mne.find_events(raw, stim_channel="STI 014")
    return raw, events


def save_evoked_data(file, event, path):
    raw, events = data_preprocessing(file)

    fig_path = os.path.join(path, "figures")
    data_path = os.path.join(path, "data")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    fig_name = os.path.join(fig_path, 'EEG_'+str(event)+'.png')
    mat_name = os.path.join(data_path, 'EEG_'+str(event)+'.mat')
    tmin = -0.1  # start of each epoch (100ms before the event)
    tmax = 0.4  # end of each epoch (400ms after the event)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']
    baseline = (None, 0)  # means from the first instant to t = 0
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    picks = mne.pick_types(raw.info, meg=True, eeg=True,
                           eog=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True,
                        picks=picks, baseline=baseline, reject=reject)
    epoch_use = epochs[event]
    evoked_use = epoch_use.average()

    # evoked_use_path = os.path.join(data_path, 'evoked_use.npy')
    # np.save(evoked_use_path, evoked_use)

    data = evoked_use.data[:, :]
    sio.savemat(mat_name, {'eeg': data})
    return raw, events, evoked_use, fig_name


def brain3d(file, s_pred, directory, hemi):
    stc = get_stc(file)
    stc.data = s_pred
    data_path = mne.datasets.sample.data_path()
    subjects_dir = data_path / 'subjects'

    plotter = BackgroundPlotter()
    brain = mne.viz.plot_source_estimates(
        stc,
        views='lateral',
        hemi=hemi,
        surface='white',
        background='white',
        size=(1000, 400),
        subjects_dir=subjects_dir,
    )
    views = ['medial', 'rostral', 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal', 'axial', 'sagittal', 'coronal', 'lateral']
    for view in views:
        view_filename = f'{view}.png'
        view_path = os.path.join(directory, 'figures', view_filename)
        brain.show_view(view)
        brain.save_image(view_path)
    plotter.app.exec_()
