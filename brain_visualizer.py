import mne
import pyvistaqt as pv
from pyvistaqt import BackgroundPlotter
from helper import get_stc, load_result, ConvDip_ESI
import os
import argparse
import scipy.io as sio


def brain3d(s_pred, hemi):
    stc = get_stc()
    stc.data = s_pred
    data_path = mne.datasets.sample.data_path()
    # subjects_dir = os.path.join(data_path, 'subjects')
    subjects_dir = data_path / 'subjects'

    # Set SUBJECTS_DIR environment variable
    # os.environ["SUBJECTS_DIR"] = subjects_dir
    plotter = BackgroundPlotter()
    brain = mne.viz.plot_source_estimates(
        stc,
        views='lateral',
        hemi=hemi,
        surface='white',
        background='white',
        size=(1000, 400),
        subjects_dir=subjects_dir,
        # time_viewer=False,
        # show_traces=False,
        # colorbar=False
        # figure=plotter
    )
    plotter.app.exec_()

    plotter.close()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Brain Visualizer Script')

    # Add arguments
    parser.add_argument('--file', type=str, help='Path to the uploaded file')
    parser.add_argument('--newId', type=str,
                        help='New ID for the uploaded file')

    # Parse arguments
    args = parser.parse_args()
    task = 'LA'

    result_path = os.path.join("data", "uploads", args.newId)

    raw = mne.io.read_raw_fif(args.file, preload=True)
    l_freq, h_freq = 1, 30
    raw.filter(l_freq, h_freq, method='fir', fir_design='firwin')
    sfreq_resample = 480
    raw = raw.resample(sfreq_resample)
    events = mne.find_events(raw, stim_channel="STI 014")
    event_dict = {
        "LA": 1,
        "RA": 2,
        "LV": 3,
        "RV": 4,
    }
    event_id = 'LA'

    # set path to save data
    path = os.path.join(result_path, "sample_data")
    if not os.path.exists(path):
        os.makedirs(path)
    fig_name = os.path.join(path, 'evoked_eeg_'+str(event_id)+'.png')
    mat_name = os.path.join(path, 'evoked_eeg_'+str(event_id)+'.mat')

    tmin = -0.1  # start of each epoch (100ms before the event)
    tmax = 0.4  # end of each epoch (400ms after the event)
    raw.info['bads'] = ['MEG 2443', 'EEG 053']
    baseline = (None, 0)  # means from the first instant to t = 0
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    picks = mne.pick_types(raw.info, meg=True, eeg=True,
                           eog=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, proj=True,
                        picks=picks, baseline=baseline, reject=reject)
    epoch_use = epochs[event_id]
    evoked_use = epoch_use.average()

    fig = evoked_use.plot_topomap(
        times=[0.0, 0.08, 0.1, 0.12, 0.2], ch_type="eeg")
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')

    data = evoked_use.data[:, :]
    sio.savemat(mat_name, {'eeg': data})

    tasks = ['LA']  # or 'LA' or ['LA'], etc.
    # set your result path
    r_path = os.path.join(result_path, "result")
    ConvDip_ESI(tasks, result_path)

    s_pred = load_result(task, r_path)

    print(args.file)

    # Call the brain3d function with the provided arguments
    brain3d(s_pred, hemi='both')
