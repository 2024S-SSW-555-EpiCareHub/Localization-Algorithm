import mne
import pyvistaqt as pv
from pyvistaqt import BackgroundPlotter
from helper import get_stc, load_result
import os


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


task = 'LA'
result_path = './result/'
s_pred = load_result(task, result_path)
# Example usage: call the brain3d function with appropriate data
brain3d(s_pred, hemi='both')  # Replace s_pred with actual data
