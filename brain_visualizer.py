
from helper import save_evoked_data, data_preprocessing, load_result, ConvDip_ESI, brain3d, brain3dOnlyVisualize
import os
import argparse
import json
import cloudinary
from cloudinary import uploader


cloudinary.config(
    cloud_name="damd1pa4a",
    api_key="419822137981766",
    api_secret="MnULqJOhq64i6VWpfoTCCd_-82c"
)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Brain Visualizer Script')

    # Add arguments
    parser.add_argument('--historic', type=str,
                        help='Path to the uploaded file')

    args, remaining_args = parser.parse_known_args()

    historic_bool = args.historic.lower() == "true"
    if historic_bool:
        parser.add_argument('--upload_dir', type=str,
                            help='For historic file directory')
    else:
        parser.add_argument('--basePath', type=str,
                            help='Root Path to the upload directory')
        parser.add_argument('--file', type=str,
                            help='Path to the uploaded file')
        parser.add_argument('--patientId', type=str,
                            help='Patient ID of the uploaded file')
        parser.add_argument('--uploadId', type=str,
                            help='New ID for the uploaded file')

    # Parse arguments
    args = parser.parse_args(remaining_args)
    event_id = "LA"
    if historic_bool:
        result_path = os.path.join(args.upload_dir, "result")
        s_pred = load_result(event_id, result_path)
        files_in_folder = os.listdir(args.upload_dir)

        fif_file_path = None

        for file in files_in_folder:
            if file.endswith('.fif'):
                fif_file_path = os.path.join(args.upload_dir, file)
                break
        if fif_file_path is None:
            raise FileNotFoundError("No .fif file found in the folder.")

        raw, events = data_preprocessing(fif_file_path)

        fig = raw.plot(
            events=events,
            start=5,
            duration=10,
            color="gray",
            event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y",
                         32: "k"},  # set color according to events id
        )

        brain3dOnlyVisualize(fif_file_path, s_pred,
                             args.upload_dir, hemi='both')
    else:
        root_path = os.path.join(args.basePath, args.uploadId)

        # set path to save data
        raw, events, evoked_use, fig_name, figure_url, mat_url = save_evoked_data(args.uploadId,
                                                                                  args.file, event_id, root_path)

        # set your result path
        result_path = os.path.join(root_path, "result")
        s_pred = ConvDip_ESI(event_id, root_path)

        # s_pred = load_result(event_id, result_path)

        # Call the brain3d function with the provided arguments

        fig = raw.plot(
            events=events,
            start=5,
            duration=10,
            color="gray",
            event_color={1: "r", 2: "g", 3: "b", 4: "m", 5: "y",
                         32: "k"},  # set color according to events id
        )
        brain3d(args.file, args.uploadId, s_pred, root_path, {
            "patientId": args.patientId,
            "uploadId": args.uploadId,
            "figUrl": figure_url,
            "matUrl": mat_url,
        }, hemi='both')

        data = {"uploadId": args.uploadId}
        # Print the data to stdout (can be captured by subprocess.run)
        with open("output.json", "w") as outfile:
            json.dump(data, outfile)
