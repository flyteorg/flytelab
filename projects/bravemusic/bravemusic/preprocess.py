import os
import json
import math
import librosa
from datasource import download_gtzan_repo, GTZAN_ZIP_FILE_PATH

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
BAD_FORMATS = ["jazz.00054.wav"]


def clean_dataset():
    for (dir_path, dir_names, filenames) in os.walk(f"{GTZAN_ZIP_FILE_PATH}/genres/"):
        print(dir_path)
        [
            os.remove(f"{dir_path}{filename}")
            for filename in filenames
            if not filename.endswith(".wav")
        ]
        [
            os.renames(
                old=f"{dir_path}/{filename}",
                new=f"{dir_path}/{filename}".replace("._", ""),
            )
            for filename in filenames
            if f"{dir_path}/{filename}".startswith("._")
        ]
        [
            os.remove(f"{dir_path}/{filename}")
            for filename in filenames
            if filename.startswith("._")
        ]


def preprocess(
    dataset_path: str,
    json_path: str,
    num_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    num_segments: int = 10,
) -> dict:
    data = {"mapping": [], "labels": [], "mfcc": []}

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dir_path, dir_names, filenames) in enumerate(
        os.walk(f"{GTZAN_ZIP_FILE_PATH}/genres/")
    ):

        # ensure we're processing a genre sub-folder level
        if dir_path is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dir_path.split("/")[-1]
            print(semantic_label)
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:
                if f not in BAD_FORMATS:
                    # load audio file
                    file_path = os.path.join(dir_path, f)
                    signal, sample_rate = librosa.load(path=file_path, sr=SAMPLE_RATE)

                    # process all segments of audio file
                    for d in range(num_segments):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(
                            y=signal[start:finish],
                            sr=sample_rate,
                            n_mfcc=num_mfcc,
                            n_fft=n_fft,
                            hop_length=hop_length,
                        )
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)
                            print("{}, segment:{}".format(file_path, d + 1))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    return data


if __name__ == "__main__":
    download_gtzan_repo()
    # clean_dataset()
    data = preprocess(dataset_path=GTZAN_ZIP_FILE_PATH, json_path="data.json")
    print(data)
