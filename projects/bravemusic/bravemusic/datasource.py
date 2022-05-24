import os
import tarfile
import git

GIT_URL = "https://huggingface.co/datasets/marsyas/gtzan"
GTZAN_PATH = "./gtzan"
GTZAN_ZIP_FILE_PATH = "./gtzan/data"
GTZAN_ZIP_FILE_NAME = "genres.tar.gz"


class Progress(git.remote.RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=""):
        print(self._cur_line)


def download_gtzan_repo():
    if not os.path.isdir(GTZAN_PATH) or not any(os.scandir(GTZAN_PATH)):
        git.Repo.clone_from(url=GIT_URL, to_path=GTZAN_PATH, progress=Progress())
        extract_gtzan_repo_tarball()
    else:
        print("dataset already exists")


def extract_gtzan_repo_tarball():
    # open file
    file = tarfile.open(f"{GTZAN_ZIP_FILE_PATH}/{GTZAN_ZIP_FILE_NAME}")
    # extracting file
    file.extractall(GTZAN_ZIP_FILE_PATH)
    file.close()


if __name__ == "__main__":
    download_gtzan_repo()
