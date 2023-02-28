from pathlib import Path
import shutil

# Common project paths
PROJECT_ROOT_DIR = Path(__file__).parents[2]
DATA_ROOT_DIR = PROJECT_ROOT_DIR / "data"
MODEL_ROOT_DIR = PROJECT_ROOT_DIR / "model"
SRC_ROOT_DIR = PROJECT_ROOT_DIR / "src"

# path to fetch data
SOURCE_DIR = DATA_ROOT_DIR / "raw" / "chest_Xray"
DEST_DIR = DATA_ROOT_DIR / "interim" / "full"


# Useful project information
class ProjectInfo():
    """_summary_
    Class that provides project information :
        - principal paths
    """
    def __init__(self) -> None:
        self.root_dir = PROJECT_ROOT_DIR
        self.data_dir = DATA_ROOT_DIR
        self.model_dir = MODEL_ROOT_DIR
        self.srcipts_dir = SRC_ROOT_DIR


# Functions for fetching data
def create_folders(exist_ok=True):
    """_summary_
    Function that creates class folder in full dataset.

    Args:
        exist_ok (bool, optional). Defaults to True.

    Returns:
        normal_dir: Path
        virus_dir: Path
        bacteria_dir: Path
    """
    normal_dir = DEST_DIR / 'normal'
    normal_dir.mkdir(parents=True, exist_ok=exist_ok)
    virus_dir = DEST_DIR / 'virus'
    virus_dir.mkdir(parents=True, exist_ok=exist_ok)
    bacteria_dir = DEST_DIR / 'bacteria'
    bacteria_dir.mkdir(parents=True, exist_ok=exist_ok)

    return normal_dir, virus_dir, bacteria_dir


def dest_file_name(directory, index):
    """_summary_
    Function that generates filename for images inside full dataset.

    Args:
        directory (Path)
        index (int)

    Returns:
        Path
    """
    return directory / "{}_img_{:04d}.jpeg".format(directory.name, index)


def check_full_dataset(print_result=False):
    """_summary_
    Function that check if the full folder is the same as
    the chest_Xray folder.
    Args:
        print_result (bool, optional). Defaults to False.

    Returns:
        bool
    """
    full = {}
    for folder in DEST_DIR.glob("*"):
        class_img = folder.name
        nb_img = len(list(folder.glob("*.jpeg")))
        full[class_img] = nb_img
        if print_result:
            print("full dataset : class {} => nb images = {}".format(
                class_img, nb_img
                ))

    nb_normal = 0
    nb_virus = 0
    nb_bacteria = 0
    for file in list(SOURCE_DIR.rglob("*.jpeg")):
        if file.parents[0].name == "NORMAL":
            nb_normal += 1
        if "virus" in file.name.lower():
            nb_virus += 1
        if "bacteria" in file.name.lower():
            nb_bacteria += 1
    source = {"bacteria": nb_bacteria, "normal": nb_normal, "virus": nb_virus}
    if print_result:
        print('-'*40)
        for key, value in source.items():
            print("raw data     : class {} => nb images = {}".format(
                key, value
                ))
    return full == source


def fetch_full_dataset():
    """_summary_
    Final function for fetching data from chest_Xray folder to full
    folder.
    """
    if check_full_dataset(print_result=False):
        print("Full dataset already created")
    else:
        normal_dir, virus_dir, bacteria_dir = create_folders()

        idx_normal = 1
        idx_virus = 1
        idx_bacteria = 1
        for file in SOURCE_DIR.rglob("*.jpeg"):
            if file.parents[0].name == "NORMAL":
                shutil.copy2(file, dest_file_name(normal_dir, idx_normal))
                idx_normal += 1
            if "virus" in file.name.lower():
                shutil.copy2(file, dest_file_name(virus_dir, idx_virus))
                idx_virus += 1
            if "bacteria" in file.name.lower():
                shutil.copy2(file, dest_file_name(bacteria_dir, idx_bacteria))
                idx_bacteria += 1


if __name__ == "__main__":
    check_full_dataset()
