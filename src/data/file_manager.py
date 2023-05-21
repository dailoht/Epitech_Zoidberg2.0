"""
This module provides a FileManager class that helps manage common project
paths. It also includes methods to check and fetch the full dataset, which is
constructed by copying and organizing image files from the raw dataset. The
module uses the `pathlib`, `shutil`, and `zipfile` modules to interact with
the file system and manage file paths.
"""

from pathlib import Path
import shutil
import zipfile

# Common project paths
PROJECT_ROOT_DIR = Path(__file__).parents[2]
DATA_ROOT_DIR = PROJECT_ROOT_DIR / "data"
MODEL_ROOT_DIR = PROJECT_ROOT_DIR / "models"
SRC_ROOT_DIR = PROJECT_ROOT_DIR / "src"
RESOURCE_DIR = PROJECT_ROOT_DIR / "report" / "resources"
NOTEBOOK_DIR = PROJECT_ROOT_DIR / "notebooks"


class FileManager:
    """
    FileManager class that manages the data folders for the project. It
    includes methods for creating, fetching, and checking the dataset.

    Attributes:
        root_dir (Path): Root directory for the project.
        data_dir (Path): Directory for project data.
        model_dir (Path): Directory for saved models.
        script_dir (Path): Directory for Python scripts.
        resource_dir (Path): Directory for project resources.
        raw_data_subdir (Path): Directory for the raw dataset.
        interim_data_subdir (Path): Directory for the interim dataset.

    Methods:
        __create_folders(exist_ok=True):
            Creates folders for the interim dataset.
        __dest_file_name(directory, index):
            Generates a filename for images inside the full dataset.
        check_full_dataset(print_result=False):
            Checks if the raw dataset is empty and if the full folder is the
            same as the chest_Xray folder.
        fetch_full_dataset():
            Fetches data from the zip file to the interim/full folder.
    """
    def __init__(self) -> None:
        """
        Initializes a new instance of the FileManager class.
        """
        self.root_dir = PROJECT_ROOT_DIR
        self.data_dir = DATA_ROOT_DIR
        self.model_dir = MODEL_ROOT_DIR
        self.script_dir = SRC_ROOT_DIR
        self.resource_dir = RESOURCE_DIR
        self.notebook_dir = NOTEBOOK_DIR
        self.raw_data_subdir = self.data_dir / 'raw' / 'chest_Xray'
        self.interim_data_subdir = self.data_dir / 'interim' / 'full'

    def __create_folders(self, exist_ok=True):
        """
        Creates a directory for each class in the interim dataset.

        Args:
            exist_ok (bool, optional): If True, the folder will not raise an
                error if it already exists. Defaults to True.

        Returns:
            normal_dir, virus_dir, bacteria_dir (Path): Paths to each of the
                three class folders.
        """
        normal_dir = Path(self.interim_data_subdir / 'normal')
        normal_dir.mkdir(parents=True, exist_ok=exist_ok)
        virus_dir = Path(self.interim_data_subdir / 'virus')
        virus_dir.mkdir(parents=True, exist_ok=exist_ok)
        bacteria_dir = Path(self.interim_data_subdir / 'bacteria')
        bacteria_dir.mkdir(parents=True, exist_ok=exist_ok)

        return normal_dir, virus_dir, bacteria_dir

    def __dest_file_name(self, directory, index):
        """
        Generates a filename for images inside the full dataset.

        Args:
            directory (Path): The path to the directory.
            index (int): The index of the image.

        Returns:
            Path: The path to the image file.
        """
        return directory / "{}_img_{:04d}.jpeg".format(directory.name, index)

    def check_full_dataset(self, print_result=False):
        """
        Checks if the raw dataset is empty and if the full folder is the same
        as the chest_Xray folder.

        Args:
            print_result (bool, optional): If True, the result will be printed
                to the console. Defaults to False.

        Raises:
            FileNotFoundError: If the raw dataset is empty.

        Returns:
            bool: True if the interim/full folder is the same as the raw
                dataset, otherwise False.
        """
        raw_files_path = list(Path(self.data_dir / "raw").glob('*'))
        if len(raw_files_path) == 1 and raw_files_path[0].name == '.gitkeep':
            raise FileNotFoundError("Folder raw is empty, \
                please provide data zip file in 'data/raw/' !")

        full = {}
        for folder in self.interim_data_subdir.glob("*"):
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
        for file in list(self.raw_data_subdir.rglob("*.jpeg")):
            if file.parents[0].name == "NORMAL":
                nb_normal += 1
            if "virus" in file.name.lower():
                nb_virus += 1
            if "bacteria" in file.name.lower():
                nb_bacteria += 1
        source = {
            "bacteria": nb_bacteria,
            "normal": nb_normal,
            "virus": nb_virus
        }
        if print_result:
            print('-'*40)
            for key, value in source.items():
                print("raw data     : class {} => nb images = {}".format(
                    key, value
                    ))
        return full == source

    def fetch_full_dataset(self):
        """
        Fetches the full dataset from the raw data directory
        to the interim data directory, creating the full dataset in the
        process. If the full dataset already exists, it does nothing.

        Returns:
            None
        """
        if self.check_full_dataset(print_result=False):
            print("Full dataset already created")
            return

        normal_dir, virus_dir, bacteria_dir = self.__create_folders()

        if not self.raw_data_subdir.is_dir():
            print("Uncompressing zip file ...")
            for zip_file_path in Path(self.data_dir / "raw").glob("*.zip"):
                with zipfile.ZipFile(zip_file_path, 'r') as file:
                    file.extractall(path=Path(self.data_dir / "raw"))
            for item in Path(self.data_dir / "raw").glob("*"):
                if item.is_dir() and 'chest_xray' in item.name.lower():
                    item.rename(self.raw_data_subdir)
                elif item.name != '.gitkeep':
                    if item.is_dir():
                        shutil.rmtree(str(item))
                    elif item.is_file():
                        item.unlink()

        for item in self.raw_data_subdir.glob('*'):
            if not item.is_dir() or item.name not in ['train', 'test', 'val']:
                if item.is_dir():
                    shutil.rmtree(str(item))
                elif item.is_file():
                    item.unlink()

        print("Copying files from raw/Chest_Xray to interim/full ...")
        idx_normal = 1
        idx_virus = 1
        idx_bacteria = 1
        for file in self.raw_data_subdir.rglob("*.jpeg"):
            if file.parents[0].name == "NORMAL":
                shutil.copy2(file,
                             self.__dest_file_name(normal_dir,
                                                   idx_normal))
                idx_normal += 1
            if "virus" in file.name.lower():
                shutil.copy2(file,
                             self.__dest_file_name(virus_dir,
                                                   idx_virus))
                idx_virus += 1
            if "bacteria" in file.name.lower():
                shutil.copy2(file,
                             self.__dest_file_name(bacteria_dir,
                                                   idx_bacteria))
                idx_bacteria += 1


if __name__ == "__main__":
    file_manager = FileManager()
    file_manager.check_full_dataset()
