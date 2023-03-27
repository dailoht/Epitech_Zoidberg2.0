from pathlib import Path
import shutil
import zipfile

# Common project paths
PROJECT_ROOT_DIR = Path(__file__).parents[2]
DATA_ROOT_DIR = PROJECT_ROOT_DIR / "data"
MODEL_ROOT_DIR = PROJECT_ROOT_DIR / "model"
SRC_ROOT_DIR = PROJECT_ROOT_DIR / "src"
RESOURCE_DIR = PROJECT_ROOT_DIR / "report" / "resources"


class FileManager:
    """_summary_
    Class that provides project information :
        - principal paths
    """
    def __init__(self) -> None:
        self.root_dir = PROJECT_ROOT_DIR
        self.data_dir = DATA_ROOT_DIR
        self.model_dir = MODEL_ROOT_DIR
        self.script_dir = SRC_ROOT_DIR
        self.resource_dir = RESOURCE_DIR
        self.raw_data_subdir = self.data_dir / 'raw' / 'chest_Xray'
        self.interim_data_subdir = self.data_dir / 'interim' / 'full'

    def __create_folders(self, exist_ok=True):
        """_summary_
        Method that creates class folder in full dataset.

        Args:
            exist_ok (bool, optional). Defaults to True.

        Returns:
            normal_dir: Path
            virus_dir: Path
            bacteria_dir: Path
        """
        normal_dir = Path(self.interim_data_subdir / 'normal')
        normal_dir.mkdir(parents=True, exist_ok=exist_ok)
        virus_dir = Path(self.interim_data_subdir / 'virus')
        virus_dir.mkdir(parents=True, exist_ok=exist_ok)
        bacteria_dir = Path(self.interim_data_subdir / 'bacteria')
        bacteria_dir.mkdir(parents=True, exist_ok=exist_ok)

        return normal_dir, virus_dir, bacteria_dir


    def __dest_file_name(self, directory, index):
        """_summary_
        Method that generates filename for images inside full dataset.

        Args:
            directory (Path)
            index (int)

        Returns:
            Path
        """
        return directory / "{}_img_{:04d}.jpeg".format(directory.name, index)


    def check_full_dataset(self, print_result=False):
        """_summary_
        Method that check if raw is empty and if the full folder is 
        the same as the chest_Xray folder.
        Args:
            print_result (bool, optional). Defaults to False.

        Returns:
            bool
        """
        
        raw_files_path = list(Path(self.data_dir / "raw").glob('*'))
        if len(raw_files_path) == 1 and raw_files_path[0].name == '.gitkeep':
            raise FileNotFoundError("Folder raw is empty, please provide data zip file in 'data/raw/' !")
        
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
        source = {"bacteria": nb_bacteria, "normal": nb_normal, "virus": nb_virus}
        if print_result:
            print('-'*40)
            for key, value in source.items():
                print("raw data     : class {} => nb images = {}".format(
                    key, value
                    ))
        return full == source


    def fetch_full_dataset(self):
        """_summary_
        Final function for fetching data from zip file to full
        folder.
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
                shutil.copy2(file, self.__dest_file_name(normal_dir, idx_normal))
                idx_normal += 1
            if "virus" in file.name.lower():
                shutil.copy2(file, self.__dest_file_name(virus_dir, idx_virus))
                idx_virus += 1
            if "bacteria" in file.name.lower():
                shutil.copy2(file, self.__dest_file_name(bacteria_dir, idx_bacteria))
                idx_bacteria += 1


if __name__ == "__main__":
    file_manager = FileManager()
    file_manager.check_full_dataset()
