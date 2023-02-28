from pathlib import Path
import shutil

PROJECT_ROOT_DIR = Path(__file__).parents[2]
SOURCE_DIR = PROJECT_ROOT_DIR / "data" / "raw" / "chest_Xray"
DEST_DIR = PROJECT_ROOT_DIR / "data" / "interim" / "full"

def create_folders(exist_ok=True):
    normal_dir = DEST_DIR / 'normal'
    normal_dir.mkdir(parents=True, exist_ok=exist_ok)
    virus_dir = DEST_DIR / 'virus'
    virus_dir.mkdir(parents=True, exist_ok=exist_ok)
    bacteria_dir = DEST_DIR / 'bacteria'
    bacteria_dir.mkdir(parents=True, exist_ok=exist_ok)
    
    return normal_dir, virus_dir, bacteria_dir


def dest_file_name(directory, index):
    return directory / "{}_img_{:04d}.jpeg".format(directory.name, index)


def fetch_full_dataset():
    all_files = list(SOURCE_DIR.rglob("*.jpeg"))
    nb_files = len(all_files)
    
    if len(list(DEST_DIR.rglob("*.jpeg"))) == nb_files:
        print("Clean dataset already created")
    else:
        normal_dir, virus_dir, bacteria_dir = create_folders()
        
        idx_normal = 1
        idx_virus = 1
        idx_bacteria = 1
        for file in all_files:
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
    fetch_full_dataset()
