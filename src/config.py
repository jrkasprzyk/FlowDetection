from pathlib import Path
from yaml import safe_load
from typing import Dict


def set_supervisor_path(computer) -> Path:

    onedrive_path = Path("OneDrive - UCB-O365/Datasets/flow_detection/")

    if computer == 'hpc':
        return Path("/projects/joka0958/supervisor/")
    elif computer == '2020laptop':
        return Path("C:/Users/josep/") / onedrive_path
    elif computer == 'seecdesktop':
        return Path("C:/Users/joka0958/") / onedrive_path


def set_output_path(computer) -> Path:

    github_path = Path("GitHub/MLtests/flow_detection/models/")

    if computer == 'hpc':
        return Path("/projects/joka0958/MLoutput/")
    elif computer == '2020laptop':
        return Path("C:/Users/josep/Documents/") / github_path
    elif computer == 'seecdesktop':
        return Path("//files.colorado.edu/CEAE/users/joka0958/Documents/") / github_path


def create_config(config_filename):
    config = safe_load(open(config_filename))
    config["supervisor_path"] = set_supervisor_path(config["computer"])
    config["output_path"] = set_output_path(config["computer"])
    return config
