from pathlib import Path


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    This function returns the path to the root directory of the project
    by navigating one level up from the current file's directory.

    :return: Path object representing the root directory of the project.
    """
    return Path(__file__).parents[1]
