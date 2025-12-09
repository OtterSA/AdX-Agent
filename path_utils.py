from pathlib import Path

def find_local_root(marker_file='.local_root') -> Path:
    """
    Find the project root directory marked by a specific file.

    Args:
        marker_file (str): The name of the marker file to look for.

    Returns:
        Path: The path to the project root directory.

    Raises:
        FileNotFoundError: If the project root cannot be found.
    """
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / marker_file).exists():
            return parent
    raise FileNotFoundError('Project root not found.')

def path_from_local_root(path = ''): 
    project_root = find_local_root()
    return project_root / path