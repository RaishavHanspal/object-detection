from pathlib import Path

def ensure_dir(path_str: str) -> None:
    """Ensure that a directory exists. If it doesn't, create it."""
    dir_path = Path(path_str)
    dir_path.mkdir(parents=True, exist_ok=True)