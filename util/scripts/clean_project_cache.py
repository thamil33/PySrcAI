
import os
import shutil

def delete_cache_dirs(root_dir, patterns=("__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints")):
    """Recursively delete cache directories matching given patterns from root_dir."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for pattern in patterns:
            if pattern in dirnames:
                full_path = os.path.join(dirpath, pattern)
                print(f"Deleting: {full_path}")
                shutil.rmtree(full_path, ignore_errors=True)

def delete_cache_files(root_dir, extensions=(".pyc", ".pyo")):
    """Delete all files with given extensions recursively from root_dir."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extensions):
                file_path = os.path.join(dirpath, filename)
                print(f"Deleting: {file_path}")
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Cleaning cache in: {project_root}")
    delete_cache_dirs(project_root)
    delete_cache_files(project_root)
    print("Done.")
