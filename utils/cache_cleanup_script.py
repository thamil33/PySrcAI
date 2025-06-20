
import os
import shutil

def remove_pycache_dirs(root_dir):
    """Recursively delete all __pycache__ directories under root_dir."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            print(f"Removing: {pycache_path}")
            shutil.rmtree(pycache_path)
            removed += 1
    print(f"Removed {removed} __pycache__ directories.")

if __name__ == "__main__":
    # Set the root directory to the workspace root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    remove_pycache_dirs(root)
