"""
Test that all __init__.py files exist in each package directory and can be imported without errors.
"""
import os
import importlib.util
import pytest

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories to check for __init__.py files (add more as needed)
PACKAGE_DIRS = [
    'concordia',
    os.path.join('concordia', 'agents'),
    os.path.join('concordia', 'associative_memory'),
    os.path.join('concordia', 'clocks'),
    os.path.join('concordia', 'components'),
    os.path.join('concordia', 'contrib'),
    os.path.join('concordia', 'deprecated'),
    os.path.join('concordia', 'document'),
    os.path.join('concordia', 'environment'),
    os.path.join('concordia', 'language_model'),
    os.path.join('concordia', 'prefabs'),
    os.path.join('concordia', 'testing'),
    os.path.join('concordia', 'thought_chains'),
    os.path.join('concordia', 'typing'),
    'pysrcai',
    os.path.join('pysrcai', 'agentica'),
    os.path.join('pysrcai', 'geo_mod'),
    # 'util',  # Excluded: not a package, no __init__.py required
    # Exclude pytests, .egg-info, and dot-directories
]

def find_init_files():
    """Yield (package_dir, init_file_path) for each package directory."""
    for pkg in PACKAGE_DIRS:
        pkg_path = os.path.join(PROJECT_ROOT, pkg)
        init_path = os.path.join(pkg_path, '__init__.py')
        yield pkg, init_path

@pytest.mark.parametrize("pkg,init_path", list(find_init_files()))
def test_init_file_exists(pkg, init_path):
    assert os.path.isfile(init_path), f"Missing __init__.py in {pkg}"

@pytest.mark.parametrize("pkg,init_path", list(find_init_files()))
def test_init_file_importable(pkg, init_path):
    if not os.path.isfile(init_path):
        pytest.skip(f"No __init__.py in {pkg}")
    # Convert path to module name
    rel_path = os.path.relpath(init_path, PROJECT_ROOT)
    mod_name = rel_path.replace(os.sep, '.')[:-3]  # strip .py
    try:
        spec = importlib.util.spec_from_file_location(mod_name, init_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Failed to import {mod_name}: {e}")
