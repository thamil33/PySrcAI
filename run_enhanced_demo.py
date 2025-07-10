"""Run script for the enhanced demo.

This script sets up the proper path and runs the enhanced demo.
"""

import os
import sys

# Get the project root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the root directory to Python path
sys.path.insert(0, root_dir)

# Import and run the enhanced demo
from pysrcai.demos.enhanced_demo import main

if __name__ == "__main__":
    main()
