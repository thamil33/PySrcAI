"""Hello World demo for PySrcAI.

This script demonstrates the minimal functionality of the PySrcAI framework
using a simple debate scenario with a whiteboard for shared state.
"""

import os
import sys
import argparse
from datetime import datetime
from pysrcai.config.config_loader import load_config
from pysrcai.core.factory import SimulationFactory
from pysrcai.core.engines.minimal_engine import MinimalEngine

# Default config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'prefabs', 'minimal.yaml')

class TeeOutput:
    """Captures output and writes to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate writing

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the PySrcAI Hello World Simulation')
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG_PATH,
                      help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    config_path = args.config
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pysrcai/data/logs/HELLO_LOG_{timestamp}.txt"
    
    # Ensure logs directory exists
    os.makedirs("pysrcai/data/logs", exist_ok=True)
    
    # Set up output capture
    tee = TeeOutput(log_filename)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=== PySrcAI Minimal Demo ===")
        print(f"Output being saved to: {log_filename}")
        print(f"Using configuration: {config_path}")
        print()
        
        # Load scenario config
        config = load_config(config_path)
        print("Loaded config:")
        print(config)

        # Create simulation using factory
        factory = SimulationFactory()
        engine, steps = factory.create_engine(config)
        
        # Get steps from config
        steps = config.get("engine", {}).get("steps", 6)
        
        print(f"\n[Engine] Running for {steps} steps...")
        engine.run(steps=steps)
        print("\n[Engine] Simulation complete.")
        
        # Print whiteboard final state
        print("\n[Engine] Final Whiteboard State:")
        for i, entry in enumerate(engine.environment.snapshot()):
            print(f"[{i+1}] {entry}")
        
        print(f"\n[Engine] Full output saved to: {log_filename}")
        
    except Exception as e:
        print(f"\n[Engine] Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = old_stdout
        tee.close()
        print(f"✅ Log saved to: {log_filename}")

if __name__ == "__main__":
    main()