import os
import sys
from datetime import datetime
from pysrcai.config.config_loader import load_config
from pysrcai.core.factory import SimulationFactory

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'basic_schema.yaml')

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
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"notes/LOG_{timestamp}.txt"
    
    # Ensure notes directory exists
    os.makedirs("notes", exist_ok=True)
    
    # Set up output capture
    tee = TeeOutput(log_filename)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=== PySrcAI Engine Demo ===")
        print(f"Output being saved to: {log_filename}")
        print()
        
        # Load scenario config
        config = load_config(CONFIG_PATH)
        print("Loaded config:")
        print(config)

        # Build engine and steps from factory
        factory = SimulationFactory()
        engine, steps = factory.create_engine(config)
        print(f"\n[Engine] Running for {steps} steps...")
        engine.run(steps=steps)
        print("\n[Engine] Simulation complete.")
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
