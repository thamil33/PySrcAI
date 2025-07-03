"""
Enhanced engine demo with interactive environment objects.

This demo showcases the use of environment objects and agent interactions
in a more complex scenario.
"""
import os
from pysrcai.src.config.config_loader import load_config
from pysrcai.src.environment.enhanced_factory import EnhancedSimulationFactory

# Use the enhanced schema with environment objects
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/scenario/enhanced_schema.yaml')

def main():
    print("=== PySrcAI Enhanced Environment Demo ===")
    # Load scenario config
    config = load_config(CONFIG_PATH)
    print("Loaded config with environment objects")
    
    # Build enhanced engine and steps from factory
    factory = EnhancedSimulationFactory()
    engine, steps = factory.create_engine(config)
    print(f"\n[Engine] Running for {steps} steps with interactive environment...")
    engine.run(steps=steps)
    print("\n[Engine] Simulation complete.")

if __name__ == "__main__":
    main()
