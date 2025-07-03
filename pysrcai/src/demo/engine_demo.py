import os
from pysrcai.src.config.config_loader import load_config
from pysrcai.src.factory import SimulationFactory

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/scenario/basic_schema.yaml')

def main():
    print("=== PySrcAI Engine Demo ===")
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

if __name__ == "__main__":
    main()
