import os

# Expose all .yaml files in this directory as accessible resources
TEMPLATES_DIR = os.path.dirname(__file__)
yaml_files = [
    f for f in os.listdir(TEMPLATES_DIR)
    if f.endswith('.yaml')
]

__all__ = [os.path.splitext(f)[0] for f in yaml_files]

def get_template_path(name: str) -> str:
    """
    Get the absolute path to a template YAML file by name (without .yaml extension).
    """
    filename = f"{name}.yaml"
    if filename in yaml_files:
        return os.path.join(TEMPLATES_DIR, filename)
    raise FileNotFoundError(f"No such template: {name}")

# Optionally, provide a function to list all available templates
def list_templates():
    return [os.path.splitext(f)[0] for f in yaml_files]
