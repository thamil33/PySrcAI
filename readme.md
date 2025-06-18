# pyscrai & concordia

## Setup

1. Create a Python 3.12 virtual environment named `.venv` at the project root:
   ```bash
    python -m venv .venv
   ```

2. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (bash):
     ```bash
     source .venv/Scripts/activate
     ```

3. Install the packages in editable mode:
   ```bash
   pip install -e .
   ```

Both `pyscrai` and `concordia` will now be available as importable modules.

4. Install python dependencies Make certain virtual environment is activated before runnning.
```bash
pip install -r requirements.txt
```

5. Run a scenario from project root, i.e 
``` bash
python -m pyscrai.scenario.determinist_libertarian 
```
