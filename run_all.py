"""Run the whole pipeline: execute the four notebooks in order, in place.

Usage (from the project root, with the virtual environment active):

    python run_all.py            # all four notebooks
    python run_all.py 03 04      # only the notebooks whose names start with 03 or 04

Each notebook is executed top to bottom with a fresh kernel and saved with its
outputs. The raw workbook is downloaded from Figshare by notebook 01 if it is
not already in data/raw/.

Only the four official notebooks listed in NOTEBOOKS are run. The original,
uncleaned notebooks are kept as backups in notebooks/archive_original/ and are
never executed by this script.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent / "notebooks"
NOTEBOOKS = [
    "01_Dataset_Audit_and_EDA.ipynb",
    "02_Data_Preparation_and_Feature_Engineering.ipynb",
    "03_Model_Development_and_Training.ipynb",
    "04_Model_Evaluation_and_Research_Findings.ipynb",
]


def main(prefixes: list[str]) -> None:
    notebooks = [NOTEBOOK_DIR / name for name in NOTEBOOKS]
    if prefixes:
        notebooks = [nb for nb in notebooks if nb.name[:2] in prefixes]
    for path in notebooks:
        start = time.time()
        print(f"Running {path.name} ...", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=3600, kernel_name="python3",
                                resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        print(f"  finished in {time.time() - start:.0f} s", flush=True)
    print("All requested notebooks ran without error.")


if __name__ == "__main__":
    main(sys.argv[1:])
