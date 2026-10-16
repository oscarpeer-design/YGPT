# This file checks that every Python package used by the code in this repository is installed and can be loaded.
# It reads the import lines of every .py file in the repository, so a package that is added to the code later is checked without changing this file.
# Run it with the same Python that runs YitzGPT:  python package_info.py
# (It needs Python 3.10 or newer.)

# sys: shows which Python is running, lists the modules that come with Python itself, and sets the exit code
import sys
# ast: reads a Python file and finds its import lines without running it
import ast
# importlib: checks whether a package is installed, loads it, and finds its version
import importlib
import importlib.util
import importlib.metadata
# Path: library used to find file paths on computer
from pathlib import Path

# Packages the program can run without (config.py falls back to the plain CPU count when psutil is missing)
OPTIONAL_PACKAGES = ["psutil"]
# Packages that are installed under a different name than the one used in "import ..."
PIP_NAMES = {"faiss": "faiss-cpu"}

# Find every .py file in the repository folder and the folders inside it. Returns the paths, in alphabetical order.
def find_python_files(repo_folder: Path) -> list[Path]:
    # the files found so far
    python_files = []
    # go through every .py file below the repository folder
    for file_path in sorted(repo_folder.rglob("*.py")):
        # the folders of the version control (.git), the editor (.vs) and Python's own cache (__pycache__) hold no code of ours
        skip = False
        # the folders between the repository folder and the file (the last part is the file name, which is not a folder)
        for folder_name in file_path.relative_to(repo_folder).parts[:-1]:
            if folder_name.startswith(".") or folder_name == "__pycache__":
                skip = True
        # keep the file unless it is inside one of those folders
        if not skip:
            python_files.append(file_path)
    return python_files

# Find the packages that one Python file imports (only the first part of the name, so "tree_sitter.Node" gives "tree_sitter").
# Returns the names, or an Exception if the file could not be read.
def find_imported_modules(file_path: Path) -> list[str] | Exception:
    try:
        # read the file as bytes, which lets Python work out the file's text encoding itself, and turn it into a tree of its statements (the code is not run)
        tree = ast.parse(file_path.read_bytes())
    except Exception as e:
        return Exception(f"Could not read the imports of {file_path.name}: {e}")
    # the names found so far
    names = []
    # look at every statement in the file, including the ones inside functions (some packages are only imported when they are needed)
    for node in ast.walk(tree):
        # "import numpy" or "import os.path, re"
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        # "from pathlib import Path". A relative import ("from . import x", level above 0) refers to this repository, so it is skipped.
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.append(node.module.split(".")[0])
    return names

# Check one package. Returns a status ("ok", "missing" or "broken") and a note: the version for "ok", the reason otherwise.
# distributions says which installed package provides each importable name (it is the same for every package, so the caller makes it once).
def check_package(name: str, distributions: dict) -> tuple[str, str]:
    try:
        # find_spec looks for the package without loading it, and gives None when it is not installed
        if importlib.util.find_spec(name) is None:
            return "missing", "not installed"
        # loading it shows problems that only appear then (for example a missing system library)
        importlib.import_module(name)
    except Exception as e:
        return "broken", f"{type(e).__name__}: {e}"
    # the installed package(s) that provide this name (for example "faiss" is provided by "faiss-cpu")
    provided_by = distributions.get(name)
    if not provided_by:
        return "ok", "version unknown"
    try:
        return "ok", importlib.metadata.version(provided_by[0])
    except Exception:
        return "ok", "version unknown"

# Check that every package used by the code in this repository is installed and can be loaded, and print what was found.
# Returns True if nothing that is needed is missing or broken.
def runtime_check() -> bool:
    print("=== PYTHON ===")
    print(sys.executable)
    print(sys.version)
    # sys.stdlib_module_names (the list of modules that come with Python) and the package list below need Python 3.10
    if sys.version_info < (3, 10):
        print("\nThis check needs Python 3.10 or newer.")
        return False

    # the repository is the folder this file is in
    repo_folder = Path(__file__).resolve().parent
    python_files = find_python_files(repo_folder)
    # the files of the repository itself are not packages to install ("import config" refers to config.py)
    local_modules = set()
    for file_path in python_files:
        local_modules.add(file_path.stem)

    # Whether everything the program needs is installed and works (a file that can not be read leaves this unknown, so it counts as a problem)
    needed_packages_work = True
    # For every package that is not part of Python and not part of the repository, the files that import it
    used_by = {}
    for file_path in python_files:
        names = find_imported_modules(file_path)
        # a file that can not be read is reported and the others are still checked
        if isinstance(names, Exception):
            print(f"Warning: {names}")
            needed_packages_work = False
            continue
        for name in names:
            # modules that come with Python and modules of the repository need no installing
            if name in sys.stdlib_module_names or name in local_modules:
                continue
            if name not in used_by:
                used_by[name] = []
            # each file is listed once, however many times it imports the package
            if file_path.name not in used_by[name]:
                used_by[name].append(file_path.name)

    # which installed package provides each importable name
    distributions = importlib.metadata.packages_distributions()

    print(f"\n=== PACKAGES USED BY THE {len(python_files)} PYTHON FILES IN {repo_folder} ===")
    # the pip names of what is missing, and whether any package is installed but can not be loaded
    to_install = []
    broken_found = False
    for name in sorted(used_by):
        status, note = check_package(name, distributions)
        is_optional = name in OPTIONAL_PACKAGES
        pip_name = PIP_NAMES.get(name, name)
        # the label shown at the start of the line
        if status == "ok":
            label = "OK"
        elif status == "missing" and is_optional:
            label = "OPTIONAL"
        elif status == "missing":
            label = "MISSING"
        else:
            label = "BROKEN"
        # what to do about it
        if status == "missing":
            to_install.append(pip_name)
        if status == "broken":
            broken_found = True
        # a missing optional package is fine, anything else that is not OK means the program may not run
        if status != "ok" and not is_optional:
            needed_packages_work = False
        print(f"{label:<9} {name:<20} {note}")
        print(f"          used in: {', '.join(used_by[name])}")

    print()
    if to_install:
        print("To install what is missing, run:")
        print(f'"{sys.executable}" -m pip install {" ".join(to_install)}')
    if broken_found:
        print("A BROKEN package is installed but can not be loaded. The message after its name says why; reinstalling it often fixes it.")
    if needed_packages_work:
        print("Everything the program needs is installed and loads.")
    return needed_packages_work

if __name__ == "__main__":
    # the exit code is 0 when everything is fine, so a script or pipeline can use the result
    sys.exit(0 if runtime_check() else 1)