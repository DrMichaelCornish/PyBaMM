import os
import subprocess
import tarfile
import argparse

try:
    # wget module is required to download SUNDIALS or SuiteSparse.
    import wget

    NO_WGET = False
except ModuleNotFoundError:
    NO_WGET = True


def download_extract_library(url, download_dir):
    # Download and extract archive at url
    if NO_WGET:
        error_msg = (
            "Could not find wget module."
            " Please install wget module (pip install wget)."
        )
        raise ModuleNotFoundError(error_msg)
    archive = wget.download(url, out=download_dir)
    tar = tarfile.open(archive)
    tar.extractall(download_dir)


def update_activate_or_bashrc(install_dir):
    # Look for current python virtual env and add export statement
    # for LD_LIBRARY_PATH in activate script.  If no virtual env found,
    # then the current user's .bashrc file is modified instead.

    export_statement = "export LD_LIBRARY_PATH={}/lib:$LD_LIBRARY_PATH".format(
        install_dir
    )

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        script_path = os.path.join(venv_path, "bin/activate")
    else:
        script_path = os.path.join(os.environ.get("HOME"), ".bashrc")

    if os.getenv("LD_LIBRARY_PATH") and "{}/lib".format(install_dir) in os.getenv(
        "LD_LIBRARY_PATH"
    ):
        print("{}/lib was found in LD_LIBRARY_PATH.".format(install_dir))
        print("--> Not updating venv activate or .bashrc scripts")
    else:
        with open(script_path, "a+") as fh:
            # Just check that export statement is not already there.
            if export_statement not in fh.read():
                fh.write(export_statement)
                print(
                    "Adding {}/lib to LD_LIBRARY_PATH"
                    " in {}".format(install_dir, script_path)
                )


# First check requirements: make and cmake
try:
    subprocess.run(["make", "--version"])
except OSError:
    raise RuntimeError("Make must be installed.")
try:
    subprocess.run(["cmake", "--version"])
except OSError:
    raise RuntimeError("CMake must be installed.")

# Create download directory in PyBaMM dir
pybamm_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
download_dir = os.path.join(pybamm_dir, "KLU_module_deps")
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Get installation location
default_install_dir = os.path.join(os.getenv("HOME"), ".local")
parser = argparse.ArgumentParser(
    description="Download, compile and install Sundials and SuiteSparse."
)
parser.add_argument("--install-dir", type=str, default=default_install_dir)
args = parser.parse_args()
install_dir = (
    args.install_dir
    if os.path.isabs(args.install_dir)
    else os.path.join(pybamm_dir, args.install_dir)
)

# 1 --- Download SuiteSparse
suitesparse_version = "5.6.0"
suitesparse_url = (
    "https://github.com/DrTimothyAldenDavis/"
    + "SuiteSparse/archive/v{}.tar.gz".format(suitesparse_version)
)
download_extract_library(suitesparse_url, download_dir)

# The SuiteSparse KLU module has 4 dependencies:
# - suitesparseconfig
# - AMD
# - COLAMD
# - BTF
suitesparse_dir = "SuiteSparse-{}".format(suitesparse_version)
suitesparse_src = os.path.join(download_dir, suitesparse_dir)
print("-" * 10, "Building SuiteSparse_config", "-" * 40)
make_cmd = ["make", "library"]
install_cmd = [
    "make",
    "install",
    "INSTALL={}".format(install_dir),
    "INSTALL_DOC=/tmp/doc",
]
print("-" * 10, "Building SuiteSparse", "-" * 40)
for libdir in ["SuiteSparse_config", "AMD", "COLAMD", "BTF", "KLU"]:
    build_dir = os.path.join(suitesparse_src, libdir)
    subprocess.run(make_cmd, cwd=build_dir)
    subprocess.run(install_cmd, cwd=build_dir)

# 2 --- Download SUNDIALS
sundials_version = "5.0.0"
sundials_url = (
    "https://computing.llnl.gov/"
    + "projects/sundials/download/sundials-{}.tar.gz".format(sundials_version)
)
download_extract_library(sundials_url, download_dir)

# Set install dir for SuiteSparse libs
# Ex: if install_dir -> "/usr/local/" then
# KLU_INCLUDE_DIR -> "/usr/local/include"
# KLU_LIBRARY_DIR -> "/usr/local/lib"
KLU_INCLUDE_DIR = os.path.join(install_dir, "include")
KLU_LIBRARY_DIR = os.path.join(install_dir, "lib")
cmake_args = [
    "-DLAPACK_ENABLE=ON",
    "-DSUNDIALS_INDEX_SIZE=32",
    "-DBUILD_ARKODE:BOOL=OFF",
    "-DBUILD_CVODE=OFF",
    "-DBUILD_CVODES=OFF",
    "-DBUILD_IDAS=OFF",
    "-DBUILD_KINSOL=OFF",
    "-DEXAMPLES_ENABLE:BOOL=OFF",
    "-DKLU_ENABLE=ON",
    "-DKLU_INCLUDE_DIR={}".format(KLU_INCLUDE_DIR),
    "-DKLU_LIBRARY_DIR={}".format(KLU_LIBRARY_DIR),
    "-DCMAKE_INSTALL_PREFIX=" + install_dir,
]

# SUNDIALS are built within download_dir 'build_sundials' in the PyBaMM root
# download_dir
build_dir = os.path.abspath(os.path.join(download_dir, "build_sundials"))
if not os.path.exists(build_dir):
    print("\n-" * 10, "Creating build dir", "-" * 40)
    os.makedirs(build_dir)

sundials_src = "../sundials-{}".format(sundials_version)
print("-" * 10, "Running CMake prepare", "-" * 40)
subprocess.run(["cmake", sundials_src] + cmake_args, cwd=build_dir)

print("-" * 10, "Building the sundials", "-" * 40)
make_cmd = ["make", "install"]
subprocess.run(make_cmd, cwd=build_dir)

update_activate_or_bashrc(install_dir)
