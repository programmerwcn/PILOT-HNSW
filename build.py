"""Build the retained Linux C++17 entry points without regenerating sources."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
TARGETS = {
    'index_float': 'src/index_hnsw.cpp',
    'index_uint8': 'src/index_hnsw_u8_parallel.cpp',
    'probe_gist': 'src/gist_zero_anchor_features.cpp',
    'probe_sift': 'src/sift100m_zero_anchor_features.cpp',
    'probe_deep': 'src/deep100m_zero_anchor_features.cpp',
    'pilot_float': 'benchmarks/pilot_resume_fast_float.cpp',
    'pilot_sift': 'benchmarks/pilot_resume_fast_sift.cpp',
    'pilot_deep': 'benchmarks/pilot_resume_fast_deep.cpp',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['all', *TARGETS], default='all')
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--eigen-include', type=Path, default=Path('/usr/include/eigen3'))
    parser.add_argument('--jsoncpp-include', type=Path, default=Path('/usr/include/jsoncpp'))
    parser.add_argument('--lightgbm-lib', type=Path, required=True, help='Full path to lib_lightgbm.so')
    parser.add_argument('--output', type=Path, default=ROOT/'build')
    args = parser.parse_args()
    if not (args.eigen_include/'Eigen/Dense').is_file():
        parser.error('--eigen-include must contain Eigen/Dense')
    lib = args.lightgbm_lib.resolve()
    if not lib.is_file():
        parser.error('--lightgbm-lib does not exist')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in TARGETS.items():
        if args.target not in ('all', name):
            continue
        command = [args.cxx, '-std=c++17', '-O3', '-march=native', '-fopenmp',
                   '-I'+str(ROOT/'src'), '-I'+str(args.eigen_include.resolve()),
                   '-I'+str(args.jsoncpp_include.resolve()), str(ROOT/source),
                   str(ROOT/'src/hnswlib/logging.cpp'), '-ljsoncpp', str(lib),
                   '-Wl,-rpath,'+str(lib.parent), '-o', str(args.output/name)]
        subprocess.run(command, check=True)
        print('Built', name, flush=True)

if __name__ == '__main__':
    main()
