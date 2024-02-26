# This file is used for running AxCaliber analysis of diffusion MRI data
# This file get command line arguments and start the analysis

import argparse
from pathlib import Path
from AxSI import axsi_main


def run_from_python(subj_folder, data_path, mask_path, bval_path, bvec_path, small_delta=15, big_delta=45, gmax=7.9, gamma_val=4257):
    subj_folder = assert_path(subj_folder)
    file_names = build_files_dict(data_path, mask_path, bval_path, bvec_path)

    if subj_folder:
        print(f"subj_folder: {subj_folder} \nfilenames: {file_names} \nsmall_delta: {small_delta} \nbig_delta: {big_delta} \ngmax: {gmax} \ngamma_val: {gamma_val}")
        axsi_main(subj_folder, file_names, small_delta, big_delta, gmax, gamma_val)


def build_files_dict(*paths):
    keys = ["data", "mask", "bval", "bvec"]
    filenames = {}
    if len(keys) != len(paths):
        print("Wrong number of input paths, make sure paths to data, mask, bval and bvec exists")
    for i in range(len(keys)):
        filenames[keys[i]] = assert_path(paths[i])
    return filenames
   


def assert_path(s: str):
    p = Path(s)
    if not p.exists():
        print(f"Path given is not valid: {s}")
        p = None
    print(p)
    return p


def run_from_terminal():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='AxSI for MRI data')

    # Define command-line arguments
    parser.add_argument('--subj_folder', type=Path, help='Path to the subject folder', required=True)
    parser.add_argument('--data', type=Path, help='Path to the data file', required=True)
    parser.add_argument('--bval', type=Path, help='Path to the bval file', required=True)
    parser.add_argument('--bvec', type=Path, help='Path to the bvec file', required=True)
    parser.add_argument('--mask', type=Path, help='Path to the mask file', required=True)
    parser.add_argument('--small_delta', type=float, help='Gradient duration in miliseconds', default=15)
    parser.add_argument('--big_delta', type=float, help='Time to scan (time interval) in milisecond', default=45)
    parser.add_argument('--gmax', type=float, help='Gradient maximum amplitude in G/cm', default=7.9)
    parser.add_argument('--gamma_val', type=int, help='Gyromagnetic ratio', default=4257)
    parser.add_argument('--num_of_processed', type=int, help='Number of processes to run in parallel', default=1)
    parser.add_argument('--preprocessed', action='store_true', help='Specify if data is preprocessed')
    parser.add_argument('--save_files', action='store_true', help='Specify if data is preprocessed')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values using dot notation
    subj_folder = assert_path(args.subj_folder)
    file_names = build_files_dict(args.data, args.mask, args.bval, args.bvec)
    small_delta = args.small_delta
    big_delta = args.big_delta
    gmax = args.gmax
    gamma_val = args.gamma_val

    if subj_folder:
        axsi_main(subj_folder, file_names, small_delta, big_delta, gmax, gamma_val)


if __name__ == "__main__":
    run_from_terminal()
