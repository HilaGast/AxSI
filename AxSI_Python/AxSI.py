# The main file for running AxCaliber analysis
# Use DTI1000, shellDTI and AxSI

import os
from pathlib import Path
from mri_scan import Scan
from ax_dti import dti
from calc_axsi import calc_axsi


def init_axsi(subj_folder: Path, file_names, *params) -> Scan:
    print("Starting AxSI")

    # make dir subj_folder
    if not os.path.exists(f'{subj_folder}{os.sep}AxSI'):
        os.mkdir(f'{subj_folder}{os.sep}AxSI')

    scan = Scan(file_names, *params)    # init Scan object
    return scan



def axsi_main(subj_folder: Path, file_names, *params) -> None:
    # initialize analysis
    scan = init_axsi(subj_folder, file_names, *params) # return Scan object
    shell_dti = []                  # array for DTI objects for each bvalue
    bval_shell = scan.get_shell()   # array of unique nonzero bvalues

    # registration?

    # run DTI1000
    dti1000 = dti(scan, 1.0, is1000=True, subj_folder=subj_folder)
    # shell DTI - for each bvalue except zero
    for i in range(len(bval_shell)):
        bvalue = bval_shell[i]      # get next bvalue
        ax_dti = dti(scan, bvalue)  # run DTI
        shell_dti.append(ax_dti)    # store result object in array
    # run AxSI analysis
    calc_axsi(scan, dti1000, shell_dti, subj_folder)
