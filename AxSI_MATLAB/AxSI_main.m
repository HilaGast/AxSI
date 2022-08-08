function [rdwi, pcsf, ph, pfr, pasi, paxsi] = AxSI_main(subj_folder, file_names,small_delta, big_delta, gmax, gamma_val, preprocessed, save_files)

%     Input:
%         subj_folder : str. The folder path of subject files
%         file_names : struct. File names (full path) of files needed for
%         analysis.
%             file_names.bval : .bval file.
%             file_names.bvec : .bvec file.
%             file_names.mask : .nii or .nii.gz file with brain mask
%             file_names.data : .nii or .nii.gz file with diffusion data.
%             Multi b-shell diffusion scans. Must have a b = 1000 shell.
%         small_delta : float. Gradient duration in miliseconds.
%         big_delta : float. Time to scan (time interval) in milisecond.
%         gmax : float. Gradient maximum amplitude in G/cm (or 1/10 mT/m)
%             gmax calculation: sqrt(bval*100/(7.178e8*0.03^2*(0.06-0.01)))
%         gamma_val : int. Gyromagnetic ratio
%             gamma_val=4257 ;
%         preprocessed : flag. Marked as true if the scan were already preprocessed for motion correction and between-volumes registration.
%         save_files : flag. Marked as true to save files during the process.
    
    
    mkdir(fullfile(subj_folder,'AxSI'));
    cd(subj_folder);

% Load Data:
    [data, info_nii, bval, bvec, mask] = load_diff_files(subj_folder, file_names);

% Prepare data for DTI on shell bvalue=1000:
    [rdata1000, bval1000, bvec1000] = data_prep_4_dti1000(data, bval, bvec, preprocessed);

% DTI1000 to calculate FA:
    [fa1000, md1000, dt1000, ~, eigvec1000]= AxSI_dti(double(bval1000), bvec1000, rdata1000, 1, mask);

% Save FA & MD:
    if save_files==1
        info_nii.Datatype = 'double'; 
        info_nii.BitsPerPixel = 64; 
        info_nii.ImageSize = size(fa1000); 
        info_nii.PixelDimensions(4)=[];

        fn = fullfile(subj_folder,'AxSI','FA_mat');
        niftiwrite(double(fa1000),fn,info_nii, 'Compressed', true);
        fn = fullfile(subj_folder,'AxSI','MD_mat');
        niftiwrite(double(md1000),fn,info_nii, 'Compressed', true);
    end

% Experimental Parameters and calculations
    b0_locs = find(bval==0);
    bval=double(bval);
    bv_norm=sqrt(bval./max(bval));
    grad_dirs=bvec.*repmat(bv_norm, [1 3]);

    scan_param = scan_param_vals(b0_locs, small_delta, big_delta, gmax, gamma_val, grad_dirs);

    add_vals=0.1:0.2:32;
    gamma_dist=gampdf(add_vals,2,2);
    gamma_dist=gamma_dist./sum(gamma_dist);

% Simulate CHARMED for motion correction
    dwi_simulates = simulate_charmed(data, scan_param, fa1000, dt1000, eigvec1000, grad_dirs, mask, add_vals, gamma_dist, bval, md1000, preprocessed);

% UNDISTORT - Registration and gradient reorientation
    [rdwi, grad_dirs] = AxSI_undistort(grad_dirs, dwi_simulates, data,bval, preprocessed);

% calculate DTI parameter for each b shel
    bshell = unique(bval)';

    sFA = zeros([size(mask) length(bshell) - 1]);
    sMD = zeros([size(mask) length(bshell) - 1]);
    sEigval = zeros([size(mask) 3 length(bshell) - 1]);
    sEigvec = zeros([size(mask) 3 length(bshell) - 1]);
    sDT = zeros([size(mask) 6 length(bshell) - 1]);

    for i = 2 : length(bshell)
        [sFA(:,:,:,i-1), sMD(:,:,:,i-1), sDT(:,:,:,:,i-1), sEigval(:,:,:,:,i-1), sEigvec(:,:,:,:,i-1)] = AxSI_dti(bval, grad_dirs, rdwi, bshell(i), mask);
    end

    save(fullfile(subj_folder,'AxSI','Charmed_vars.mat'),'bval', 'mask', 'grad_dirs', 'scan_param', 'rdwi','eigvec1000', 'dt1000', 'sMD','bshell', 'sEigval','-v7.3'); 
    
% AxSI:
    [rdwi, pcsf, ph, pfr, pasi, paxsi] = Calc_AxSI(subj_folder, file_names);