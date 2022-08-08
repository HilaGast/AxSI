function [data, info_nii, bval, bvec, mask] = load_diff_files(subj_folder, file_names)
    
    data = double(niftiread(file_names.data)); 
    info_nii = niftiinfo(file_names.data);
    data(data<0) = 0;

    A = fopen(file_names.bval); 
    bval=fscanf(A,'%f');
    A = fopen(file_names.bvec);
    bvec=fscanf(A,'%f')';
    bvec=reshape(bvec,[length(bval) 3]);
    bvec=bvec(:,[1 2 3]);
    
    bval=(round(2*bval,-2)./2000);

    mask = niftiread(file_names.mask);

    % Remove bval<1000 from calculation:
    blow_locs = find(bval>0 & bval<1);
    bval(blow_locs)=[];
    bvec(blow_locs,:)=[];
    data(:,:,:,blow_locs)=[];

end