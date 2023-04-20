function [rdwi, pcsf, ph, pfr, pasi, paxsi] = Calc_AxSI(subj_folder, file_names)

    clearvars -except subj_folder file_names
               
    load(fullfile(subj_folder,'AxSI','Charmed_vars.mat'),'bval', 'mask', 'grad_dirs', 'rdwi','scan_param','eigvec1000', 'dt1000', 'sMD','bshell', 'sEigval');
    [X, Y, Z] = size(mask);

    b0_dwi_mean = squeeze(mean(rdwi(:,:,:,find(bval==0)),4)); 

    sMDi=squeeze(sEigval(:,:,:,3,:));
    DT1000 = real(dt1000);
    eigvec1000 = real(eigvec1000);
    [decay_hindered, MD0]=predict_hindered(DT1000, mask, bval, grad_dirs, sMDi, bshell);

    pixpredict_csf = zeros(1,length(bval));
    d_mat = [4 0 0; 0 4 0; 0 0 4];
    for k = 1 : length(bval)
        pixpredict_csf(k) = exp(-4 .* (grad_dirs(k, :) * d_mat * grad_dirs(k,:)'));
    end

    prcsf = predict_csf(sMD(:,:,:,1));

    A=ones(1,162);
    b=1;
    n = 160;
    L = eye(n)+[[zeros(1,n-1);-eye(n-1)],zeros(n,1)];
    L = [L,zeros(n,2)];
    Aeq=ones(1,162);
    beq=1;
    al = 3;
    be = 2;
    add_vals = 0.1:0.2:32;
    yd=gampdf(add_vals,al, be);
    yd=yd.*(pi*(add_vals/2).^2);
    yd=yd./sum(yd);
    
    eigvec1000 = reshape(eigvec1000,[X*Y*Z,size(eigvec1000,4)]);
    b0_dwi_mean = reshape(b0_dwi_mean,[X*Y*Z,1]);
    rdwi = reshape(rdwi,[X*Y*Z,size(rdwi,4)]);
    decay_hindered = reshape(decay_hindered,[X*Y*Z,size(decay_hindered,4)]);
    MD0 = reshape(MD0,[X*Y*Z,1]);
    prcsf = reshape(prcsf,[X*Y*Z,1]);

    CMDfh = zeros(X*Y*Z,1);
    CMDfr = zeros(X*Y*Z,1);
    CMDcsf = zeros(X*Y*Z,1);
    ph = zeros(X*Y*Z,1);
    pcsf = zeros(X*Y*Z,1);
    pfr = zeros(X*Y*Z,1);
    pasi = zeros(X*Y*Z,1);
    paxsi = zeros(X*Y*Z,length(add_vals));
    vCSF=pixpredict_csf(:);
    parfor i = 1 : X * Y * Z
        if mask(i)
            eigvec1000_i = squeeze(eigvec1000(i, :));
            b0_mean = squeeze(b0_dwi_mean(i));
            ydata = squeeze(rdwi(i, :))';
            pixpredictH = squeeze(decay_hindered(i, :));
            vH = pixpredictH(:);
            vH(vH > 1) = 0;
            decayR = predict_restricted(scan_param, eigvec1000_i, b0_mean, add_vals/2, MD0(i));
            vR = double(decayR);
            vR = vR ./ max(vR(:));
            vRes = vR * yd';
            x0 = double([0.5  5000]);
            min_val = [ 0  0];
            max_val = [ 1  20000];
            h = optimset('DiffMaxChange', 1e-1, 'DiffMinChange', 1e-3, 'MaxIter', 20000, ...
                'MaxFunEvals', 20000, 'TolX', 1e-6, ...
                'TolFun', 1e-6, 'Display', 'off');
            [parameter_hat, ~, ~, ~, ~] = lsqnonlin('reg_func', ...
                x0, min_val, max_val, h, ydata, vH, vRes, vCSF, prcsf(i));
            CMDfh(i) = parameter_hat(1);
            CMDfr(i) = 1 - parameter_hat(1) - prcsf(i);
            CMDcsf(i) = prcsf(i);
    
            vdata = ydata ./ parameter_hat(2);
            preds = [vR vH vCSF];
            lb = zeros(size(yd'));
            ub = ones(size(yd'));
            lb(161) = 0;
            ub(161) = 1;
            lb(162) = prcsf(i) - 0.02;
            ub(162) = prcsf(i) + 0.02;
            Lambda = 1;
            Xprim = [preds; sqrt(Lambda) * L];
            yprim = [vdata; zeros(160, 1)];
            options = optimoptions('lsqlin', 'Display',  'off');
            x = lsqlin(Xprim, yprim, A, b, Aeq, beq, lb, ub, yd, options);
            if isempty(x) == 1
                x = zeros(160, 1);
            end
            x(x < 0) = 0;
            a_h = x(161);
            a_csf = prcsf(i);
            a_fr = 1 - a_csf - a_h;
            if a_fr < 0
                a_fr = 0;
            end
            nx = x(1 : 130);
            nx = nx ./ sum(nx);
            ph(i) = a_h;
            pcsf(i) = a_csf;
            pfr(i) = sum(a_fr);
            pasi(i) = sum(nx .* add_vals(1 : 130)');      
            paxsi(i, :) = x(1 : 160);
        end
    end

rdwi = reshape(rdwi, [X, Y, Z, size(rdwi, 2)]);
ph = reshape(ph, [X, Y, Z]);
pcsf = reshape(pcsf, [X, Y, Z]);
pfr = reshape(pfr, [X, Y, Z]);
pasi = reshape(pasi, [X, Y, Z]);
paxsi = reshape(paxsi, [X, Y, Z, size(paxsi, 2)]);


if save_files
    info_nii = niftiinfo(file_names.data);
    info_nii.Datatype = 'double'; 
    info_nii.BitsPerPixel = 64; 
    info_nii.ImageSize = size(pasi); 
    info_nii.PixelDimensions(4)=[];
    fn = fullfile(subj_folder,'AxSI','ADD');
    niftiwrite(double(pasi),fn,info_nii,'Compressed',true);

    fn = fullfile(subj_folder,'AxSI','pfr');
    niftiwrite(double(pfr),fn,info_nii,'Compressed',true);

    fn = fullfile(subj_folder,'AxSI','ph');
    niftiwrite(double(ph),fn,info_nii,'Compressed',true);

    fn = fullfile(subj_folder,'AxSI','pcsf');
    niftiwrite(double(pcsf),fn,info_nii,'Compressed',true);

    info_nii = niftiinfo(file_names.data);
    info_nii.Datatype = 'double'; 
    info_nii.BitsPerPixel = 64; 
    info_nii.ImageSize = size(paxsi); 
    fn = fullfile(subj_folder,'AxSI','ADD_allvalues');
    niftiwrite(double(paxsi),fn,info_nii,'Compressed',true);

    info_nii = niftiinfo(file_names.data);
    info_nii.Datatype = 'double'; 
    info_nii.BitsPerPixel = 64; 
    info_nii.ImageSize = size(rdwi); 
    fn = fullfile(subj_folder,'rdata');
    niftiwrite(double(rdwi),fn,info_nii,'Compressed',true);
end

cd(subj_folder);
delete('AxSI/Charmed_vars.mat');
      
end
