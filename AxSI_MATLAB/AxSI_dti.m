function [fa, md, dt, eigval, eigvec]=AxSI_dti(bval, bvec, data, bv, mask)
    norm_bvec = zeros(length(bval),size(bvec,2));
    for i = 1:length(bval)
        norm_bvec(i,:) = bvec(i,:) ./ norm(bvec(i,:));
    end

    bval_real = bval * 1000;
    b0locs = find(bval_real == 0);
    bvlocs = find(bval_real == bv * 1000);

    signal_0 = data(:,:,:,b0locs);
    signal = data(:,:,:,bvlocs);
    bval_real = bval_real(bvlocs);
    norm_bvec = norm_bvec(bvlocs,:);

    signal_0 = mean(signal_0, 4);

    b = zeros([3 3 size(norm_bvec,1)]);
    for i = 1:size(norm_bvec,1)
        b(:,:,i)=bval_real(i) * norm_bvec(i,:)' * norm_bvec(i,:);
    end

    signal_log = zeros(size(signal),'double');
    for i = 1:size(norm_bvec,1)
        signal_log(:,:,:,i) = log((signal(:,:,:,i) ./ signal_0)+eps);
        if ~isreal(log((signal(:,:,:,i) ./ signal_0) + eps))
            true
        end
    end

    bval_mat = squeeze([b(1,1,:), 2 * b(1,2,:), 2 * b(1,3,:), b(2,2,:), 2 * b(2,3,:), b(3,3,:)])';

    [X, Y, Z] = size(mask);
    signal_log = reshape(signal_log, [X*Y*Z, size(signal_log, 4)]);
    dt = zeros([X*Y*Z 6], 'single');
    eigval = zeros([X*Y*Z 3], 'single');
    fa = zeros([X*Y*Z 1], 'single');
    md = zeros([X*Y*Z 1], 'single');
    eigvec = zeros([X*Y*Z 3], 'single');

    parfor i = 1:X*Y*Z
        if mask(i)
            signal_log_i = -squeeze(signal_log(i,:));
            signal_log_i_norm=bval_mat \ signal_log_i';
            dt_i = [signal_log_i_norm(1) signal_log_i_norm(2) signal_log_i_norm(3); signal_log_i_norm(2) signal_log_i_norm(4) signal_log_i_norm(5); signal_log_i_norm(3) signal_log_i_norm(5) signal_log_i_norm(6)];
    
            [eigen_vec_i,diag_val] = eig(dt_i); 
            eigen_val_i = diag(diag_val);
            [~,index] = sort(eigen_val_i);
            eigen_val_i = eigen_val_i(index) * 1000; 
            eigen_vec_i = eigen_vec_i(:, index);
    
            eigen_val_i_org = eigen_val_i;
            if ((eigen_val_i(1)<0) && (eigen_val_i(2) < 0) && (eigen_val_i(3) < 0)), eigen_val_i = abs(eigen_val_i);end
            if (eigen_val_i(1) <= 0), eigen_val_i(1) = eps; end
            if (eigen_val_i(2) <= 0), eigen_val_i(2) = eps; end
            if (eigen_val_i(3) <= 0), eigen_val_i(3) = eps; end

            md_i = (eigen_val_i(1) + eigen_val_i(2) + eigen_val_i(3)) / 3;

            fa_i = sqrt(1.5) * (sqrt((eigen_val_i(1) - md_i) .^ 2 + (eigen_val_i(2) - md_i) .^ 2 + (eigen_val_i(3) - md_i) .^ 2) ./ sqrt(eigen_val_i(1) .^ 2 + eigen_val_i(2) .^ 2 + eigen_val_i(3) .^ 2));
    
            md(i) = md_i;
            eigval(i,:) = eigen_val_i;
            dt(i,:) = [dt_i(1:3) dt_i(5:6) dt_i(9)];
            fa(i)=fa_i;
            eigvec(i,:)=eigen_vec_i(:,end)*eigen_val_i_org(end);
        end
    end

    dt = reshape(dt,[X,Y,Z,size(dt, 2)]);
    eigval = reshape(eigval,[X,Y,Z,size(eigval, 2)]);
    fa = reshape(fa,[X,Y,Z]);
    md = reshape(md,[X,Y,Z]);
    eigvec = reshape(eigvec,[X,Y,Z,size(eigvec, 2)]);

end
