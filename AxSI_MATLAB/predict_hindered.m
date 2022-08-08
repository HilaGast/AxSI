function [decay, MD0] = predict_hindered(dt, mask, bval, grad_dirs, sMD, bshell)
       
    [X, Y, Z] = size(mask);
    len_bval = length(bval);

    MD0 = zeros(X*Y*Z,1);
    dt = 1000 .* dt;
    decay = zeros([X*Y*Z len_bval]);
    dt_maps(:,:,:,1,1) = dt(:,:,:,1);
    dt_maps(:,:,:,1,2) = dt(:,:,:,2);
    dt_maps(:,:,:,1,3) = dt(:,:,:,3);
    dt_maps(:,:,:,2,1) = dt(:,:,:,2);
    dt_maps(:,:,:,2,2) = dt(:,:,:,4);
    dt_maps(:,:,:,2,3) = dt(:,:,:,5);
    dt_maps(:,:,:,3,1) = dt(:,:,:,3);
    dt_maps(:,:,:,3,2) = dt(:,:,:,5);
    dt_maps(:,:,:,3,3) = dt(:,:,:,6);    
    bloc = find(bshell >= 1);
    bshell_nonzero = bshell(bloc);
    midi = sMD(:,:,:,bloc-1);
    midi = reshape(midi, [X*Y*Z,length(bloc)]);
    dt_maps = reshape(dt_maps,[X*Y*Z, size(dt_maps, 4), size(dt_maps, 5)]);
    
    parfor i = 1 : X*Y*Z
        if mask(i)
            f = fit(double(bshell_nonzero'),squeeze(midi(i,:))','exp1');
            fac = f.a ./ midi(i,1);
            d_mat = squeeze(1.0 * fac .* dt_maps(i, : ,:));    
            for j = 1 : len_bval
                decay(i, j) = exp(-max(bval) .* (grad_dirs(j,:) * d_mat * grad_dirs(j,:)'));
            end
            MD0(i) = 1.0*f.a;
        end
    end
    
    decay = reshape(decay,[X,Y,Z,length(bval)]);
    MD0 = reshape(MD0,[X,Y,Z]);
end












