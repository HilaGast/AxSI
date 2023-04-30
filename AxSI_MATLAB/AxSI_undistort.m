function [rdwi, grad_dirs] = AxSI_undistort(grad_dirs, dwi_simulates, data, bval, scan_param, preprocessed)

    if preprocessed
        rdwi = zeros(size(data));
        for i = 1 : length(bval)
            rdwi(:, :, :, i) = smooth3(data(:, :, :, i), 'gaussian', 5);
        end
    else
        [optimizer,metric] = imregconfig('multimodal');
        optimizer.InitialRadius = optimizer.InitialRadius/100;
        optimizer.MaximumIterations = 200;

        h = waitbar(0,'UNDISTORT...');
        for i = 1:length(bval)
            waitbar(i / length(grad_dirs));
            dwi_sim_i = dwi_simulates(:,:,:,i); 
            dwi_sim_i = real(dwi_sim_i);
            data_slice = data(:, :, :, i);
            tform = imregtform(data_slice, dwi_sim_i, 'affine', optimizer, metric);
            newgrad(i, :) = [grad_dirs(i, :) sqrt(sum(grad_dirs(i,:) .^ 2))] * tform.T;
            grad_dirs(i, :) = grad_dirs(i, :) * tform.T(1:3, 1:3);
            rdwi(:, :, :, i) = imwarp(data_slice, tform, 'OutputView', imref3d(size(data_slice)));
        end
        close(h)

        grad_dirs(1:scan_param.nb0, :) = zeros(scan_param.nb0, 3);

        for i = 1 : length(bval)
            rdwi(:, :, :, i) = smooth3(rdwi(:, :, :, i), 'gaussian', 5);
        end
    end
end
