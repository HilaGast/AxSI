function dwi_simulates = simulate_charmed(data, scan_param, fa1000, dt1000, eigvec1000, grad_dirs, mask, add_vals, gamma_dist, bval, md1000, preprocessed)

    b0_data = data(:,:,:,scan_param.nb0); 
    b0_map = mean(b0_data,4);
    if preprocessed==1
        rb0_data = b0_data;
    else
        [optimizer,metric] = imregconfig('multimodal');
        optimizer.InitialRadius = optimizer.InitialRadius/1000;
        optimizer.MaximumIterations = 300;
        rb0_data = zeros(size(b0_data));
        for i=1:scan_param.nb0
            [rb0_data(:,:,:,i)] = imregister(b0_map, b0_data(:,:,:,i), 'affine', optimizer, metric);
            rb0_data(:,:,:,i)=smooth3(rb0_data(:,:,:,i), 'box', [3 3 3]);
        end
    end
    rb0_map = mean(rb0_data,4);

    dwi_simulates = SimChm(rb0_map, fa1000, dt1000, real(eigvec1000), grad_dirs, mask, scan_param, add_vals, gamma_dist, bval, md1000);

    for i = 1:length(bval)
        dwi_sim1 = dwi_simulates(:, :, :, i);
        dwi_sim1_vec = dwi_sim1(:);
        dwi_sim1_vec = dwi_sim1_vec(dwi_sim1_vec > 0);
        dwi_sim1_vec_sorted = sort(dwi_sim1_vec);
        th_loc = round(0.995 * length(dwi_sim1_vec_sorted));
        dwi_sim1(dwi_sim1>(dwi_sim1_vec_sorted(th_loc))) = dwi_sim1_vec_sorted(th_loc);
        dwi_simulates(:,:,:,i) = dwi_sim1;
    end

end
