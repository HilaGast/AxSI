function dwi_simulates = simchm(rb0_map, fa, dt, vec, grad_dirs, mask, scan_param, add_vals, gamma_dist, bval, md)
    % R\a = add_vals
    % weight = gamma_dist

    [xlocs ylocs zlocs] = ind2sub(size(mask), find(mask==1));
    dwi_simulates = zeros([size(mask) length(bval)]);
    
    add_vals = add_vals/2;
    len_av = length(add_vals);
    len_r = length(scan_param.r);
    r_mat = repmat(add_vals,[len_r 1]);
    gamma_dist_norm = gamma_dist./sum(gamma_dist);
    gamma_matrix = repmat(gamma_dist_norm,[len_r 1]);

    for i = 1:length(zlocs)
        xi = xlocs(i); yi = ylocs(i); zi = zlocs(i);
        b0_signal = rb0_map(xi, yi, zi);
        hindered_fraction = (1-fa(xi, yi, zi));
        restricted_fraction = 1-hindered_fraction;
        [phi_sim, theta_sim, ~] = cart2sph(vec(xi, yi, zi, 1), vec(xi, yi, zi, 2), -vec(xi, yi, zi, 3));
        md_i = md(xi, yi, zi);
        dt_mat = [dt(xi, yi, zi, 1) dt(xi, yi, zi, 2) dt(xi, yi, zi, 3); dt(xi, yi, zi, 2) dt(xi, yi, zi, 4) dt(xi, yi, zi, 5); dt(xi, yi, zi, 3) dt(xi, yi, zi, 5) dt(xi, yi, zi, 6)];
        
        estimated_hindered = zeros(1,length(bval));
        for bi = 1:length(bval)
            estimated_hindered(bi) = hindered_fraction .* exp(-4 .* (grad_dirs(bi,:) * (1000.*dt_mat) * grad_dirs(bi,:)'));       
        end
    
    
        factor_angle_term_par = abs(cos(scan_param.theta) .* cos(theta_sim) .* ...
            cos(scan_param.phi-phi_sim) + sin(scan_param.theta) * sin(theta_sim));
        factor_angle_term_perp = sqrt(1 - factor_angle_term_par .^ 2);
        q_par_sq = (scan_param.r .* factor_angle_term_par) .^ 2;
        q_par_sq_matrix = repmat(q_par_sq, [1, len_av]);
        q_perp_sq = (scan_param.r .* factor_angle_term_perp) .^ 2;
        q_perp_sq_matrix = repmat(q_perp_sq, [1, len_av]);
    
        exp_q_perp = exp(-4 * pi ^ 2 * q_perp_sq_matrix .* r_mat .^ 2);
        exp_q_par = exp(-4 * pi ^ 2 * q_par_sq_matrix * (scan_param.big_delta - scan_param.small_delta / 3) * md_i) .* exp_q_perp;
    
        estimated_restricted = sum(exp_q_par .* gamma_matrix, 2);    
        dwi_simulates(xi, yi, zi, :) = b0_signal .* (restricted_fraction .* estimated_restricted' + estimated_hindered);
    end
end