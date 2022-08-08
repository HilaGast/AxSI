function decay = predict_restricted(scan_param, vec, b0_mean, av, MD0i)

    l_q = numel(scan_param.r);
    l_a = numel(av);
    r_mat = repmat(av, [l_q 1]);
    gamma = ones(size(av));
    gamma_matrix = repmat(gamma, [l_q 1]);
    d_r = repmat(MD0i, [1 160]);
    d_r = repmat(d_r, [l_q 1]);
    [phi_n, theta_n, ~] = cart2sph(vec(1), vec(2), -vec(3));
    factor_angle_term_par = abs(cos(scan_param.theta) .* cos(theta_n) .* ...
    cos(scan_param.phi-phi_n) + ...
    sin(scan_param.theta) * sin(theta_n));
    factor_angle_term_perp=sqrt(1-factor_angle_term_par.^2);
    q_par_sq=(scan_param.r .* factor_angle_term_par).^2;
    q_par_sq_matrix=repmat(q_par_sq,[1,l_a]);
    q_perp_sq=(scan_param.r .*factor_angle_term_perp).^2;
    q_perp_sq_matrix=repmat(q_perp_sq,[1,l_a]);

    E=exp(-4*pi^2*q_perp_sq_matrix.*r_mat.^2);
    E=exp(-4*pi^2*q_par_sq_matrix.*(scan_param.big_delta-scan_param.small_delta/3).*d_r).*E;

    decay=b0_mean.*(E.*gamma_matrix);

end