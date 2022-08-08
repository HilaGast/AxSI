function scan_param = scan_param_vals(b0_locs, small_delta, big_delta, gmax, gamma_val, grad_dirs)
    
    [phi_q, theta_q, R_q]=cart2sph(grad_dirs(:,1), grad_dirs(:,2), -grad_dirs(:,3));
    scan_param.nb0 = b0_locs;
    scan_param.small_delta = small_delta;
    scan_param.big_delta = big_delta;
    scan_param.gmax = gmax;
    scan_param.theta = theta_q;
    scan_param.phi = phi_q;
    scan_param.max_q = gamma_val.*scan_param.small_delta.*scan_param.gmax./10e6;
    r_q = R_q.*scan_param.max_q;
    scan_param.bval = 4.*pi^2.*r_q.^2.*(scan_param.big_delta-scan_param.small_delta/3);
    scan_param.r = r_q;
