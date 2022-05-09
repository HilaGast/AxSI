function decay=predictR_singleR(theta_q, phi_q, vec, R_q, delta, smalldel, grad_dirs, B0p, R, MD250)

decay=zeros([length(grad_dirs) length(R)]);
B0=B0p;
a=R;
l_q=numel(R_q);
l_a=numel(a);
R_mat=repmat(a,[l_q 1]);
gamma=ones(size(a));
gamma_matrix=repmat(gamma,[l_q 1]);
D_r=repmat(MD250,[1 160]);
D_r=repmat(D_r,[l_q 1]);
M0=B0(1);
[phi_N, theta_N, ~]=cart2sph(vec(1), vec(2), -vec(3));
factor_angle_term_par=abs(cos(theta_q).*cos(theta_N).*...
    cos(phi_q-phi_N)+...
    sin(theta_q)*sin(theta_N));
factor_angle_term_perp=sqrt(1-factor_angle_term_par.^2);
q_par_sq=(R_q.*factor_angle_term_par).^2;
q_par_sq_matrix=repmat(q_par_sq,[1,l_a]);
q_perp_sq=(R_q.*factor_angle_term_perp).^2;
q_perp_sq_matrix=repmat(q_perp_sq,[1,l_a]);

E=exp(-4*pi^2*q_perp_sq_matrix.*R_mat.^2);
E=exp(-4*pi^2*q_par_sq_matrix.*(delta-smalldel/3).*D_r).*E;

decay=M0.*(E.*gamma_matrix);


