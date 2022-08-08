% Simulate CHARMED
function [dwis ]=SimChm(B0, FA, DTmaps, mag, vec, grad_dirs, mask, maxq, delta, smalldel, theta_q, phi_q, R, weight, R_q, bval, MD250)
[xlocs ylocs zlocs]=ind2sub(size(mask), find(mask==1));
dwis=zeros([size(mask) length(grad_dirs)]);
bmatrix=zeros([3 3 length(bval)]);
for i=1:length(bval)
bmatrix(:,:,i)=bval(i)*grad_dirs(i,:)'*grad_dirs(i,:);
end
a=R;
l_a=length(a);
l_q=length(R_q);
R_mat=repmat(a,[l_q 1]);

gamma=weight./sum(weight);
gamma_matrix=repmat(gamma,[l_q 1]);

for i=1:length(zlocs);
    M0=B0(xlocs(i), ylocs(i), zlocs(i));
 
    f_h=(1-FA(xlocs(i), ylocs(i), zlocs(i)));
    tf_r=1-f_h;
    f_r=mag(xlocs(i), ylocs(i), zlocs(i));
    [phi_N theta_N R_N]=cart2sph(vec(xlocs(i), ylocs(i), zlocs(i),1), vec(xlocs(i), ylocs(i), zlocs(i),2), -vec(xlocs(i), ylocs(i), zlocs(i),3));
    D_r=MD250(xlocs(i), ylocs(i), zlocs(i));
    
    D_mat=[DTmaps(xlocs(i), ylocs(i), zlocs(i), 1) DTmaps(xlocs(i), ylocs(i), zlocs(i), 2) DTmaps(xlocs(i), ylocs(i), zlocs(i), 3); DTmaps(xlocs(i), ylocs(i), zlocs(i), 2) DTmaps(xlocs(i), ylocs(i), zlocs(i), 4) DTmaps(xlocs(i), ylocs(i), zlocs(i), 5); DTmaps(xlocs(i), ylocs(i), zlocs(i), 3) DTmaps(xlocs(i), ylocs(i), zlocs(i), 5) DTmaps(xlocs(i), ylocs(i), zlocs(i), 6)];
    for j=1:length(bval)
        E_h(j)=f_h.*exp(-4.*(grad_dirs(j,:)*(1000.*D_mat)*grad_dirs(j,:)'));       
    end
    
    
    factor_angle_term_par=abs(cos(theta_q).*cos(theta_N).*...
        cos(phi_q-phi_N)+...
        sin(theta_q)*sin(theta_N));
    factor_angle_term_perp=sqrt(1-factor_angle_term_par.^2);
    q_par_sq=(R_q.*factor_angle_term_par).^2;
    q_par_sq_matrix=repmat(q_par_sq,[1,l_a]);
    q_perp_sq=(R_q.*factor_angle_term_perp).^2;
    q_perp_sq_matrix=repmat(q_perp_sq,[1,l_a]);
    
    E=exp(-4*pi^2*q_perp_sq_matrix.*R_mat.^2);
    
    E1=exp(-4*pi^2*q_par_sq_matrix*(delta-smalldel/3)*D_r).*E;
    
    E_r=sum(E1.*gamma_matrix,2);
    
    dwis(xlocs(i), ylocs(i), zlocs(i),:)=M0.*(f_r.*E_r'+E_h);
end
