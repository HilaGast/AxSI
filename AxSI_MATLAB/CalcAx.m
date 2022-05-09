function CalcAx(main_path, file_names)
clearvars -except main_path file_names
ala = (3);
bea = (2);


load(fullfile(main_path,'AxSI','alldata.mat'),'bval', 'mask', 'grad_dirs', 'phi_q','theta_q', 'R_q', 'rDWI','Param','vec', 'DT1000', 'sMD','bshel', 'sEigval');

pfr=zeros(size(mask));
ph=zeros(size(mask));
pcsf=zeros(size(mask));
pasi=zeros(size(mask));
paxsi=zeros([size(mask) 160]);
CMDfr=zeros(size(mask));
CMDfh=zeros(size(mask));
CMDcsf=zeros(size(mask));
pixpredictCSF=zeros(1,length(bval));
%load maskCC
mask2=mask;
B0s=squeeze(mean(rDWI(:,:,:,find(bval==0)),4)); %B0s=squeeze(mean(rDWI(:,:,:,find(bval==0)),4));

sMDi=squeeze(sEigval(:,:,:,3,:));
DT1000 = real(DT1000);
vec = real(vec);
[decayH, MDi0]=predictH(DT1000, mask2, bval, grad_dirs, sMDi, bshel);

D_mat=[4 0 0; 0 4 0; 0 0 4];
for k=1:length(bval)
    pixpredictCSF(k)=exp(-4.*(grad_dirs(k,:)*D_mat*grad_dirs(k,:)'));
end
prcsf=predictCSF(sMD(:,:,:,1));
ax=0.1:0.2:32;
maskmap=zeros(size(mask));

A=ones(1,162);
b=1;
N = 160;
L = eye(N)+[[zeros(1,N-1);-eye(N-1)],zeros(N,1)];
L = [L,zeros(N,2)];
Aeq=ones(1,162);
beq=1;

[X, Y, Z] = size(mask2);

for ab=1:length(ala)
    al = ala(ab);
    be = bea(ab);
    yd=gampdf(ax,al, be);
    yd=yd.*(pi*(ax/2).^2);
    yd=yd./sum(yd);

    %h1 = waitbar(0,'Charmed/AxCaliber Analysis...');

    
    
    maskmap = reshape(maskmap,[X*Y*Z,1]);
    vec = reshape(vec,[X*Y*Z,size(vec,4)]);
    B0s = reshape(B0s,[X*Y*Z,1]);
    rDWI = reshape(rDWI,[X*Y*Z,size(rDWI,4)]);
    decayH = reshape(decayH,[X*Y*Z,size(decayH,4)]);
    MDi0 = reshape(MDi0,[X*Y*Z,1]);
    CMDfh = reshape(CMDfh,[X*Y*Z,1]);
    CMDfr = reshape(CMDfr,[X*Y*Z,1]);
    CMDcsf = reshape(CMDcsf,[X*Y*Z,1]);
    prcsf = reshape(prcsf,[X*Y*Z,1]);
    ph = reshape(ph,[X*Y*Z,1]);
    pcsf = reshape(pcsf,[X*Y*Z,1]);
    pfr = reshape(pfr,[X*Y*Z,1]);
    pasi = reshape(pasi,[X*Y*Z,1]);
    paxsi = reshape(paxsi,[X*Y*Z,size(paxsi,4)]);
    
    parfor i=1:X*Y*Z
        if mask2(i)
            %waitbar(i / X*Y*Z);
            maskmap(i)=i;
            fvec=squeeze(vec(i,:));
            B0sp=squeeze(B0s(i));
            ydata=squeeze(rDWI(i,:))';
            pixpredictH=squeeze(decayH(i,:));
            vH=pixpredictH(:);
            vH(vH>1)=0;
            vCSF=pixpredictCSF(:);
            decayR=predictR_singleR(theta_q, phi_q, fvec, R_q, Param.bigdel, Param.smalldel, grad_dirs, B0sp, ax/2, MDi0(i));
            vR=double(decayR);
            vR=vR./max(vR(:));
            vRes=vR*yd';
            x0=double([0.5  5000]);
            min_val=[ 0  0];
            max_val=[ 1  20000];
            h=optimset('DiffMaxChange',1e-1,'DiffMinChange',1e-3,'MaxIter',20000,...
                'MaxFunEvals',20000,'TolX',1e-6,...
                'TolFun',1e-6, 'Display', 'off');
            [parameter_hat,~,~,~,~]=lsqnonlin('regression_function',...
                x0,min_val,max_val,h, ydata, vH, vRes, vCSF, prcsf(i)) ;
            CMDfh(i)=parameter_hat(1);
            CMDfr(i)=1-parameter_hat(1)-prcsf(i);
            CMDcsf(i)=prcsf(i);
    
            vdata=ydata./parameter_hat(2);
            preds=[vR vH vCSF];
            lb=zeros(size(yd'));
            ub=ones(size(yd'));
            lb(161)=0; %parameter_hat(1);
            ub(161)=1;
            lb(162)=prcsf(i)-0.02;
            ub(162)=prcsf(i)+0.02;
            Lambda=1;
            Xprim=[preds; sqrt(Lambda)*L];
            yprim=[vdata;zeros(160,1)];
            options=optimoptions('lsqlin', 'Display',  'off');
            x = lsqlin(Xprim,yprim,A,b,Aeq,beq,lb,ub, yd, options);
            if isempty(x)==1
                x=zeros(160,1);
            end
            x(x<0)=0;
            a_h=x(161); %parameter_hat(1);
            a_csf=prcsf(i);
            a_fr=1-a_csf-a_h;
            if a_fr<0
                a_fr=0;
            end
            nx=x(1:130);
            nx=nx./sum(nx);
            ph(i)=a_h;
            pcsf(i)=a_csf;
            pfr(i)=sum(a_fr);
            pasi(i)=sum(nx(1:130).*ax(1:130)');      
            paxsi(i, :)=x(1:160);
        end
    end
maskmap = reshape(maskmap,[X,Y,Z]);
vec = reshape(vec,[X,Y,Z,size(vec,2)]);
B0s = reshape(B0s,[X,Y,Z]);
rDWI = reshape(rDWI,[X,Y,Z,size(rDWI,2)]);
decayH = reshape(decayH,[X,Y,Z,size(decayH,2)]);
MDi0 = reshape(MDi0,[X,Y,Z]);
CMDfh = reshape(CMDfh,[X,Y,Z]);
CMDfr = reshape(CMDfr,[X,Y,Z]);
CMDcsf = reshape(CMDcsf,[X,Y,Z]);
prcsf = reshape(prcsf,[X,Y,Z]);
ph = reshape(ph,[X,Y,Z]);
pcsf = reshape(pcsf,[X,Y,Z]);
pfr = reshape(pfr,[X,Y,Z]);
pasi = reshape(pasi,[X,Y,Z]);
paxsi = reshape(paxsi,[X,Y,Z,size(paxsi,2)]);

info_nii = niftiinfo(file_names.data);

info_nii.Datatype = 'double'; 
info_nii.BitsPerPixel = 64;
info_nii.ImageSize = size(pasi); 
info_nii.PixelDimensions(4)=[];

fn = fullfile(main_path,'AxSI','3_2_AxPasi7');
niftiwrite(double(pasi),fn,info_nii,'Compressed',true);
cd(main_path);
delete('AxSI/alldata.mat');



    
    
end
