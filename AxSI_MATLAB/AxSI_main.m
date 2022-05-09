function AxSI_main(main_path,file_names,inpsmldel, inpbigdel, inpgmax)
mkdir(fullfile(main_path,'AxSI'));
% load data
i1 = niftiread(file_names.data); 
info_nii = niftiinfo(file_names.data);
data = double(i1);
data(data<0) = 0;
A = fopen(file_names.bval); 
bval=fscanf(A,'%f');
A = fopen(file_names.bvec);
bvec=fscanf(A,'%f')';
bvec=reshape(bvec,[length(bval) 3]);
bvec=bvec(:,[1 2 3]);
if inpbigdel == 45 || inpbigdel == 43.1
    bval=(round(bval,-2)./1000);
else
    bval = bval./1000;
end
mask = niftiread(file_names.mask); 

% Remove bval<1000 from calculation:
blow_locs = find(bval>0 & bval<1);
bval(blow_locs)=[];
bvec(blow_locs,:)=[];
data(:,:,:,blow_locs)=[];

% DTI1000 FA calc
b1000locs=find(bval==1);
data1000=data(:,:,:,[1,b1000locs']);
rdata1000=zeros(size(data1000));
rdata1000(:,:,:,1)=data1000(:,:,:,1);
bvec1000=bvec([1 b1000locs'],:);
bval1000=bval([1 b1000locs']);
nbvec1000=zeros(size(bvec1000));

[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = optimizer.InitialRadius/300;
optimizer.MaximumIterations = 200;
h = waitbar(0,'Motion Correction for DTI...');
for i=1:(length(b1000locs)+1)
    waitbar(i/(length(b1000locs)+1));
    [rdata1000(:,:,:,i)] = imregister(data1000(:,:,:,i), rdata1000(:,:,:,1), 'affine', optimizer, metric);
    rdata1000(:,:,:,i)=smooth3(rdata1000(:,:,:,i), 'gaussian',5);
    tform = imregtform(data1000(:,:,:,i), rdata1000(:,:,:,1), 'affine', optimizer, metric);
    nbvec1000(i,:)=bvec1000(i,:)*tform.T(1:3,1:3);
end
close(h);
 

[FA1000, MD1000, DT1000, ~, Eigvec1000]=dti(double(bval1000), bvec1000, rdata1000, 1, mask);


info_nii.Datatype = 'double';
info_nii.BitsPerPixel = 64; 
info_nii.ImageSize = size(FA1000); 
info_nii.PixelDimensions(4)=[];


fn = fullfile(main_path,'AxSI','FA');
niftiwrite(double(FA1000),fn,info_nii, 'Compressed', true);
fn = fullfile(main_path,'AxSI','MD');
niftiwrite(double(MD1000),fn,info_nii, 'Compressed', true);

% Experimental Parameters and calculations
Param.nb0 = 1;
bval=double(bval);
bvfac=sqrt(bval./max(bval));
grad_dirs=bvec.*repmat(bvfac, [1 3]);
gamma=4257 ;
Param.smalldel = inpsmldel; %in ms
Param.bigdel = inpbigdel; % in ms
Param.TE = 89; %in ms
%Gmax calc: sqrt(bval*100/(7.178e8*0.03^2*(0.06-0.01)))
Param.maxG = inpgmax; % in G/cm = 1/10 mT/m
Param.ndir = length(grad_dirs);
Param.maxq = gamma.*Param.smalldel.*Param.maxG./10e6;
Param.qdirs = grad_dirs.*Param.maxq;
[phi_q, theta_q, R_q]=cart2sph(grad_dirs(:,1), grad_dirs(:,2), -grad_dirs(:,3));
R_q=R_q.*Param.maxq;
Param.bval=4.*pi^2.*R_q.^2.*(Param.bigdel-Param.smalldel/3);
Param.phi=phi_q;
Param.theta=theta_q;
Param.R=R_q;
ax=0.1:0.2:32;
gw3=gampdf(ax,2,2);
gw3=gw3./sum(gw3);

% Simulate CHARMED for motion correction
FixedB0=data(:,:,:,1,1);
[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = optimizer.InitialRadius/1000;
optimizer.MaximumIterations = 300;
FA=FA1000;
vec=Eigvec1000;
mag=FA;
DTmaps=DT1000; %(:,:,:,[1 2 3 5 6 9]);

B0s=data(:,:,:,1:Param.nb0,1);
rB0 = zeros(size(B0s));
for i=1:Param.nb0
    [rB0(:,:,:,i)] = imregister(FixedB0, B0s(:,:,:,i), 'affine', optimizer, metric);
    rB0(:,:,:,i)=smooth3(rB0(:,:,:,i), 'box', [3 3 3]);
end
B0map=mean(rB0(:,:,:,1:Param.nb0),4);
simdwis = SimChm(B0map, FA, DTmaps, mag, real(vec), grad_dirs, mask, Param.maxq, Param.bigdel, Param.smalldel, theta_q, phi_q, ax/2, gw3, Param.R, bval, MD1000);
for i=1:length(bval)
    A1=simdwis(:,:,:,i);
    A2=A1(:);
    A2=A2(A2>0);
    mA2=sort(A2);
    locl=round(0.995*length(mA2));
    A1(A1>(mA2(locl)))=mA2(locl);
    simdwis(:,:,:,i)=A1;
end
% UNDISTORT - Registration and gradient reorientation

[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = optimizer.InitialRadius/100;
optimizer.MaximumIterations = 200;

h = waitbar(0,'UNDISTORT...');
for i=1:length(grad_dirs)
    A1=simdwis(:,:,:,i); 
    A1 = real(A1);
    KK=data(:,:,:,i);
     tform = imregtform(KK, A1, 'affine', optimizer, metric);
    newgrad(i,:)=[grad_dirs(i,:) sqrt(sum(grad_dirs(i,:).^2))]*tform.T;
    newgrad2(i,:)=grad_dirs(i,:)*tform.T(1:3,1:3);
    rDWI(:,:,:,i)=imwarp(data(:,:,:,i), tform, 'OutputView', imref3d(size(KK)));
    waitbar(i/length(grad_dirs));
end
close(h)
for i=1:length(newgrad)
    grad_dirs(i,:)=newgrad(i,4).*newgrad(i,1:3)./sqrt(sum(newgrad(i,1:3).^2));
end
grad_dirs=newgrad2;
grad_dirs(1:Param.nb0,:)=zeros(Param.nb0,3);

for i=1:length(bval)
    rDWI(:,:,:,i)=smooth3(rDWI(:,:,:,i), 'gaussian',5);
end

% calculate DTI parameter for each b shel
bshel(1)=0;
indb=2;
for i=2:length(bval)
if ismember(bval(i), bshel)==0
bshel(indb)=bval(i);
indb=indb+1;
end
end
sFA=zeros([size(mask) length(bshel)-1]);
sMD=zeros([size(mask) length(bshel)-1]);
sEigval=zeros([size(mask) 3 length(bshel)-1]);
sEigvec=zeros([size(mask) 3 length(bshel)-1]);
sDT=zeros([size(mask) 6 length(bshel)-1]);

for i=2:length(bshel)
    [sFA(:,:,:,i-1), sMD(:,:,:,i-1), sDT(:,:,:,:,i), sEigval(:,:,:,:,i-1), sEigvec(:,:,:,:,i-1)]=dti(bval, grad_dirs, rDWI, bshel(i), mask);
end

save(fullfile(main_path,'AxSI','alldata.mat'),'bval', 'mask', 'grad_dirs', 'phi_q','theta_q', 'R_q', 'rDWI','Param','vec', 'DT1000', 'sMD','bshel', 'sEigval','-v7.3'); %save([fname,'_alldata.mat'],'-v7.3');
% AxCalibering!
cd(main_path);
%CC_Mask_Creater(fname,'');
CalcAx(main_path, file_names);

