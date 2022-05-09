function [decay, MD0]=predictH(DTmaps1, mask, bval, grad_dirs, sMD, bshel)

MD0=zeros(size(mask));
DTmaps1=1000.*DTmaps1;
decay=zeros([size(mask) length(bval)]);
DTmaps(:,:,:,1,1)=DTmaps1(:,:,:,1);
DTmaps(:,:,:,1,2)=DTmaps1(:,:,:,2);
DTmaps(:,:,:,1,3)=DTmaps1(:,:,:,3);
DTmaps(:,:,:,2,1)=DTmaps1(:,:,:,2);
DTmaps(:,:,:,2,2)=DTmaps1(:,:,:,4);
DTmaps(:,:,:,2,3)=DTmaps1(:,:,:,5);
DTmaps(:,:,:,3,1)=DTmaps1(:,:,:,3);
DTmaps(:,:,:,3,2)=DTmaps1(:,:,:,5);
DTmaps(:,:,:,3,3)=DTmaps1(:,:,:,6);    
bloc=find(bshel>0.99);
xm=bshel(bloc);
midi=sMD(:,:,:,bloc-1);

[X, Y, Z] = size(mask);
decayr = reshape(decay,[X*Y*Z,length(bval)]);
MD0r = reshape(MD0,[X*Y*Z,1]);
midi = reshape(midi,[X*Y*Z,length(bloc)]);
DTmaps = reshape(DTmaps,[X*Y*Z,size(DTmaps,4),size(DTmaps,5)]);
lbval = length(bval);
parfor i=1:X*Y*Z
    if mask(i)
    f = fit(double(xm'),squeeze(midi(i,:))','exp1');
    fac=f.a./midi(i,1);
    MD0i= 1.0*f.a;
    D_mat=squeeze(1.0*fac.*DTmaps(i, : ,:));    
    for j=1:lbval
        decayr(i,j)=exp(-max(bval).*(grad_dirs(j,:)*D_mat*grad_dirs(j,:)'));
    end
    MD0r(i) = MD0i;
    end
end
decay = reshape(decayr,[X,Y,Z,length(bval)]);
MD0 = reshape(MD0r,[X,Y,Z]);












