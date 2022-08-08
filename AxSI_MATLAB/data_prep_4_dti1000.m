function [rdata1000, bval1000, bvec1000] = data_prep_4_dti1000(data, bval, bvec, preprocessed)
    
    b1000locs=find(bval==1);
    data1000=data(:,:,:,[1,b1000locs']);
    rdata1000=zeros(size(data1000));
    rdata1000(:,:,:,1)=data1000(:,:,:,1);
    bvec1000=bvec([1 b1000locs'],:);
    bval1000=bval([1 b1000locs']);
    


    if preprocessed
        rdata1000 = data1000;
        for i=1:(length(b1000locs)+1)
            rdata1000(:,:,:,i)=smooth3(rdata1000(:,:,:,i), 'gaussian',5);
        end
    else
        [optimizer,metric] = imregconfig('multimodal');
        optimizer.InitialRadius = optimizer.InitialRadius/300;
        optimizer.MaximumIterations = 200;
        h = waitbar(0,'Motion Correction for DTI...');
        for i=1:(length(b1000locs)+1)
            waitbar(i/(length(b1000locs)+1));
            [rdata1000(:,:,:,i)] = imregister(data1000(:,:,:,i), rdata1000(:,:,:,1), 'affine', optimizer, metric);
            rdata1000(:,:,:,i)=smooth3(rdata1000(:,:,:,i), 'gaussian',5);
        end
        close(h);
    end
end
 