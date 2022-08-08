function [prcsf]=predict_csf(md_b1000)

    md_b0_nonzero = md_b1000(:);
    md_b0_nonzero = md_b0_nonzero(md_b0_nonzero > 0);

    s=struct('mu', [0.5 1 2 ]', 'Sigma', [0.3 0.3 0.3]', 'PCcomponents', [0.7 0.2 0.1]');
    sigma(1,1,1)=0.3*max(md_b0_nonzero);
    sigma(1,1,2)=0.3*max(md_b0_nonzero);
    sigma(1,1,3)=0.3*max(md_b0_nonzero);
    s.Sigma=sigma;
    options = statset('MaxIter', 10000, 'TolFun', 1e-7, 'Display', 'final');
    gm=gmdistribution.fit(md_b0_nonzero, 3, 'Options', options, 'Start', s);

    p1=zeros(size(md_b1000));
    p2=zeros(size(md_b1000));
    p3=zeros(size(md_b1000));

    gm1=gmdistribution(gm.mu(1), gm.Sigma(1,1,1), 1);
    gm2=gmdistribution(gm.mu(2), gm.Sigma(1,1,2), 1);
    gm3=gmdistribution(gm.mu(3), gm.Sigma(1,1,3), 1);

    [xlocs, ylocs, zlocs] = ind2sub(size(md_b1000), find(md_b1000 > 0));

    for i=1:length(xlocs)
        p1(xlocs(i), ylocs(i), zlocs(i))=pdf(gm1, md_b1000(xlocs(i), ylocs(i), zlocs(i)));
        p2(xlocs(i), ylocs(i), zlocs(i))=pdf(gm2, md_b1000(xlocs(i), ylocs(i), zlocs(i)));
        p3(xlocs(i), ylocs(i), zlocs(i))=pdf(gm3, md_b1000(xlocs(i), ylocs(i), zlocs(i)));
    end

    tp = p1 + p2 + p3;
    prcsf = p3 ./ tp;
    prcsf(isnan(prcsf)) = 0;
end
