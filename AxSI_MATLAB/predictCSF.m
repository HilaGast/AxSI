function [prcsf]=predictCSF(im)

zi0=im(:);
zi0=zi0(zi0>0);

s=struct('mu', [0.5 1 2 ]', 'Sigma', [0.3 0.3 0.3]', 'PCcomponents', [0.7 0.2 0.1]');
sigma(1,1,1)=0.3*max(zi0);
sigma(1,1,2)=0.3*max(zi0);
sigma(1,1,3)=0.3*max(zi0);

s.Sigma=sigma;
options = statset('MaxIter', 10000, 'TolFun', 1e-7, 'Display', 'final');
obj=gmdistribution.fit(zi0, 3, 'Options', options, 'Start', s);

%objC=gmdistribution(obj.mu, obj.Sigma, obj.PComponents);
%x=min(zi0):(max(zi0)-min(zi0))/99:max(zi0);
%B=pdf(objC,[x']);

p1=zeros(size(im));
p2=zeros(size(im));
p3=zeros(size(im));

objp1=gmdistribution(obj.mu(1), obj.Sigma(1,1,1), 1);
objp2=gmdistribution(obj.mu(2), obj.Sigma(1,1,2), 1);
objp3=gmdistribution(obj.mu(3), obj.Sigma(1,1,3), 1);

[xlocs, ylocs, zlocs]=ind2sub(size(im), find(im>0));

for i=1:length(xlocs)
    p1(xlocs(i), ylocs(i), zlocs(i))=pdf(objp1, im(xlocs(i), ylocs(i), zlocs(i)));
    p2(xlocs(i), ylocs(i), zlocs(i))=pdf(objp2, im(xlocs(i), ylocs(i), zlocs(i)));
    p3(xlocs(i), ylocs(i), zlocs(i))=pdf(objp3, im(xlocs(i), ylocs(i), zlocs(i)));
end

tp=p1+p2+p3;
%p1=p1./tp;
%p2=p2./tp;
p3=p3./tp;

p4=p3;
% p4(find(p4>0.4))=0.4;
% p4=p4./0.4;
prcsf=p4;

