function [cost, grad ] = SGDTestLossfunc(theta, data, labels)
batch_size = length(labels);
errs = data * theta - labels;
cost = sum((errs.^2)/(2.0 * batch_size));
for j = 1:length(theta)
    dftheta(j) = (data(:,j)' * (errs));
end
grad  = ((1.0/batch_size) * dftheta)';
end