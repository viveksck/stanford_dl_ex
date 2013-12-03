function [batch, label_batch ] = SGDTestGetBatch(data, labels, s, e)
data1 = data';
labels1 = labels';
batch = data1(s:e, :);
label_batch = labels1(s:e);
end

