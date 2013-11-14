function [batch, label_batch ] = SGDTestGetBatch(data, labels, s, e)
batch = data(s:e, :);
label_batch = labels(s:e);
end

