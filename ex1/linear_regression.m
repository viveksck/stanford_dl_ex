function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  labels = y';
  data = X';
  batch_size = length(labels);
  errs = data * theta - labels;
  cost = sum((errs.^2)/(2.0 * batch_size));
  for j = 1:length(theta)
    dftheta(j) = (data(:,j)' * (errs));
  end
  grad  = ((1.0/batch_size) * dftheta)'; 
  f = cost;
  g = grad;

