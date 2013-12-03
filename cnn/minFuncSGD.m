function [opttheta] = minFuncSGD(funObj,theta,data,labels,getbatchObj,...
                                    options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;

if ~isfield(options, 'enhanced')
    options.enhanced = false;
end

epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));
UpdateThetaAndVelocity = @UpdateThetaAndVelocityBasic;
UpdateAlpha = @UpdateAlphaBasic;


if options.enhanced
    C = options.C;
    G = options.G;
    add_noise = options.add_noise;
end

%%======================================================================
%% SGD loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        [mb_data, mb_labels] = getbatchObj(data, labels, s, s + minibatch - 1);

        % evaluate the objective function on the next minibatch
        [cost ,grad] = funObj(theta,mb_data,mb_labels);
        
        % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
        [theta, velocity] = UpdateThetaAndVelocity(alpha, grad, mom,...
            velocity, theta, options);
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;

    % aneal learning rate by factor of two after each epoch
    alpha = UpdateAlpha(alpha, e, options);

end;

opttheta = theta;


end

function [theta, velocity] = UpdateThetaAndVelocityBasic(alpha, grad, mom, oldvel, oldtheta, options)
velocity = (alpha * grad) + (mom * oldvel);
theta = oldtheta - velocity;
end

function [alpha] = UpdateAlphaBasic(old_alpha, epoch_no, options)
alpha = old_alpha/2.0;
end
