function [W, b]=NucLR_APGM(cfg, X, Y, lambda, bQuiet)
% NucLR_APGM   : Code for nuclear-norm regularized linear regression using the accelerated 
%              proximal gradient method (APGM)
%
% Optimization problem: min_{W, b} f(W, b) + g(W), 
%                       where f(W, b) = sum_i(trace(W'*X_i) + b - Y_i)^2, g(W) = lambda*nuc_norm(W)
%
% Syntax:
%  [W, b]=NucLR_APGM(cfg, X, Y, lam, bQuiet)
%
% Inputs:
%  cfg       : Configuration variable. Must include cfg.isintercept (1: include bias terms, 0: no bias)
%  X         : EEG covariance matrices.
%              [C,T,M] array: Each X(:, :, i) assumed to be symmetric.
%  Y         : Response variable, such as clinical scores. [1, n] matrix.
%  lam    : Regularization parameter.
%  bQuiet    : Binary. true suppresses outputs. (default)
%
% Outputs:
%  W         : Weight matrix with size of [C,T].
%  b         : Intercept term.


%% INITIALIZATION
if ~exist('bQuiet','var') || isempty(bQuiet)
    bQuiet = true;
end

[C,T,~] = size(X); 

W0 = zeros(C,T);
W = zeros(C,T);

alpha0 = 0.1;
b = 0;
b0 = b;
f = squaresumerror(W, b, X, Y);
g = 0;
o0 = f + g;
t = 1;
alpha = 0.1;
%% OPTIMIZATION VIA ACCELERATED PROXIMAL GRADIENT METHOD
if cfg.isintercept == 1
    % intercept is used in regression model
    if ~bQuiet
       fprintf('Starting APGM...\n');
    end 
    iter = 1;
    while 1
        if ~bQuiet
           fprintf('Iteration %d\n', iter);
        end
        iter = iter + 1;
        
        P = W + (alpha0 - 1)/alpha*(W - W0);
        W0 = W;
        b = b + (alpha0 - 1)/alpha*(b - b0);
        b0 = b;
        G_b = gradientfb(P, b0, X, Y);      
        G_P = gradientfW(P, b0, X, Y);       
        while 1
            Q = P - t*G_P;
            b = b - t*G_b;
            [U, S, V] = svd(Q, 'econ');
            s = diag(S);
            sigma = pos(s - lambda*t);
            W = U*diag(sigma)*V';
            f = squaresumerror(W, b, X, Y);
            g = lambda*sum(sigma);
            o1 = f + g;
            if f <= gammafun(t, W, P, b, b0, X, Y)
                break;
            end
            t = 0.5*t;
        end
        
        delta = norm(o0 - o1)/o0;
        if ~bQuiet
           fprintf('Cost function decrement %6.5f\n', delta);
        end
        if ((iter > 2) && (delta <= 1e-5)) || (iter > 5000)
            break;
        end   
        o0 = o1;
        alpha0 = alpha;
        alpha = (1+sqrt(1+4*alpha0^2))/2;
    end
else
    % intercept is not used in regression model
    b = 0;
    if ~bQuiet
       fprintf('Starting APGM...\n');
    end
    iter = 1;
    while 1
        if ~bQuiet
           fprintf('Iteration %d\n', iter);
        end
        iter = iter + 1;
        P = W + (alpha0 - 1)/alpha*(W - W0);
        %         P = W + (iter/(iter + 3))*(W - W0);
        W0 = W;
        G_P = gradientfW(P, b, X, Y);
        while 1
            Q = P - t*G_P;
            [U, S, V] = svd(Q, 'econ');
            s = diag(S);
            sigma = pos(s - lambda*t);
            W = U*diag(sigma)*V';
            f = squaresumerror(W, b, X, Y);
            g = lambda*sum(sigma);
            o1 = f + g;
            if f <= gammafun(t, W, P, b, b, X, Y)
                break;
            end
            t = 0.5*t;
        end
        
        delta = norm(o0 - o1)/o0;
        if ~bQuiet
           fprintf('Cost function decrement %6.5f\n', delta);
        end
        if ((iter > 2) && (delta <= 1e-7)) || (iter > 5000)
            break;
        end       
        o0 = o1;
        alpha0 = alpha;
        alpha = (1+sqrt(1+4*alpha0^2))/2;
    end
end
end

function f = squaresumerror(W, b, X, Y)
[C,T,M] = size(X); % C: # of channels£¬ T: # of timepoints, M : # of samples
w = vec(W);
x = reshape(X, [C*T, M]);
f = norm(w'*x +b - Y)^2;
end

function Gf = gradientfW(W, b, X, Y)
[C,T,M] = size(X); % % C: # of channels£¬ T: # of timepoints, M : # of samples
w = vec(W);
x = reshape(X, [C*T, M]);
Gf = reshape(2*(w'*x +b - Y)*x', C, T);
end

function Gf = gradientfb(W, b, X, Y)
[C,T,M] = size(X); % C: # of channels£¬ T: # of timepoints, M : # of samples
w = vec(W);
x = reshape(X, [C*T, M]);
Gf = sum(2*(w'*x + b - Y));
end

function ga = gammafun(mu, W, W0, b, b0, X, Y)
f = squaresumerror(W0, b0, X, Y);
GW = gradientfW(W0, b0, X, Y);
Gb = gradientfb(W0, b0, X, Y);
ga = f + trace((W - W0)'*GW) + (b - b0)*Gb + ...
    1/(2*mu)*norm(W - W0, 'fro')^2 + 1/(2*mu)*(b - b0)^2;
end

