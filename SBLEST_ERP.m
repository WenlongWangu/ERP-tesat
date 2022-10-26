function [W,SF,TF,alpha] = SBLEST_ERP(X, Y, Maxiters, e, lambda)
% ************************************************************************
% SBLEST_ERP: Decoding ERP signals using spatio-temporal filtering 
%
% --- Inputs ---
% Y         : Observed label vector.
% X         : M-trial EEG signals from train set.
%             M cells: Each X{1,i} represents a trial with size of [C*T],
%             C:# of channels,  T:# of timepoints. 
% Maxiters  : Maximum number of iterations (500 is default).
% e         : Convergence threshold.
% lambda    : Regularization parameter.
%
% --- Outputs ---
% W         : Estimated low-rank weighted matrix with size of [C*T],
%             where rank(W) = L and W = SF*diag(alpha)*TF'.
% alpha     : Regression weights (singular values of W). [L, 1].
% SF        : Spatial filter matrix (left singular vectors of W). [C, L].
%             Each column is a spatial filter.
% TF        : Temporal filter matrix (right singular vectors of W). [T, L].
%             Each column is a temporal filter.

%% Initialization
[~, M] = size(X); % M: # of samples;
epsilon =e;
Loss_old = 0;
threshold = 0.05;
%% reashape the dataset into two dimension and compute initial Sigma_y
[X_all,C,T]= compute_X_all(X); % reashape the dataset into two dimension
Psi = eye(C); % the covariance matrix of Gaussian prior distribution is initialized to be Unit diagonal matrix
Sigma_y = compute_Sigma_y(M,T,X_all,lambda,Psi,C);%
for i = 1:Maxiters
   %% Update covariance parameters Omega
    Omega = zeros(C,C);
    for t = 1:T
        start = (t-1)*C + 1; stop = start + C - 1;
       Omega = Omega+ X_all(start:stop,:)*Sigma_y^(-1)*X_all(start:stop,:)';
    end
    O_1 = Omega^(1/2);
    O_2 = O_1^(-1);
%% Obtain weighted data and update W via APGM
   X_APGM = zeros(C,T,M);
    for m =1:M
        X_APGM(:,:,m) = O_2*X{m};
    end   
     cfg.isintercept =0; %  intercept is not used in regression model
     [V, ~] =NucLR_APGM(cfg, X_APGM, Y', lambda);% Obtaining V via APGM
     W = O_2*V;% compute W via V;
   %% Compute Psi 
   Psi = real(O_2*(O_1*(W*W')*O_1)^(1/2)*O_2);
   %% update Sigma_Y
   Sigma_y = compute_Sigma_y(M, T, X_all, lambda, Psi, C); 
   %% Output display and  convergence judgement
      Loss = Y'*Sigma_y^(-1)*Y + log(det(Sigma_y));
      delta_loss = norm(Loss - Loss_old,'fro')/norm( Loss_old,'fro');  
      if (delta_loss < epsilon)
          disp('EXIT: Change in Loss below threshold');
          break;
      end
      Loss_old = Loss;
      if (~rem(i,10))
          disp(['Iterations: ', num2str(i),  '  Loss: ', num2str(Loss), '  Delta_Loss: ', num2str(delta_loss)]);
      end   
end
    %% Eigendecomposition of W
     [SF_all, D, TF_all] = svd(W); % each column of V represents a spatio-temporal filter
     alpha_all = diag(D); % classifier parameters
     %% Select L pairs of Spatio filters, Temporal filter and alpha.
    d = abs(diag(D)); d_max = max(d); 
    w_norm = d/d_max; % normalize eigenvalues of W by the maximum eigenvalue
    index = find(w_norm > threshold); % find index of selected V by pre-defined threshold,.e.g., 0.05
    SF = SF_all(:,index);
    TF = TF_all(:,index);
    alpha = alpha_all(index);
    % select Spatio filters, Temporal filter and alpha by index  
end


function [X_all, C, T] = compute_X_all(X)
% reashape the dataset into two dimension
M = length(X);
[C,T] = size(X{1,1});
 X_all = zeros(C*T,M);
for m = 1:M
X_all(:,m)  = vec(X{m});
end
end

function  Sigma_y = compute_Sigma_y(M, T, X_all, lambda, Psi, C)
% compute Sigma_Y
    lam = lambda/2;
     RPR = zeros(M, M); 
     for t = 1:T
         start = (t-1)*C + 1; stop = start + C - 1;
         Temp = Psi*X_all(start:stop,:); 
         RPR =  RPR + X_all(start:stop,:)'*Temp;  
     end
     Sigma_y = RPR + lam*eye(M);
end