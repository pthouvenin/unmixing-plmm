function M = admm_d(Y,A,M,dM,rho,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,bool_rho,M0,beta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% Implementation inspired from Bioucas2010whispers (sunsal)
% Parallelization not necessary in this context
% varagin contains only a lambda function to compute the additional
% regularization (differentiable)
% [17/05/2017]

[L,R] = size(M);
N = size(A,2);

% Auxiliary variables
Delta = zeros(L,N);
bound = zeros(L,R);
for n = 1:N
    Delta(:,n) = dM(:,:,n)*A(:,n);
    bound = bsxfun(@max, bound, -dM(:,:,n)); % constraint M + dMn >=0 && M >=0
end

% bool_rho = 1 : update rho
if bool_rho % update rho
    AAt = A*(A');
    YDAt = (Y - Delta)*(A') + beta*M0;
    
    % Lagrange's multiplier initialisation
    Lambda = zeros(L,R);
    
    % Initial splitting parameter
    V = M;
    
    for q = 1:nIter
        V_prev = V;
        
        % M update
        M = (YDAt + rho*(V - Lambda))/(AAt + (rho + beta)*eye(R));
        
        % Splitting variable update
        V = bsxfun(@max, M + Lambda, 0);
        
        % Lagrange's mutliplier update
        Lambda = Lambda + M - V;
        
        % Error computation / stopping criterion
        norm_primal = norm(M - V,'fro');
        norm_dual = rho*norm(V - V_prev,'fro');
        eps_primal = sqrt(L*R)*eps_abs + eps_rel*max(norm(M,'fro'), norm(V,'fro'));
        eps_dual = sqrt(L*R)*eps_abs + eps_rel*rho*norm(Lambda,'fro'); % rho added, since scaled form
        
        if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
            break
        end
        
        % Regularisation parameter update
        if (norm_primal > mu*norm_dual)
            rho = tau_incr*rho;
            % Lagrange's multiplier update if rho updated
            Lambda = Lambda/tau_incr;
        elseif (norm_dual > mu*norm_primal)
            rho = rho/tau_decr; % correction
            % Lagrange's multiplier update if rho updated
            Lambda = Lambda*tau_decr;
        end
    end
else % no update of rho       
    % Lagrange's multiplier initialisation
    YDAt = (Y - Delta)*(A') + beta*M0;
    Lambda = zeros(L,R);
    
    % Initial splitting parameter
    V = M;
    
    % Auxiliary variables (pre-computation)
    [L1,U1] = lu(A*(A') + (rho + beta)*eye(R));
    Binv = inv(U1)*inv(L1);
    
    for q = 1:nIter
        V_prev = V;
        
        % M update
        M = (YDAt + rho*(V - Lambda))*Binv;
        
        % Splitting variable update
        V = bsxfun(@max, M + Lambda, 0);
        
        % Lagrange's mutliplier update
        Lambda = Lambda + M - V;
        
        % Error computation / stopping criterion
        norm_primal = norm(M - V,'fro');
        norm_dual = rho*norm(V - V_prev,'fro');     
        eps_primal = sqrt(L*R)*eps_abs + eps_rel*max(norm(M,'fro'), norm(V,'fro'));
        eps_dual = sqrt(L*R)*eps_abs + eps_rel*rho*norm(Lambda,'fro'); % rho added, since scaled form
        
        if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
            break
        end
    end
end

