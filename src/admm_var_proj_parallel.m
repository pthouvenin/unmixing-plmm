function dM = admm_var_proj_parallel(Y,A,M,dM,rho,gamma,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,bool_rho)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% [17/05/2017]

[L,R] = size(M);
N = size(A,2);

% bool_rho = 1 : update rho
if bool_rho % update rho
    parfor n = 1:N
        % Selection of the associated perturbation matrix
        aat = A(:,n)*(A(:,n).');
        
        % Lagrange's multiplier initialisation
        Lambda = zeros(L,R);
        
        % Initial splitting parameter / regularization parameter
        V = dM(:,:,n);
        rho_n = rho;
        
        % Auxiliary variables (pre-computation)
        C = (Y(:,n) - M*A(:,n))*(A(:,n).');
        
        for q = 1:nIter
            
            V_prev = V;
            
            % dMn update
            dM(:,:,n) = (C + rho_n*(V - Lambda))/(aat + rho_n*eye(R));
            
            % Splitting variable update
            V = bsxfun(@max, dM(:,:,n) + Lambda, -M); % M + dMn >= 0
            V = V*min([1,gamma/norm(V,'fro')]); % energy constraint
            
            % Lagrange's mutliplier update
            Lambda = Lambda + dM(:,:,n) - V;
            
            % Error computation / count
            norm_primal = norm(dM(:,:,n) - V, 'fro');
            norm_dual = rho_n*norm(V - V_prev, 'fro');
            
            eps_primal = sqrt(L*R)*eps_abs + eps_rel*max(norm(dM(:,:,n),'fro'), norm(V,'fro'));
            eps_dual = sqrt(L*R)*eps_abs + eps_rel*rho_n*norm(Lambda,'fro'); % rho added since scaled form
            
            if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
                break
            end
            
            % Regularisation parameter update
            if (norm_primal > mu*norm_dual)
                rho_n = tau_incr*rho_n;
                % Lagrange's multiplier update if rho updated
                Lambda = Lambda/tau_incr;
            elseif (norm_dual > mu*norm_primal)
                rho_n = rho_n/tau_decr;
                % Lagrange's multiplier update if rho updated
                Lambda = Lambda*tau_decr;
            end
        end
    end
else % no update of rho % problème (divergence...)
    parfor n = 1:N             
        % Lagrange's multiplier initialisation
        Lambda = zeros(L,R);
        
        % Initial splitting parameter / regularization parameter
        V = dM(:,:,n);
        
        % Auxiliary variables (pre-computation)
        [L1,U1] = lu(A(:,n)*(A(:,n).') + (rho + gamma)*eye(R));
        Binv = inv(U1)*inv(L1);
        C = (Y(:,n) - M*A(:,n))*(A(:,n).');
                
        for q = 1:nIter
            
            V_prev = V;
            
            % dMn update
            dM(:,:,n) = (C + rho*(V - Lambda))*Binv;
%             dM(:,:,n) = (C + rho*(V - Lambda))/(A(:,n)*(A(:,n).') + (rho + gamma)*eye(R));
            
            % Splitting variable update
            V = bsxfun(@max, dM(:,:,n) + Lambda, -M); % M + dMn >= 0
            V = V*min([1,gamma/norm(V,'fro')]); % energy constraint
            
            % Lagrange's mutliplier update
            Lambda = Lambda + dM(:,:,n) - V;
            
            % Error computation / count
            norm_primal = norm(dM(:,:,n) - V, 'fro');
            norm_dual = rho*norm(V - V_prev, 'fro');
            
            eps_primal = sqrt(L*R)*eps_abs + eps_rel*max(norm(dM(:,:,n),'fro'), norm(V,'fro'));
            eps_dual = sqrt(L*R)*eps_abs + eps_rel*rho*norm(Lambda,'fro'); % rho added since scaled form
            
            if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
                break
            end
        end
    end
end

