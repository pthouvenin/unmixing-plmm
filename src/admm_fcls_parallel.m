function A = admm_fcls_parallel(Y,A,M,dM,rho,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,bool_rho)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% Implementation inspired from Bioucas2010whispers (sunsal)
% [17/05/2017]

[R,N] = size(A);

% bool_rho = 1 : update rho
if bool_rho % update rho
    parfor n = 1:N
        % Selection of the associated perturbation matrix
        M1 = M + dM(:,:,n);
        MtM = M1'*M1;
        
        % Lagrange's multiplier initialisation
        lambda = zeros(R,1);
        
        % Initial splitting parameter / regularization parameter
        u = A(:,n);
        rho_n = rho;
        
        % Auxiliary variables (pre-computation)
        MtY = M1'*Y(:,n);
        
        for q = 1:nIter
            
            u_prev = u;
            
            % A(:,n) update
            [L1,U] = lu(MtM + rho_n*eye(R));
            Binv = inv(U)*inv(L1);
            Binvw = (MtM + rho_n*eye(R))\(MtY + rho_n*(u - lambda));
            A(:,n) = Binvw - Binv*ones(R,1)*(sum(Binvw(:)) - 1)/sum(Binv(:));
            
            % Splitting variable update
            u = bsxfun(@max, A(:,n) + lambda, 0);
            
            % Lagrange's mutliplier update
            lambda = lambda + A(:,n) - u;
            
            % Error computation / count
            norm_primal = norm(A(:,n) - u, 2);
            norm_dual = rho_n*norm(u - u_prev, 2);
            
            eps_primal = sqrt(R)*eps_abs + eps_rel*max(norm(A(:,n),2), norm(u,2));
            eps_dual = sqrt(R)*eps_abs + eps_rel*rho_n*norm(lambda,2); % rho added since scaled form
            
            if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
                break
            end
            
            % Regularisation parameter update
            if (norm_primal > mu*norm_dual)
                rho_n = tau_incr*rho_n;
                % Lagrange's multiplier update if rho_n updated
                lambda = lambda/tau_incr;
            elseif (norm_dual > mu*norm_primal)
                rho_n = rho_n/tau_decr; % correction
                % Lagrange's multiplier update if rho_n updated
                lambda = lambda*tau_decr;
            end
        end
    end
else % no update of rho % problème (divergence...)
    parfor n = 1:N
       
        % Selection of the associated perturbation matrix
        M1 = M + dM(:,:,n);
        
        % Lagrange's multiplier initialisation
        lambda = zeros(R,1);
        
        % Initial splitting parameter
        u = A(:,n);
        
        % Auxiliary variables (pre-computation)
        [L1,U] = lu(M1'*M1 + rho*eye(R));
        Binv = inv(U)*inv(L1);
        MtY = M1'*Y(:,n);
        c = sum(Binv(:));
        
        for q = 1:nIter
            u_prev = u;
            
            % A(:,n) update
            Binvw = Binv*(MtY + rho*(u - lambda));
            A(:,n) = Binvw - ((sum(Binvw) - 1)/c)*Binv*ones(R,1);
            
            % Splitting variable update
            u = bsxfun(@max, A(:,n) + lambda, 0);
            
            % Lagrange's mutliplier update
            lambda = lambda + A(:,n) - u;
            
            % Error computation / count
            norm_primal = norm(A(:,n) - u, 2);
            norm_dual = rho*norm(u - u_prev, 2);
            
            eps_primal = sqrt(R)*eps_abs + eps_rel*max(norm(A(:,n),2), norm(u,2));
            eps_dual = sqrt(R)*eps_abs + eps_rel*rho*norm(lambda,2); % rho added, since scaled form
                      
            if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
                break
            end
        end
    end    
end

