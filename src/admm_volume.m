function M = admm_volume(Y,A,M,dM,P,U,Up,Um,idp,idm,beta,y_bar,rho,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,bool_rho)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% Parallelization not available in this context.
% [17/05/2017]

% Auxiliary variables
[L,R] = size(M);
N = size(A,2);
K = size(U,2); % K = R-1
idk = true(1,K);
idk2 = true(1,R);
idr = true(1,R);
% B = bsxfun(@min,0,min(dM,[],3)); % [07/09/2016] (bounds)
B = zeros(L,R);
Delta = zeros(L,N);
for n = 1:N
   Delta(:,n) = dM(:,:,n)*A(:,n);
   B = bsxfun(@min, B, dM(:,:,n));
end
AAt = A*(A.');
C = (U.')*(Y - Delta - y_bar*sum(A,1))*(A.');
c = beta/factorial(K)^2;
% Projection onto the PCA space (M -> Tm)
Tm = P*bsxfun(@minus, M, y_bar);
% Bounds
tm = zeros(1,R);
tp = zeros(1,R);
% Volume regularization
d = zeros(R,1);
H = [Tm;ones(1,R)];

% ADMM
if bool_rho
    for k = 1:K
        % Interval bounds
        idk(k) = false;
        for r = 1:R
            %  !! The bounds are affected by the update of t_kr !!
            tm(r) = -Inf;
            tp(r) = Inf;
            v = - bsxfun(@rdivide, y_bar + U(:,idk)*Tm(idk,r) + B(:,r), U(:,k)); % [correction : 07/09/2016]
            % t-
            if idp(k)
                tm(r) = max(v(Up(:,k)));
            end
            % t+
            if idm(k)
                tp(r) = min(v(Um(:,k)));
            end
        end
        idk(k) = true;
        
%         keyboard
        idk2(k) = false;
        % Volume (development)
        for r = 1:R
            idr(r) = false;
            d(r) = ((-1)^(r+k))*det(H(idk2,idr)); % computation of the minor
            idr(r) = true; % return to the initial matrix
        end
        idk2(k) = true;
        
        % Initialize ADMM
        rho_k = rho;
        u = Tm(k,:);
        lambda = zeros(1,R);
        
        for q = 1:nIter
            
            u_prev = u;
            
            % Update T(k,:)
            Tm(k,:) = C(k,:)/(AAt + c*d*(d.') + rho_k*eye(R));
            
            % Update u
			u = Tm(k,:) + lambda;
            u = bsxfun(@max, u, tm); % correction [29/05/2017] -> à revoir, tester
			u = bsxfun(@min, u, tp);
            
            % Update lambda
            lambda = lambda + Tm(k,:) - u;
            
            % Error computation / stopping criterion
            norm_primal = norm(Tm(k,:) - u, 2);
            norm_dual = rho_k*norm(u - u_prev, 2);
            eps_primal = sqrt(R)*eps_abs + eps_rel*max(norm(Tm(k,:),'fro'), norm(u,'fro'));
            eps_dual = sqrt(R)*eps_abs + eps_rel*rho_k*norm(lambda,'fro'); % rho added, since scaled form
            
            if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
                break
            end
            
            % Regularisation parameter update
            if (norm_primal > mu*norm_dual)
                rho_k = tau_incr*rho_k;
                % Lagrange's multiplier update if rho updated
                lambda = lambda/tau_incr;
            elseif (norm_dual > mu*norm_primal)
                rho_k = rho_k/tau_decr; % correction
                % Lagrange's multiplier update if rho updated
                lambda = lambda*tau_decr;
            end
        end
        
        % Update H for the next iteration
        H(k,:) = Tm(k,:);
    end
else
    for k = 1:K
        % Interval bounds
        idk(k) = false;
        for r = 1:R
            %  !! The bounds are affected by the update of t_kr !!
            tm(r) = -Inf;
            tp(r) = Inf;
            v = - bsxfun(@rdivide, y_bar + U(:,idk)*Tm(idk,r) + B(:,r), U(:,k)); % [correction : 07/09/2016]
            % t-
            if idp(k)
                tm(r) = max(v(Up(:,k)));
            end
            % t+
            if idm(k)
                tp(r) = min(v(Um(:,k)));
            end
        end
        idk(k) = true;
        
        % Volume (development)
        idk2(k) = false;
        for r = 1:R
            idr(r) = false;
            d(r) = ((-1)^(r+k))*det(H(idk2,idr)); % computation of the minor
            idr(r) = true; % return to the initial matrix
        end
        idk2(k) = true;
        
        
        % Initialize ADMM
        u = Tm(k,:);
        lambda = zeros(1,R);
        
        for q = 1:nIter
            
            u_prev = u;
            
            % Update T(k,:)
            Tm(k,:) = C(k,:)/(AAt + c*d*(d.') + rho*eye(R));
            
            % Update u
            u = bsxfun(@max, Tm(k,:) + lambda, 0);
            
            % Update lambda
            lambda = lambda + Tm(k,:) - u;
            
            %-Error computation / stopping criterion
            norm_primal = norm(Tm(k,:) - u, 2);
            norm_dual = rho*norm(u - u_prev, 2);
            eps_primal = sqrt(R)*eps_abs + eps_rel*max(norm(Tm(k,:),'fro'), norm(u,'fro'));
            eps_dual = sqrt(R)*eps_abs + eps_rel*rho*norm(lambda,'fro'); % rho added, since scaled form
            
            if q > 1 && ((norm_primal < eps_primal && norm_dual < eps_dual))
                break
            end
        end
        
        % Update H for the next iteration
        H(k,:) = Tm(k,:);
    end
end

% Back-projection onto the original space (Tm -> M)
M = bsxfun(@plus,U*Tm,y_bar);

end
