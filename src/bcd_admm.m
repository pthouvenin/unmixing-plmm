function [f,A,M,dM] = bcd_admm(Y,A,M,dM,W,H,gamma,flag_proj,flag_parallel,flag_update,eps_abs,eps_rel,epsilon,varargin)
%Resolution by BCD/ADMM of the hyperspectral linear unmixing problem
% accounting for spectral and spatial variabilities.
%%
% Code : Pierre-Antoine Thouvenin, May 25th 2017.
%
%% Model
%-------------------------------------------------------------------------%
%   Y   =  M    A   + [dM1*a1|...|dMn*an] +   B     (PLMM)
% (L|N)  (L|R)(R|N)           (dMA)         (L|N)
%
% 0.5|Y - MA - dMA|�+0.5*alpha*\Phi(A)+0.5*beta*\Psi(M)+0.5*gamma*|dM|�
% 
% s.t.            A >= 0         with N : number of pixels;    
%                 M >= 0              R : number of endmembers;
% forall n  M + dMn >= 0              L : spectral bands number.
%             A'   1 =  1
%           (N|R)(R|1)(N|1)
%-------------------------------------------------------------------------%
%%
% Inputs:
% > Y      pixels (hyperspectral data cube)(L|N)(lexicographically ordered);
% > A      initial abundance matrix (R|N);
% > M      initial endmembers matrix (L|R);
% > dM     initial perturbation matrices (L|R|N);
% > W      width of the image; (N = W*H);
% > H      heigth of the image;
% > gamma  constraint / regularization parameter on the variability;
% > flag_proj         select constrained or regularized formulation for the
%                     variability terms;
% > flag_parallel     enable parallelization;
% > flag_update       enable the update of the augmented Lagrangians parameters
% > eps_abs,eps_rel   subproblems stopping criterion;
% > epsilon           global stopping criterion;
%
% Optional parameters
% >>'AL PARAMETERS' ({rhoA,rhoM,rhodM})
% > muA, muM, mu_dM         augmented Lagrangian (AL) parameters;
% 
% >> AL INCREMENT' ({tau_incr,tau_decr,mu})
% > mu, tau_incr, tau_decr  value to update the AL parameters during ADMM;
%
% >> 'PENALTY A'
% > alpha                   abundance penalty parameter (spatial smoothness)
%
% >> 'MAX STEPS' ({nIterADMM,nIterBCD})
% > nIterADMM                   ADMM maximum iteration number.
% > nIterBCD                BCD maximum iteration number
%
% >>> 'PENALTY M  ({typeM, beta, aux})
% > typeM    string describing the endmember penalization ('NONE','DISTANCE','MUTUAL DISTANCE','VOLUME')
% > beta     endm. penalty parameter 
%
% >> DISTANCE (aux = {M0})
% --> M0     reference signatures
%
% >> VOLUME (aux = {Y_bar,U,V})
% --> Y_bar  data mean (mean(Y,2))(L|1)
% --> U      (PCA variables -> M)(L|R-1);
% --> V      (M -> PCA variables)(R-1|L);
%
% Outputs:
% < f   objective function value at each iteration;
% < A   abundance matrix (R|N);
% < M   endmember matrix (L|R);
% < dM   perturbation matrices (L|R|N)
%%
% Complete example (with typeM = 'VOLUME'):
% [f,A,M,dM] = bcd_admm(Y,A,M,dM,W,H,gamma,flag_proj,flag_parallel,flag_update,...
%              eps_abs,eps_rel,epsilon,'HYPERPARAMETERS',{muA,muM,mudM},...
%              'PENALTY A',alpha,'PENALTY M',{type,beta,Y_bar,U,V},...
%              'AL INCREMENT',{tau_incr,tau_decr,mu},'MAX STEPS',nIter);
%-------------------------------------------------------------------------%
%%
% Number of inputs must be >=minargs and <=maxargs.
narginchk(13, 23);

%--------------------------------------------------------------
% Default parameters (ADMM, regularizations)
%--------------------------------------------------------------
% Regularization
typeM = 'none';
beta = 0;
input = {typeM,beta};
alpha = 0;
% ADMM
tau_incr = 1.1;
tau_decr = 1.1;
rhoA = 1e-3;
rhoM = 1e-3;
rhodM = 1e-3;
nIter = 100;
% BCD
nIterBCD = 100;

%--------------------------------------------------------------
% Optional parameters / selection of the appropriate updates 
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'AL PARAMETERS'
                rhoA = varargin{i+1}{1};
                rhoM = varargin{i+1}{2};
                rhodM = varargin{i+1}{3};
            case 'AL INCREMENT'
                tau_incr = varargin{i+1}{1};
                tau_decr = varargin{i+1}{2};
                mu = varargin{i+1}{3};
            case 'MAX STEPS'
                nIter = varargin{i+1}{1};
                nIterBCD = varargin{i+1}{2};
            case 'PENALTY M'
                typeM = lower(varargin{i+1}{1});
                beta = varargin{i+1}{2};
                switch typeM
                    case 'dist'
                        M0 = varargin{i+1}{3};
                        input = {typeM,beta,M0};
                    case 'volume'
                        Y_bar = varargin{i+1}{3};
                        U = varargin{i+1}{4};
                        V = varargin{i+1}{5};
                        Up = (U > 0);
                        Um = (U < 0);
                        idp = any(Up,1);
                        idm = any(Um,1);
                        input = {typeM,beta,[],Y_bar,U,V};
                    case 'mdist'
                        G = mdist_prior(size(M,2));
                        GGt = G*(G.');
                        input = {typeM,beta};
                    otherwise
                        typeM = 'none';
                        beta = 0;
                        input = {typeM,beta};
                end
            case 'PENALTY A'
                alpha = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%--------------------------------------------------------------
% Algorithm
%--------------------------------------------------------------
f = zeros(nIterBCD+1,1);
f(1) = objective_p(Y,M,A,dM,H,W,alpha,gamma,'PENALTY',input);

for k = 2:nIterBCD + 1
    disp(['Iteration ', num2str(k-1)]);
    
    % Update A
    if alpha > 0
        A = admm_asmooth(Y,A,M,dM,alpha,rhoA,eps_abs,eps_rel,mu, ... % [: 29/05/2017]
            tau_incr,tau_decr,nIter,flag_update,H,W);
    else
        if flag_parallel
            A = admm_fcls_parallel(Y,A,M,dM,rhoA,eps_abs,eps_rel,mu,tau_incr,...
                tau_decr,nIter,flag_update);
        else
            A = admm_fcls(Y,A,M,dM,rhoA,eps_abs,eps_rel,mu,tau_incr,...
                tau_decr,nIter,flag_update);
        end
    end
    
    % Update M
    switch typeM
        case 'none'
            M = admm_p(Y,A,M,dM,rhoM,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update);
        case 'dist'
            M = admm_d(Y,A,M,dM,rhoM,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update,M0,beta);
        case 'mdist'
            M = admm_dendm(Y,A,M,dM,rhoM,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update,GGt,beta);
        case 'volume'
            M = admm_volume(Y,A,M,dM,V,U,Up,Um,idp,idm,beta,Y_bar,rhoM,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update);
    end
    
    % Update dM
    if flag_proj
        if flag_parallel
            dM = admm_var_proj_parallel(Y,A,M,dM,rhodM,gamma,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update);
        else
            dM = admm_var_proj(Y,A,M,dM,rhodM,gamma,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update);
        end
    else
        if flag_parallel
            dM = admm_var_parallel(Y,A,M,dM,rhodM,gamma,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update);
        else
            dM = admm_var(Y,A,M,dM,rhodM,gamma,eps_abs,eps_rel,mu,tau_incr,tau_decr,nIter,flag_update);
        end
    end
    
    % Objective function value
    f(k) = objective_p(Y,M,A,dM,H,W,alpha,gamma,'PENALTY',input);
    
    % Stopping criterion    
    err = abs(f(k-1)-f(k))/f(k-1);
    
    if k > 2 && (err < epsilon)
        break
    end
end

end
