function [alpha,beta,typeM] = penalty_term_plmm(Y,M,A,dM,H,W,percent,varargin)
% Estimates for alpha and beta.
%%
% Code : Pierre-Antoine Thouvenin, February 16th 2015.
%%
%-------------------------------------------------------------------------%
%   Y   =  M    A   + [dM1*a1|...|dMn*an] +   B     (LMM)
% (L|N)  (L|K)(K|N)           (dMA)         (L|N)
%
% 0.5*|Y - MA - dMA|² + 0.5*alpha\phi(A) + 0.5*beta\psi(M) + 0.5*gamma|dM|²
%  
% s.t.            A >=  0         with N : number of pixels;    
%                 M >=  0              K : number of endmembers;
% forall n  M + dMn >=  0              L : number of spectral bands.
%             A'   1 =  1
%           (N|K)(K|1)(N|1)
%-------------------------------------------------------------------------%
% Input: 
% > Y   hypespectral image (L|N);
% > M   initial endmembers (L|K);
% > A   initial abundances (L|N);
% > dM  initial variability terms [cell(1|N)];
% > H   image height;
% > W   image width;
% > percent  weight given to the penalty terms (relatively to the data
%            fitting term).
%
% Optional parameters
% >> DISTANCE
% --> M0   reference signatures
%
% >> VOLUME
% --> Y_proj  projected data(PCA)(K-1|N)
% --> Y_bar   data mean (mean(Y,2))(L|1)
% --> U       inverse projector (PCA variables -> M)(L|K-1);
% --> V       projector (M -> PCA variables)(K-1|L);
%
% Output:
% < f   objective function value.
%-------------------------------------------------------------------------%
%%
%--------------------------------------------------------------
% Initial data fitting term
%--------------------------------------------------------------
[L,R] = size(M);
N = size(A,2);
dMA = zeros(L,N);
for n = 1:N
    dMA(:,n) = dM(:,:,n)*A(:,n);
end
data_fit = norm(Y-M*A-dMA,'fro')^2;

%--------------------------------------------------------------
% Optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'PENALTY'
                typeM = varargin{i+1}{1};
                switch lower(typeM)
                    case 'none'
                         beta = 0;                       
                    case 'dist'
                        M0 = varargin{i+1}{2}; 
                        beta = (R/N)*percent*data_fit/(norm(M-M0,'fro')^2); % L*R/(L*N) 
                    case 'mdist'
                        G = mdist_prior(R);
                        beta = R^2/(L*N)*percent*data_fit/(norm(M*G,'fro')^2); % R^2/(L*N)    
                    case 'volume'
                        Y_bar = varargin{i+1}{2};
                        V = varargin{i+1}{3};
                        K = size(V,1);
                        vol = volume(M, V, Y_bar);
                        beta = (K*R)/(L*N)*percent*data_fit/(vol^2); % R*K/(L*N)
                    otherwise
                        typeM = 'none';
                        beta = 0;
                end
            otherwise
                typeM = 'none';
                beta = 0;
        end
    end
end

%--------------------------------------------------------------
% Abundance penalization term
%--------------------------------------------------------------
D = Nbr_operator(H,W,1,0); % lexicographical order
alpha = (4*R/L)*percent*data_fit/sum(sum((A*D).^2)); % 4*N*R/(L*N)

end

