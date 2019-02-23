%%
%-------------------------------------------------------------------------%
%     HYPERSPECTRAL UNMIXING USING A PERTURBED LINEAR MIXING MODEL        %
%-------------------------------------------------------------------------%
%% File
% File : main.m
% Author : P.A. Thouvenin (05/11/2014)
% Last update : 25/05/2017
%=========================================================================%
% Related article :
% P.-A. Thouvenin, N. Dobigeon and J.-Y. Tourneret, "Hyperspectral unmixing
% with spectral variability using a perturbed linear mixing model", 
% IEEE Trans. Signal Processing, vol. 64, no. 2, pp. 525-538, Jan. 2016.
%=========================================================================%
clc, clear all, close all, format compact;
addpath utils;
addpath src;
addpath data;
%=========================================================================%
%% Remarks
% The codes associated with the following papers have been directly 
% downloaded from their authors' website: 
%
% [1] J. M. Nascimento and J. M. Bioucas-Dias, “Vertex component analysis: 
% a fast algorithm to unmix hyperspectral data,” IEEE Trans. Geosci. Remote
% Sens., vol. 43, no. 4, pp. 898–910, April 2005. [vca.m]
% [2] J. M. Bioucas-Dias and M. A. T. Figueiredo, "Alternating direction 
% algorithms for constrained sparse regression: Application to hyperspectral
% unmixing," in Proc. IEEE GRSS Workshop Hyperspectral Image Signal
% Process.: Evolution in Remote Sens. (WHISPERS), Reykjavik, Iceland,
% June 2010. [sunsal.m]
%=========================================================================%
%%
%--------------------------------------------------------------
% Data section
%--------------------------------------------------------------
dataName = 'Moffett_vca';
fileName = ['bcd_admm_',lower(dataName)];
folderName = 'results/';
mkdir(folderName);

% Moffett
load(dataName,'A_VCA','M0','data','Y_bar','U','V','H','W');
[L,R] = size(M0);
N = H*W;
Y = (reshape(permute(data,[2 1 3]),H*W,L))'; % lexicographical data ordering
A0 = (reshape(permute(A_VCA,[2 1 3]),H*W,R))';


%--------------------------------------------------------------
% Unmixing parameters
%--------------------------------------------------------------
% Stopping criteria
epsilon = 1e-4; % BCD iteration
eps_abs = 1e-2; % primal residual
eps_rel = 1e-4; % dual residual
% ADMM parameters
nIterADMM = 100;       % maximum number of ADMM subiterations
nIterBCD = 100;
rhoA = 100; 
rhoM = 1e-1;
rhodM = 1e-1;
tau_incr = 1.1;
tau_decr = 1.1;
mu = 10.;
% Regularization 
typeM = 'mdist';    % regularization type ('NONE','MUTUAL DISTANCE','VOLUME','DISTANCE')
percent = 0.1;
alpha = 0;          % abundance regularization parameter
beta = 5.4e-4;      % endmember regularization parameter
gamma = 1e-1;       % variability regularization parameter
flag_proj = false;  % select dM update version (constrained or penalized version)
if N > 1e4          % enable parallelization (or not), depending on the number of pixels
    flag_parallel = true;
else
    flag_parallel = false;
end
flag_update = true; % enable augemented Lagrangian update

%--------------------------------------------------------------
% Initialization
%--------------------------------------------------------------
% Endmember initialization (VCA [1])
% [M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(Y,4,'vca');
% [L,R] = size(M0);
% Abundance initialization (SUNSAL [2])
% A0 = sunsal(M0,Y,'POSITIVITY','yes','ADDONE','yes');
% A0 = max(bsxfun(@minus,A0,max(bsxfun(@rdivide,cumsum(sort(A0,1,'descend'),1)-1,(1:size(A0,1))'),[],1)),0);

% Perturbation matrices initialization
dM0 = zeros(L,R,N);
switch lower(typeM)
    case 'none'
        aux = {typeM};
        aux1 = {typeM,0.};
    case 'dist'
       aux = {typeM,M0};
       aux1 = {typeM,0.,M0};
    case 'mdist'
        aux = {typeM};
        aux1 = {typeM,0.};
    case 'volume'
        aux = {typeM,Y_bar,V};
        aux1 = {typeM,0.,Y_bar,U,V};
    otherwise
        typeM = 'none';
        aux = {typeM};
        aux1 = {typeM,0.};
end
[alpha,beta,typeM] = penalty_term_plmm(Y,M0,A0,dM0,H,W,percent,'PENALTY',aux);
alpha = 10*alpha;
aux1{2} = beta;             

%--------------------------------------------------------------
% BCD/ADMM unmixing (based on the PLMM)
%--------------------------------------------------------------
disp(['ADMM processing (M : ', typeM,', alpha = ',num2str(alpha),', beta = ',num2str(beta),', gamma = ',num2str(gamma),')...']);
tic
[f,A,M,dM] = bcd_admm(Y,A0,M0,dM0,W,H,gamma,flag_proj,flag_parallel,flag_update,eps_abs,eps_rel,epsilon,'AL PARAMETERS',{rhoA,rhoM,rhodM},'PENALTY A',alpha,'PENALTY M',aux1,'AL INCREMENT',{tau_incr,tau_decr,mu},'MAX STEPS',{nIterADMM,nIterBCD});
time =  toc;

%--------------------------------------------------------------
% Error computation
%--------------------------------------------------------------
[RE, aSAM, var_map] = real_error(Y,A,M,dM,W,H);
if alpha > 0
    aux_name = ['ss_',lower(typeM)];
else
    aux_name = lower(typeM);
end
if flag_proj
    aux_name = [aux_name,'_p'];
end
save([folderName,fileName,'_',aux_name],'f','A','M','dM','alpha','beta','gamma','typeM','eps_abs','eps_rel',...
'tau_incr','tau_decr','mu','epsilon','nIterADMM','nIterBCD','W','H',...
'RE','aSAM','time','var_map','rhoA','rhoM','rhodM');
disp('---------------------------------------------------------------------------');
