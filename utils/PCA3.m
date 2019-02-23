function [U,Ut,P,Pt,y_bar,Y_proj,d] = PCA3(Y,K)
% Principal component analysis (adapted from find_endm)
%%
% Code : Pierre-Antoine Thouvenin, March 20th 2015 / May 11th 2016.
%%
%-------------------------------------------------------------------------%
% Input: 
% > Y     hypespectral image (L|N);
% > K     subspace dimension.
%
% Output:
% < U      inverse projector (L|K-1)
% < P      projector (K-1|L)
% < Y_bar  data mean (L|1);
% < Y_proj projected data (PCA space) (R-1|N).
%-------------------------------------------------------------------------%
%%
%--------------------------------------------------------------
% PCA
%--------------------------------------------------------------
y_bar = mean(Y,2);
Rmat = bsxfun(@minus,Y,y_bar);
Rmat = Rmat*(Rmat');   % empirical covariance matrix
[V,d] = eig(Rmat,'vector') ; % first K eigenvectors
[d,id] = sort(d,'descend');
V = V(:,id);

% projector
P = V(:,1:K)';
Pt = V(:,K+1:end)';
% P = D^(-1/2)*(V'); % permet de remédier aux problèmes numériques pouvant apparaître si le simplex est trop "étiré" 
                     % dans une direction (différence d'amplitude importante entre les valeurs propres)
% inverse projector
U = pinv(P);  % pinv(P)*D^(1/2);
Ut = pinv(Pt);

% projecting
Y_proj = P*bsxfun(@minus,Y,y_bar);  % projection of the data on the K principal axes.

end

