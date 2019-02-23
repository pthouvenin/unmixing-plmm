function [RE, aSAM, var_map] = real_error(Y,A,M,dM,W,H)
% Error computation for rela data.
%%
% Code: Pierre-Antoine Thouvenin, May 23rd 2015.
%%
%-------------------------------------------------------------------------%
% Inputs:
% >   Y     lexicographically ordered hyperspectral data (L|N);
% >   A     abundance matrix (K|N);
% >   M     endmember matrix (L|K);
% >  dM     variability matrices (L|NK)(cell 1|N);
% >   W     width of the image; (N = W*H)
% >   H     heigth of th image.
%
% Outputs:
% < RE       reconstruction mean square error; || Y - MA - Delta || / sqrt(LN)
% <  A       re-shaped abundance matrix (K|N);
% < var_map  variability energy map associated to each endmember (H|W|K).
%-------------------------------------------------------------------------%
N = W*H;
[L,R] = size(M);

%--------------------------------------------------------------
% Abundance data (lexicographical order)
%--------------------------------------------------------------
% if (size(A,3) > 1)
%     A = (reshape(permute(A,[2 1 3]),H*W,R))';  %(K|N) : abundance map
% end

Yhat = zeros(L,N);
for n = 1:N
    Yhat(:,n) = (M + dM(:,:,n))*A(:,n);
end

% Errors
RE = sum((Y(:)-Yhat(:)).^2)/(L*N);
SA = 180*(acos(sum(Y.*Yhat,1)./sqrt(sum(Y.^2,1).*sum(Yhat.^2,1))))/pi;
aSAM = mean(abs(SA));
var_map = squeeze(sqrt(sum(dM.^2,1)/L)); % [R|N]

% % Abundance reshape
% if (size(A,3) <= 1)
%     A = permute(reshape(A',W,H,R),[2 1 3]); %(H|W|K) : abundance cube    
% end

end