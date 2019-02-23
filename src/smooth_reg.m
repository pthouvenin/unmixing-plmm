function x = smooth_reg(A,H,W)
%Compute the abudance spatial smoothness regularization.
%  [23/05/2017]
% Spatial smoothness constraint (/!\) lexicographic ordering
[R,N] = size(A);
UpA = zeros(R,N);
for k = 1:N-W
    UpA(:,W+k) = A(:,W+k) - A(:,k);
end
DownA  = [-UpA(:,W+1:end),zeros(R,W)];
LeftA  = zeros(R,N);
RightA = zeros(R,N);
for k = 0:H-1
    LeftA(:,1+k*W:(k+1)*W)  = [zeros(R,1), diff(A(:,1+k*W:(k+1)*W),1,2)];
    RightA(:,1+k*W:(k+1)*W) = [-LeftA(:,2+k*W:(k+1)*W), zeros(R,1)];
end
AH = [LeftA,RightA,UpA,DownA];
x = sum(AH(:).^2);

end

