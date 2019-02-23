function f = data_fit(Y,M,A,dM)
%Data fitting term.
%   [23/05/2017]
[L,N] = size(Y);
dMA = zeros(L,N);
for n = 1:N
    dMA(:,n) = dM(:,:,n)*A(:,n);
end
f = norm(Y-M*A-dMA,'fro')^2/2; 

end

