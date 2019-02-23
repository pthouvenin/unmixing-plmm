function v = volume(M, V, Y_bar)
%Compute the volume of the simplex enclosing the data (LMM interpretation).
%  [23/05/2017]
R = size(M,2);
T = V*bsxfun(@minus, M, Y_bar); 
v = (det([T;ones(1,R)])/factorial(R-1));
end

