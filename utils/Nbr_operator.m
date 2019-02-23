function [D,DDt,l2Norm_DDt] = Nbr_operator(H,W,order,bool)
% Neighbour difference operator for a 4 or an 8 neighbourhood.
%-------------------------------------------------------------------------%
% Input: 
% > H       image height
% > W       image width
% > order   pixel ordering (0: MATLAB default ordering, 1: lexicographical
%           ordering)
% > bool    neighbourhood selection (0: 4 nbrs., 1: 8 nbrs.)
%
% Output:
% < D       neighbourhood operator (sparse [N|8/4 N])
% < DDt     D*transpose(D).
%-------------------------------------------------------------------------%
%%
% Code : Pierre-Antoine Thouvenin, November 13th 2015.
%%

% D only necessary to compute the objective fonction without for loop
% (compare execution speed with the for loop version)

N = H*W;

switch order
    case 0 % MATLAB default ordering
        % Down
        B = [[-ones(H-1,1);0],[ones(H-1,1);0]];
        d = [-1,0];
        temp = mat2cell(repmat(spdiags(B,d,H,H),1,W),H,H*ones(W,1));
        Down = blkdiag(temp{:}); % block-diagonal matrix with H blocks

        % Up
        B = [[0;-ones(H-1,1)],[0;ones(H-1,1)]];
        d = [0,1];
        temp = mat2cell(repmat(spdiags(B,d,H,H),1,W),H,H*ones(W,1));
        Up = blkdiag(temp{:}); % block-diagonal matrix with H blocks

        % Left
        B = [[zeros(H,1);ones(N-H,1)],[zeros(H,1);-ones(N-H,1)]];
        d = [0,H];
        Left = spdiags(B,d,N,N);

        % Right
        B = [[-ones(N-H,1);zeros(H,1)],[ones(N-H,1);zeros(H,1)]];
        d = [-H,0];
        Right = spdiags(B,d,N,N);

        D = [Left,Right,Up,Down];
        DDt = Left*(Left') + Right*(Right') + Up*(Up') + Down*(Down'); 

        if bool % 8 neighbourhood if flag activated
            % UpLeft
            B = [[zeros(H,1);repmat([0;ones(H-1,1)],W-1,1)],[zeros(H,1);repmat([0;-ones(H-1,1)],W-1,1)]];
            d = [0,H+1];
            Ul = spdiags(B,d,N,N);

            % DownRight
            B = [[repmat([-ones(H-1,1);0],W-1,1);zeros(H,1)],[repmat([ones(H-1,1);0],W-1,1);zeros(H,1)]];
            d = [-H-1,0];
            Dr = spdiags(B,d,N,N);

            % UpRight
            B = [[repmat([0;-ones(H-1,1)],W-1,1);zeros(H,1)],[repmat([0;ones(H-1,1)],W-1,1);zeros(H,1)]];
            d = [-H+1,0];
            Ur = spdiags(B,d,N,N);  

            % DownLeft
            B = [[zeros(H,1);repmat([ones(H-1,1);0],W-1,1)],[zeros(H,1);repmat([-ones(H-1,1);0],W-1,1)]];
            d = [0,H-1];
            Dl = spdiags(B,d,N,N);

            DDt = DDt + ( Ul*(Ul') + Ur*(Ur') + Dl*(Dl') + Dr*(Dr') );   
            D = [D,Ul,Ur,Dl,Dr];
        end
        
    case 1 % Lexicographical order
        % Left
        B = [[0;ones(W-1,1)],[0;-ones(W-1,1)]];
        d = [0,1];
        temp = mat2cell(repmat(spdiags(B,d,W,W),1,H),W,W*ones(H,1));
        Left = blkdiag(temp{:}); % block-diagonal matrix with H blocks

        % Right
        B = [[-ones(W-1,1);0],[ones(W-1,1);0]];
        d = [-1,0];
        temp = mat2cell(repmat(spdiags(B,d,W,W),1,H),W,W*ones(H,1));
        Right = blkdiag(temp{:}); % block-diagonal matrix with H blocks

        % Up
        B = [[zeros(W,1);ones(N-W,1)],[zeros(W,1);-ones(N-W,1)]];
        d = [0,W];
        Up = spdiags(B,d,N,N);

        % Down
        B = [[-ones(N-W,1);zeros(W,1)],[ones(N-W,1);zeros(W,1)]];
        d = [-W,0];
        Down = spdiags(B,d,N,N);

        D = [Left,Right,Up,Down];
        DDt = Left*(Left') + Right*(Right') + Up*(Up') + Down*(Down'); 

        if bool % 8 neighbourhood if flag activated
            % Upleft
            B = [[zeros(W,1);repmat([0;ones(W-1,1)],H-1,1)],[zeros(W,1);repmat([0;-ones(W-1,1)],H-1,1)]];
            d = [0,W+1];
            Ul = spdiags(B,d,N,N);

            % Upright
            B = [[zeros(W,1);repmat([ones(W-1,1);0],H-1,1)],[zeros(W,1);repmat([-ones(W-1,1);0],H-1,1)]];
            d = [0,W-1];
            Ur = spdiags(B,d,N,N);

            % Downleft
            B = [[repmat([0;-ones(W-1,1)],H-1,1);zeros(W,1)],[repmat([0;ones(W-1,1)],H-1,1);zeros(W,1)]];
            d = [-W+1,0];
            Dl = spdiags(B,d,N,N);

            % Downright
            B = [[repmat([-ones(W-1,1);0],H-1,1);zeros(W,1)],[repmat([ones(W-1,1);0],H-1,1);zeros(W,1)]];
            d = [-W-1,0];
            Dr = spdiags(B,d,N,N);   

            DDt = DDt + ( Ul*(Ul') + Ur*(Ur') + Dl*(Dl') + Dr*(Dr') );
            D = [D,Ul,Ur,Dl,Dr];
        end 
end

l2Norm_DDt = sqrt(sum(DDt(:).^2)); % possible de faire encore plus efficace...

end

