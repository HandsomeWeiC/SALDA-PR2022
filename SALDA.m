function [Y,s, W, o] = SALDA(X_train, Y_train, feature_num, perr)
% input---
% X_train: data [n*d]
% Y_train: label [n*1]
% feature_num: reduced dimensionality
% output---
% s: local weight
% W: linear transformation [d*m]

% addpath('.\util') 

[n_t, n_d] = size(X_train); % data num, data dimension
% WW_w = ones(n_t,n_t); % inner
% WW_w = WW_w - diag(diag(WW_w));
% WW_b = 1/(2*n_t)*ones(n_t,n_t); % inner + outer
% WW_b = WW_b - diag(diag(WW_b));

%% lean s and W
% initialize weight
c = unique(Y_train); % classes
s = sparse(n_t,n_t);
for i = 1:length(c)
    ind = find(Y_train==c(i));
    s(ind,ind) = 1/(length(ind)-1);
%     WW_w(ind,ind) = 1/(length(ind)-1);
end
s = s - diag(diag(s));

interval = 1;
Obj = 1e+4;
o = [];
interval_set = [];
count=0;
disp('start iteration')

H = eye(n_t) - 1/n_t*ones(n_t);
Sb = X_train'*H*X_train;
Sb = (Sb + Sb')/2;
% invSb = inv(Sb);

while( abs(interval)>perr)
    % fix s, update W
    D_w = spdiags(sum(s,2),0,n_t,n_t); % degree matrix
    L_w = D_w - s;
    L_w = (L_w + L_w')/2;
    Sw = X_train'*L_w*X_train;
    Sw = (Sw + Sw')/2;
    
    if count>0 
        interval_set = [interval_set; interval];
        o = [o;obj_mole/obj_deno];
    end
    
%%%%%%%%%%% W'*W=I  %%%%%%%% %%%
%      M = invSb*Sw;
%      [W, Lambda]=eig(M);
%      lambda=diag(Lambda);
%      [lambda, SortOrder]=sort(lambda,'ascend');
%      W=W(:,SortOrder(1:feature_num));

%  [W2, ~] = TraceRatio_fast(Sw_w, Sb_b, feature_num, 0);
%      if trace(W1'*Sw*W1)/trace(W1'*Sb*W1)<trace(W2'*Sw*W2)/trace(W2'*Sb*W2)
%           W = W1;
%      else
%         W = W2;
 %    end
%%%%%%%%%%%%  W'*W=I  %%%%%%%%%%%   
 
 [W]  = opt_TRR(Sb, Sw, feature_num);

%%%%%%%%%%%%  W'*St*W=I  %%%%%%%%%%%

% M = invSb*Sw;
% [W, Lambda]=eig(M);
% lambda=diag(Lambda);
% [lambda, SortOrder]=sort(lambda,'ascend');
% W=W(:,SortOrder(1:feature_num));
% W = W*diag(1./sqrt(diag(W'*Sb*W)));     

    % fix W, update s
    y = X_train*W;
    for class = 1:length(c)
        ind = find(Y_train==c(class));
        yc = y(ind,:);
        vc = slmetric_pw(yc',yc','eucdist');
        vc = vc + diag(diag(1e+14*ones(size(vc,1))));
        s(ind,ind) = 1./(2*(vc+eps));
    end
    s = (s+s')/2;
    obj_mole = trace(W'*Sw*W);
    obj_deno = trace(W'*Sb*W);
    interval = Obj-obj_mole;%检测分子的变化
    Obj = obj_mole;
    count = count+1;
%     disp(['循环次数是：', num2str(count)])
    if  count > 2000
        break
    end
end
Y = X_train*W;

