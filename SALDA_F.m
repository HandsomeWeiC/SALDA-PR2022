function [Y, W, obj_wei,s] = SALDA_F(X_train, Y_train, feature_num, perr)
% input---
% X_train: data [n*d]
% Y_train: label [n*1]
% feature_num: reduced dimensionality
% output---
% s: local weight
% W: linear transformation [d*m]

[n_t, n_d] = size(X_train); % data num, data dimension
WW_w = ones(n_t,n_t); % inner
WW_w = WW_w - diag(diag(WW_w));
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
obj_wei = [];
interval_set = [];
count=0;
disp('start iteration')

H = eye(n_t) - 1/n_t*ones(n_t);
Sb = X_train'*H*X_train;
Sb = (Sb + Sb')/2;
invSb = inv(Sb);

while(abs(interval)>perr)
    % fix s, update W
    D_w = spdiags(sum(WW_w.*s,2),0,n_t,n_t); % degree matrix
    L_w = D_w - WW_w.*s;
    L_w = (L_w + L_w')/2;
    Sw = X_train'*L_w*X_train;
    Sw = (Sw + Sw')/2;
   
    if count>0
        interval_set = [interval_set; interval];
        o = [o;obj_mole/obj_deno];
    end
    
%%%%%%%%%%% W'*W=I  %%%%%%%% %%%
%      [W2, Lambda]=eig(inv(Sb)*Sw);
%      lambda=diag(Lambda);
%      [lambda, SortOrder]=sort(lambda,'ascend');
%      W2=W2(:,SortOrder(1:feature_num));
%      W=W2;    

%  [W2, ~] = TraceRatio_fast(Sw_w, Sb_b, feature_num, 0);
%      if trace(W1'*Sw*W1)/trace(W1'*Sb*W1)<trace(W2'*Sw*W2)/trace(W2'*Sb*W2)
%           W = W1;
%      else
 %        W = W2;
 %    end
%%%%%%%%%%%%  W'*W=I  %%%%%%%%%%%   
% [W1]  = opt_TRR(Sb, Sw, feature_num);
% %     [W, ~] = TraceRatio_fast(Sw, Sb, feature_num, 0);
% [W2, LAMBDA]=eig(Sb-Sw);
% lambda=diag(LAMBDA);
% [lambda, SortOrder]=sort(lambda,'descend');
% W2=W2(:,SortOrder(1:feature_num));
% 
% if trace(W1'*Sw*W1)/trace(W1'*Sb*W1)<trace(W2'*Sw*W2)/trace(W2'*Sb*W2)
%     W = W1;
% else
%     W = W2;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[W]  = opt_TRR(Sb, Sw, feature_num);

% M = invSb*Sw;
% [W, Lambda]=eig(M);
% lambda=diag(Lambda);
% [lambda, SortOrder]=sort(lambda,'ascend');
% W=W(:,SortOrder(1:feature_num));
W = W*diag(1./sqrt(diag(W'*Sb*W)));     %满足约束条件
 
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
    interval = Obj-obj_mole;
    Obj = obj_mole;
    count = count+1;
    obj_wei(count) = objv(X_train, Y_train,W,c);
    disp(['循环次数是：', num2str(count)])
    if  count > 200
        break
    end

end
Y = X_train*W;
end


function val = objv(X_train, Y_train,W,c)
val = 0;
for k = 1:length(c)
    ind = find(Y_train==c(k));
    nk = length(ind);
    valk = 0;
    for i = 1:nk
        for j = 1:nk
            mid = (X_train(i,:)-X_train(j,:))*W;
            valk = valk +norm(mid,2);
        end
    end
    val = val +valk;
end

end
