function [W]  = opt_TRR(Sb, Sw, feature_num)
% Sb, Sw: scatter matrices
% feature_num: reduced dimensionality
% p_err: the error to optimum


[evec_sw, eval_sw] = eig(Sw);%特征向量，特征值
eval_sw = abs(diag(eval_sw));
nzero_sw = length(find(eval_sw<=1e-6));
if feature_num <= nzero_sw
    [dumb, iEvals] = sort(eval_sw);
    Z = evec_sw(:,iEvals(1:nzero_sw));
    [evec_sb, eval_sb] = eig(Z'*Sb*Z);
    [dumb, iEvals] = sort(diag(eval_sb), 'descend');
    W = Z * evec_sb(:,iEvals(1:feature_num));
else
    [evec_sb, eval_sb] = eig(Sb);
    eval_sb = sort(diag(eval_sb), 'descend');
    max_numerator = sum(eval_sb(1:feature_num));
    [evec_sw, eval_sw] = eig(Sw);
    eval_sw = sort(diag(eval_sw));
    min_denominator = sum(abs(eval_sw(1:feature_num)));
    lamda_sup = max_numerator/min_denominator;
    lamda_inf = trace(Sb)/trace(Sw);
    interval = lamda_sup - lamda_inf;
    lamda = (lamda_inf+lamda_sup)/2;
%     while interval > p_err
    [evec, eval] = eig(Sb - lamda*Sw);
    [eval, index] = sort(diag(eval),'descend');
    sum_eval = sum(eval(1:feature_num));
    if sum_eval > 0
        lamda_inf = lamda;
    else
        lamda_sup = lamda;
    end;
    interval = lamda_sup - lamda_inf;
    lamda = (lamda_inf+lamda_sup)/2;
%     end;
    W = evec(:,index(1:feature_num));
end;