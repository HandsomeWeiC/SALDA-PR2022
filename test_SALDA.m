clc
clear
addpath('funs')
load('threering_N100.mat'); % data: X  label: Y
% X = data;
label = Y;
[num, dim] = size(X);
num1 = length(label);
if num1 ~=num
    X = X';
    num = num1;
    dim = num;
end   
[Y,s, W, obj] = SALDA(X, Y, 2, 1e-6);
% [Y, W, obj,s] = SALDA_F(X, Y, 2, 1e-6);
S = full(s);
S = S*10;
S(find(S>=255))=255;
Smax = S/(max(max(S))-min(min(S)));
imagesc(S)



