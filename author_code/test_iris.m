clear all

global KTYPE
global KSCALE

KTYPE = 1;  % gaussian kernel
C = 1;      % soft margin regularization parameter

load iris_train
load iris_test
[L,N] = size(xtrain);
disp(size(xtrain));
disp(size(ytrain));
% if L~=768 | N~=8
%     fprintf('diabetes data error: wrong matrix dimensions\n')
%     return
% end
% Ltrain = 576;

% xtrain = x_norm(1:Ltrain,:);
% ytrain = y(1:Ltrain);
% xtest =x_norm(Ltrain+1:L,:);
% ytest =y(Ltrain+1:L);

[a,b,D,inds,inde,indwLOO,iter, process] = svcm_train(xtrain, ytrain, C);
disp('iTER COUNT: ')
disp(sum(process))
disp(iter)
disp('DONE TRAINING')
[ypred,indwTEST] = svcm_test(xtest,ytest,xtrain,ytrain,a,b);
disp('DONE TESTING')