function [  ] = Metaheuristic_ANFIS
data = importdata('BoxJenkins.mat');
data = [data(3:end,2),data(2:end-1,2),data(1:end-2,2),data(3:end,1),data(2:end-1,1)];
X = data(:,2:end);
T = data(:,1);

xTrain = X(1:size(X,1)/2,:);
tTrain = T(1:size(X,1)/2,:);
xTest = X(size(X,1)/2+1:end,:);
tTest = T(size(X,1)/2+1:end,:);

nInput = size(xTrain,2);

upper_input = max(X);
lower_input = min(X);
upper_sigma = 50*ones(1,nInput);
lower_sigma = 1e-5*ones(1,nInput);
upper_weight = 6*ones(1,nInput);
lower_weight = -6*ones(1,nInput);
upper_bias = 6;
lower_bias = -6;

nRule = 3;

UB = repmat([upper_input, upper_sigma, upper_weight, upper_bias],1,nRule);
LB = repmat([lower_input, lower_sigma, lower_weight, lower_bias],1,nRule);

dim = nRule*(nInput*3+1);


objFun = @(solution)fitnessFcn(solution,nRule,xTrain,tTrain);

solution = PSO(objFun,100,50000,dim,LB,UB);


model = constructor(solution,nInput,nRule);
output = estimator(xTest,model);

subplot(2,1,1)
hold on
plot(tTest,'b')
plot(output,'r')
subplot(2,1,2)
plot(tTest-output,'b')
end

function model = constructor(solution,nInput,nRule)
solution = reshape(solution,length(solution)/nRule,nRule)';
model.center = solution(:,1:nInput);
model.sigma = solution(:,nInput+1:nInput*2);
model.weight = solution(:,nInput*2+1:end-1)';
model.bias = solution(:,end)';
model.nRule = nRule;
end

function output = estimator(X,model)
N = size(X,1);
Rule = inf(N,model.nRule);
for i = 1:model.nRule
    MF = exp(-0.5*((X-repmat(model.center(i,:),N,1))./repmat(model.sigma(i,:),N,1)).^2);
    Rule(:,i) = prod(MF,2);
end
normalize = Rule./repmat(sum(Rule,2),1,model.nRule);
u = X*model.weight+repmat(model.bias,N,1);
output = sum(u.*normalize,2);
end

function fitness = fitnessFcn(solution,nRule,X,T)
model = constructor(solution,size(X,2),nRule);
Y = estimator(X,model);
fitness = (Y-T)'*(Y-T);
end
