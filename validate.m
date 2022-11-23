cd 'C:\Users\walaa\Documents\MATLAB\neural_project'
if(y==1)
    t=0;
else(y==0)
    t=1;
end  

X= [X features_vector];
T=[T t];

trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Train the Network
[net,tr] = train(net,X,T);
 
a=round(net(features_vector))

