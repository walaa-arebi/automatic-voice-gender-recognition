cd 'C:\Users\walaa\Documents\MATLAB\neural_project'
X=[];
T=[];
Files=dir(['women' '/*.wav']);
for k=1:numel(Files)
 name=Files(k).name;
 cd 'C:\Users\walaa\Documents\MATLAB\neural_project\women'
 [audioIn, fs] = audioread(name);
 a=9.5;
windowLength = round((numel(audioIn)/(27660*a))*fs);
overlapLength = round((numel(audioIn)/(33192*a))*fs);
fs=16000;
features = [];
melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
feat = [melC,f0];

features = [features;feat];
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
features_vector=reshape(features',[],1);
features_vector=features_vector(1:1155,:);
X(:,k) = features_vector;
T(k)=1;
end

cd 'C:\Users\walaa\Documents\MATLAB\neural_project'
Files=dir(['women_real' '/*.wav']);
for l=1:numel(Files)
 name=Files(l).name;
 cd 'C:\Users\walaa\Documents\MATLAB\neural_project\women_real'
 [audioIn, fs] = audioread(name);
 audioIn=reshape(audioIn,[],1);
 new=[];
n=numel(audioIn);
j=0;
for i=1 : n

    if(audioIn(i)~=0)
        j=j+1;
        new(j)=audioIn(i);
    end
end
audioIn=new';
fs=16000;
 a=9.5;
windowLength = round((numel(audioIn)/(27660*a))*fs);
overlapLength = round((numel(audioIn)/(33192*a))*fs);

features = [];
melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
feat = [melC,f0];

features = [features;feat];
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
features_vector=reshape(features',[],1);
features_vector=features_vector(1:1155,:);
X(:,k+l) = features_vector;
T(k+l)=1;
end


cd 'C:\Users\walaa\Documents\MATLAB\neural_project'

Files=dir(['men' '/*.wav']);
for g=1:length(Files)
 name=Files(g).name;
 cd 'C:\Users\walaa\Documents\MATLAB\neural_project\men'
 [audioIn, fs] = audioread(name);
 a=9.5;
windowLength = round((numel(audioIn)/(27660*a))*fs);
overlapLength = round((numel(audioIn)/(33192*a))*fs);
fs=16000;
features = [];
melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
feat = [melC,f0];

features = [features;feat];
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
features_vector=reshape(features',[],1);
features_vector=features_vector(1:1155,:);
X(:,k+l+g) = features_vector;
T(k+l+g)=0;
end

cd 'C:\Users\walaa\Documents\MATLAB\neural_project'

Files=dir(['men_real' '/*.wav']);
for m=1:length(Files)
 name=Files(m).name;
 cd 'C:\Users\walaa\Documents\MATLAB\neural_project\men_real'
 [audioIn, fs] = audioread(name);
 audioIn=reshape(audioIn,[],1);
 new=[];
n=numel(audioIn);
j=0;
for i=1 : n
    if(audioIn(i)~=0)
        j=j+1;
        new(j)=audioIn(i);
    end
end

audioIn=new';
 
fs=16000;
 a=9.5;
windowLength = round((numel(audioIn)/(27660*a))*fs);
overlapLength = round((numel(audioIn)/(33192*a))*fs);

features = [];
melC = mfcc(audioIn,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
f0 = pitch(audioIn,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
feat = [melC,f0];

features = [features;feat];
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
features_vector=reshape(features',[],1);
features_vector=features_vector(1:1155,:);
X(:,k+l+g+m) = features_vector;
T(k+l+g+m)=0;
end
x = X;
t = T;


trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Train the Network
[net,tr] = train(net,x,t);

w1 = net.IW{1} ;%the input-to-hidden layer weights
w2 = net.LW{2} ;%the hidden-to-output layer weights
b1 = net.b{1} ;%the input-to-hidden layer bias
b2 = net.b{2};%the hidden-to-output layer bias

% Test the Network
%a2=logsig(w2*(tansig(w1*mapminmax(x)+b1))+b2);
a= net(X);
y=round(a);
e = gsubtract(t,y);
count=0;
for i=1:numel(e)
if (e(i)~=0)
count=count+1;
end
end
count
audio_num=0;

cd 'C:\Users\walaa\Documents\MATLAB\neural_project'
save("newnetwork.mat");