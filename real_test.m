cd 'C:\Users\walaa\Documents\MATLAB\neural_project\save for testing'
 [audioIn, fs] = audioread('m37.wav');
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


%Test the Network
%a2=logsig(w2*(tansig(w1*mapminmax(x)+b1))+b2);
a= net(features_vector);
y=round(a)
