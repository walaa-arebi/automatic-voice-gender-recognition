cd 'C:\Users\walaa\Documents\MATLAB\neural_project'
X=[];
T=[];
Files=dir(['real_women' '/*.wav']);
for k=1:length(Files)
 name=Files(k).name;
 cd 'C:\Users\walaa\Documents\MATLAB\neural_project\real_women'
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
X(:,k) = features_vector;
T(k)=1;
end



cd 'C:\Users\walaa\Documents\MATLAB\neural_project'

Files=dir(['real_men' '/*.wav']);
for g=1:length(Files)
 name=Files(g).name;
 cd 'C:\Users\walaa\Documents\MATLAB\neural_project\real_men'
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
X(:,k+g) = features_vector;
T(k+g)=0;
end


% Test the Network
%a2=logsig(w2*(tansig(w1*mapminmax(x)+b1))+b2);
a= net(X);
y=round(a);
e = gsubtract(T,y);
count=0;
for i=1:numel(e)
if (e(i)~=0)
count=count+1;
end
end
count


