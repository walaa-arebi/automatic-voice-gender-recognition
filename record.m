cd 'C:\Users\walaa\Documents\MATLAB\neural_project'
audio_num=audio_num+1;;
audio=audiorecorder(16000,8,1)
display("say Hello clearly")
recordblocking(audio,3);
sound=getaudiodata(audio);
play(audio)
filename = "online_testing\audioIn"+num2str(audio_num)+".wav";
audiowrite(filename, sound, 16000);
new=[];
k=numel(sound);
j=0;
for i=1 : k
    if(sound(i)~=0)
        j=j+1;
        new(j)=sound(i);
    end
end

new=new';
fs=16000;
a=9.5;
windowLength = round((numel(new)/(27660*a))*fs);
overlapLength = round((numel(new)/(33192*a))*fs);

features = [];
melC = mfcc(new,fs,'Window',hamming(windowLength,'periodic'),'OverlapLength',overlapLength);
f0 = pitch(new,fs,'WindowLength',windowLength,'OverlapLength',overlapLength);
feat = [melC,f0];

features = [features;feat];
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
features_vector=reshape(features',[],1);
features_vector=features_vector(1:1155,:);

a= net(features_vector);
y=round(a)
