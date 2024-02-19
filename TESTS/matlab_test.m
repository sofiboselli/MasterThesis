clc
clear all
close all

X = readNPY('X.npy');
y = readNPY('y.npy');
%%
%aud = load('matdic.mat');
aud = readtable('aud.csv');
aud = aud{:,:};
%%
att = aud(:,5);
mas = aud(:,9);    
%%

my_audio_path = './Files_Audio/';

for i = 1:length(att)
    att(i) = append(my_audio_path,att(i));
end

%%

my_audio_path = './Files_Audio/';

for i = 1:length(mas)
    mas(i) = append(my_audio_path, mas(i));
end

%%
test = resample_envelope(att{1,1}(1:end-3));
%%
id = 5;
n = 4;
datapoints = 20;

eeg = X(datapoints*id:datapoints*(id+n),:,5*128:38*128);
attended = resample_envelope(att{1,1}(1:end-3)); %zeros(length(att),0);
masker = resample_envelope(mas{1,1});

for i = datapoints*id:datapoints*(id+n)
    attended =  [attended, resample_envelope(att{i,1}(1:end-3))];
    masker = [masker, resample_envelope(mas{i,1})];
end

eeg = eeg(1:end-1,:,1:end-1);
attended = attended(:,2:end-1);
masker = masker(:,2:end-1);

%%
eeg_size = size(eeg);
nr_samples = eeg_size(1);
nr_train = floor(nr_samples*0.8);
nr_val = floor(nr_samples*0.2);

%%
eeg = normalize(eeg,1);
%%
figure(1)
plot(masker(1,:))

%%
att_cell = num2cell(attended', 2);
eeg_cell = num2cell(permute(eeg,[1,3,2]),[2 3]);
%%
for i = 1:80
    eeg_cell{i} = reshape(eeg_cell{i}, [size(eeg_cell{i}, 2), size(eeg_cell{i}, 3)]);
    att_cell{i} = reshape(att_cell{i}, [size(att_cell{i}, 2), size(att_cell{i}, 1)]);
end
%%
Dir = -1; % direction of causality
tmin = 0; % minimum time lag  -0.3
tmax = 250; % maximum time lag  0.1
lambdas = 10.^(-6:2:6); % regularization values  0.3

cv = mTRFcrossval(att_cell(1:60),eeg_cell(1:60),128,Dir,tmin,tmax,lambdas,'zeropad',0,'fast',1);

%%
model = mTRFtrain(att_cell(1:60),eeg_cell(1:60),128,Dir,tmin,tmax,lambda);
%%
[pred,stats] = mTRFpredict(att_cell(1:60),eeg_cell(1:60),model);
%%
total = 0;
for i = 1:length(pred)
    mas_pred = corr(masker(:,i),pred{i},'type','Pearson')
    att_pred = corr(attended(:,i),pred{i},'type','Pearson')
    if att_pred>mas_pred
        total= total+1;
    end
end
total = total
percentage = total/length(pred)*100

%%

% ---Model training---

% Get optimal hyperparameters
[rmax,idx] = max(mean(cv.r));
lambda = lambdas(idx);
nlambda = length(lambdas);

% Train model
model = mTRFtrain(stimtrain,resptrain,fs,Dir,tmin,tmax,lambda,'zeropad',0);

%%
function data=resample_envelope(filename)
    [x,fs] = audioread(filename);
    %b = fir1(34,[1 30]);
    %freqz(b,1)
    %data = mTRFenvelope(x,128,fs);
    y = hilbert(x);
    dat = abs(y);
    dat = resample(dat,128,fs);
    [b,a] = butter(3,[2 8]/64*2);
    data = filter(b,a,dat);
end

function data=normeeg(eeg)
    data = [];
    for i = 1:length(eeg)
        d = (eeg(i)-mean(eeg(i)))/std(eeg(i));
        data.append(d)
    end
end