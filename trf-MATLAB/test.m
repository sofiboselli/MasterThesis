clc
clear all
close all

load ExJobb\X.mat;
load ExJobb\mas.mat;
load ExJobb\att.mat;

%%

ds = 64;
results = [];
corr1 = [];
corr2 = [];



for i = progress(0:12)

[eeg,att,mas] = loadsubject(data,sound_attended,sound_masker,i);

[attended , masker] = audioloadresamp(att,mas,ds);
[resp,stim1,stim2] = preparedata(eeg,attended,masker,ds);


[resptrain,resptest,stimtrain,stimtest,mastrain,mastest] = partition(resp,stim1,stim2,0.125);



Dir = -1; % direction of causality
tmin = 0; % minimum time lag -0.3
tmax =  500; % maximum time lag 0.1
lambdas = [1e-6,1e-4,1e-2,1,1e2,1e4,1e6]; % regularization values  0.3


[cv,st1,st2] = mTRFattncrossval(stimtrain,mastrain,resptrain,ds,Dir,tmin,tmax,lambdas,'fast',1,'verbose',0);


[r_max,idx] = max(cv.d);
lambda = lambdas(idx);
nlambda = length(lambdas);


model = mTRFtrain(stimtrain,resptrain,ds,Dir,tmin,tmax,lambda,'verbose',0);
[train_accuracy,~,~] = ppredict(model,stimtrain,resptrain,mastrain);
[result,stats1,stats2] = ppredict(model,stimtest,resptest,mastest);


results(i+1) = result;

corr1 = [corr1 stats1.r];
corr2 = [corr2 stats2.r];

end

%%

bar(results)







%%

plot(mean(corr1))
hold on
plot(mean(corr2))






