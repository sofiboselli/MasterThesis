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

for i=progress(randperm(80))


    [eeg,att,mas] = loadblock(data,sound_attended,sound_masker,i);
    
    
    
    [attended,masker] = audioloadresamp(att',mas',ds,bpFilter);
    
    [resp,stim1,stim2] = preparedata(eeg,attended,masker,ds,bpFilter);
    
   
    
    %[resptrain,resptest,stimtrain,stimtest,mastrain,mastest] = partition(resp,stim1,stim2,0.125);
    
    
    
    Dir = -1; % direction of causality
    tmin = 0; % minimum time lag -0.3
    tmax =  500; % maximum time lag 0.1
    lambda =1e5; % regularization values  0.3 
    
   
    
    
    [st1,st2] = mTRFattnnestedcrossval(stim1,stim2,resp,ds,Dir,tmin,tmax,lambda,'verbose',0);
    
    [result,d] = mTRFattnevaluate(st1',st2');
    
    
    results(i+1) = result;
    
    
    corr1 = [corr1 st1'];
    corr2 = [corr2 st2'];
end


%%
model = mTRFtrain(stim1(1:20),resp(1:20),64,Dir,tmin,200,lambda,'verbose',0)

[pred,s] = mTRFpredict(stim2(20),resp(20),model);

%%
function [ BP_equirip ] = construct_bpfilter( params )

Fs = params.intermediateSampleRate;
Fst1 = params.highpass-0.45;
Fp1 = params.highpass+0.45;
Fp2 = params.lowpass-0.45;
Fst2 = params.lowpass+0.45;
Ast1 = 20; %attenuation in dB
Ap = 0.5;
Ast2 = 15;
BP = fdesign.bandpass('Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2',Fst1,Fp1,Fp2,Fst2,Ast1,Ap,Ast2,Fs);
BP_equirip = design(BP,'equiripple');

end






