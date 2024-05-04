clc
clear all
close all

%load ExJobb\X.mat;
%%
load C:\Users\gauta\Thesis\mas.mat;
load C:\Users\gauta\Thesis\att.mat;



%%

ds = 64;
results = [];
corr1 = [];
corr2 = [];

%for i=progress(randperm(80))    

direct = dir("C:\Users\gauta\Thesis\ExJobb");
dirs = zeros([1,length(dir)]);

l = string([]);
sounds = dir('ExJobb\Files_Audio');
sounds = sounds(3:end);

sounds = {sounds.name};



dirpath = direct.folder;
names = {direct.name};
for j = 1:numel(names)
    if names{j}(1)=='I'
        m = dirpath + "\" + names{j} + "\PreprocIV_" + names{j} + "_mast.mat";
        l(end+1) = m;
    else
        continue;
    end
end


%%


Dir = -1; % direction of causality
tmin = -100; % minimum time lag -0.3
tmax =  400; % maximum time lag 0.1
%lambda = 10.^[-2:2:10]; % regularization values %0.3 
lambda = 1e5;
corr1 = [];
corr2 = [];
results = [];
rng('default')

for i = progress(1:10)
    data = load(l(i));
    data = data.ICAcleaned_data_allsess;

    EEG = {};
    att = {};
    mas = {};

    for z = 1:numel(data)
        tar = data{z}.targetfiles;
        mask = data{z}.maskerfiles;

        eeg = data{z}.trial;
        eeg = cellfun(@prep_eeg,eeg,'UniformOutput',false);
        attended = cellfun(@splitaudio,tar,'UniformOutput',false);
        masker = cellfun(@splitaudio,mask,'UniformOutput',false);
        
        EEG = [EEG eeg];
        att = [att attended];
        mas = [mas masker];
    end
        n = length(EEG);
        part = cvpartition(n,'HoldOut',0.12);
        idtrain = training(part);
        idtest = test(part);
        EEG_train = EEG(idtrain);
        att_train = att(idtrain);
        mas_train = mas(idtrain);
        EEG_test = EEG(idtest);
        att_test = att(idtest);
        mas_test = mas(idtest);

        EEG_train = cellfun(@(x) splitCellArray(x,11),EEG_train,'UniformOutput',false);
        att_train = cellfun(@(x) splitCellArray(x,11),att_train,'UniformOutput',false);
        mas_train = cellfun(@(x) splitCellArray(x,11),mas_train,'UniformOutput',false);

        EEG_train = horzcat(EEG_train{:});
        att_train = horzcat(att_train{:});
        mas_train = horzcat(mas_train{:});

        EEG_test = cellfun(@(x) splitCellArray(x,11),EEG_test,'UniformOutput',false);
        att_test = cellfun(@(x) splitCellArray(x,11),att_test,'UniformOutput',false);
        mas_test = cellfun(@(x) splitCellArray(x,11),mas_test,'UniformOutput',false);

        EEG_test = horzcat(EEG_test{:});
        att_test = horzcat(att_test{:});
        mas_test = horzcat(mas_test{:});


        


        
    model = mTRFtrain(att_train',EEG_train',ds,Dir,tmin,tmax,lambda,'verbose',0,'zeropad',0);

    [pred,st1] = mTRFpredict(att_test',EEG_test',model);
    [pred,st2] = mTRFpredict(mas_test',EEG_test',model);

    %[~,idx] = max(mean(st1.r));

    [result,d] = mTRFattnevaluate(st1.r,st2.r);
    results(i) = result;
    corr1 = [corr1 st1.r];
    corr2 = [corr2 st2.r];

end



        



%%




%%




[st1,st2] = mTRFattnnestedcrossval(stim2,stim1,resp,ds,Dir,tmin,tmax,lambda,'verbose',0);

[result,d] = mTRFattnevaluate(st1',st2');


results(i+1) = result;


corr1 = [corr1 st1'];
corr2 = [corr2 st2'];
%end


%%
model = mTRFtrain(stim2(1:19),resp(1:19),64,Dir,tmin,500,lambda,'verbose',1)


[pred,s] = mTRFpredict(stim2(20),resp(20),model);








