clear all;
clc;

direct = dir;
dirs = zeros([1,length(dir)]);

l = string([]);

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

data = [];
label = [];
sound_attended = [];
sound_masker = [];
for k =progress(1:numel(l))
    s = load(l(k));
    ica = s.ICAcleaned_data_allsess;
    for j = 1:numel(ica)
        data = [data;cell2mat(ica{j}.trial)];
        label = [label;ica{j}.trialid(:,6,:)];
        a = ica{j}.targetfiles;
        m = ica{j}.maskerfiles;
        sound_attended = [sound_attended;cellfun(@myfun,a)];
        sound_masker = [sound_masker;cellfun(@myfun,m)];
    end
end

sound_attended = dirpath+sound_attended;
sound_masker = dirpath + sound_masker


%%

save('att.mat',"sound_attended",'-mat')
save('mas.mat',"sound_masker",'-mat')
save('X.mat',"data",'-v7.3')
save('y.mat',"label")

