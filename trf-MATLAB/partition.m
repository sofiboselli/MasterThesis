function [resptrain,resptest,stimtrain,stimtest,mastrain,mastest] = partition(resp,stim,stim2,ratio)


% resp = resp([21:40,61:80],:);
% stim = stim([21:40,61:80],:);
% stim2 = stim2([21:40,61:80],:);

cv = cvpartition(size(resp,1),'HoldOut',ratio);
idx = cv.test;
% Separate to training and test data
resptrain = resp(~idx,:);
resptest  = resp(idx,:);

stimtrain = stim(~idx,:);
stimtest  = stim(idx,:);

mastrain = stim2(~idx,:);
mastest = stim2(idx,:);

end