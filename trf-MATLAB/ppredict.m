function [result,stats1,stats2] = ppredict(model,stimtest,resptest,mastest)

[~,stats1] = mTRFpredict(stimtest,resptest,model,'verbose',false);
[~,stats2] = mTRFpredict(mastest,resptest,model,'verbose',false);

[result,d] = mTRFattnevaluate(stats1.r,stats2.r);

result = result*100;

end