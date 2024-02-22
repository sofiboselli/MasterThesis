function data=resample_envelope(filename,ds)
    [x,~] = audioread(filename);
    b = -1;
    a = 1;
    %x = ((x - min(x))./(max(x)-min(x)))*(b-a)+a;
    %x = ((x - min(x))./(max(x)-min(x)))*-1+1;
    %x = x(44100*0.5+1:end-44100*0.5,:);
    x = resample(x,44032,44100);
    x = downsample(x,44032/ds);
    x = abs(hilbert(x));
   % x = downsample(abs(x),44032/ds);
    %data = mTRFenvelope(x,44032,64);
    [b,a] = butter(3,[1 8]/(ds/2));
    data = filter(b,a,x);
    data = normalize(data);
    data = normalize(data,'range');
end