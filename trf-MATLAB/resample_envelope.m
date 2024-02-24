function data=resample_envelope(filename,ds,bpFilter)
    [x,sr] = audioread(filename);
    x = resample(x,8000,sr);
    [x,~] = envelope(x);
    x = resample(x,128,8000);
    %x = downsample(abs(x),44032/ds);
    [b,a] = butter(3,[1 9]/128*2);
    data = filter(b,a,x);
    data = resample(data,64,128);
    data = normalize(data);
    %data = normalize(data,'range');
end