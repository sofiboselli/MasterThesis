function out = splitaudio(att)

temp = split(att,"\");
out = temp{end};
ds = 64;
[x,sr] = audioread(out);

[b,a] = butter(3,[1 8]/(256/2),'bandpass');

aud = abs(hilbert(x));

aud = resample(aud,256,sr);

aud = filtfilt(b,a,aud);

aud = resample(aud,ds,256);

out = normalize(aud);

%out = aud;




end