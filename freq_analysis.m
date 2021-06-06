clc
clear all
close all

%%
%Design filter for live run filter
load('C:\Users\Allan\Desktop\JRF\Heart_Rate\Python\POS\Dataset_pulse_2.mat')
% infra
x_n=realpulse;
figure(1)
plot(x_n)
x_n=movmean(x_n,8);
Fs=25;
x=x_n-mean(x_n);%Eliminate DC
x=x-std(x);


nfft=length(x);
nfft2=2.^nextpow2(nfft);
fy=fft(x,nfft2);
fy=abs(fy(1:nfft2/2));
xfft=Fs.*(0:nfft2/2-1)/nfft2;
[v,h1]=max(fy);
figure(2)
plot(xfft,(fy/max(fy)))

%Define cut off frequency and calculate filter coefficients
cut_off=[0.5 4]/Fs/2;
order=35;
h=fir1(order,cut_off,'bandpass');

con=filtfilt(h,1,x);
figure(3)
plot(con)

nfft3=length(con);
nfft4=2.^nextpow2(nfft3);
con1=fft(con,nfft4);
con1=abs(con1(1:nfft4/2));
xfft3=Fs.*(0:nfft4/2-1)/nfft4;
figure(4)
plot(xfft3,con1)
