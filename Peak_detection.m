clc
clear all
close all

fs=25;
Ts=1/10;
load('C:\Users\Allan\Desktop\JRF\Heart_Rate\Python\POS\Dataset_pulse_2.mat')

%x=con(30:end);
%x=depth_filt;
%x=Force;
x=realpulse;

x=movmean(x,8);
final_digi=ones(length(x),1);
t=linspace(0,50,length(x));
short_win=zeros([1 10]);
j=1;
k=1;
l=1;
y=1;
z=1;
l1=10;
H=30;
HOLD=0;
PEAK_FLAG=0;
VALLEY_FLAG=0;
HOLD_FLAG=0;
tot_time=[];
g=1;
wind_time=length(short_win)*[1:length(x)/length(short_win)];
%%
for i =1:length(x)
    short_win=circshift(short_win,1);
    short_win(:,1)=x(i);
    HOLD_FLAG(i)=0;
    if mod(i,length(short_win))==0
        short_std(j)=std(short_win);
        
        
        if j<=H  
            long_std(k)=median(short_std(1:j));
            
        else
            long_std(k)=median(short_std(j-H:j-1));
            
        end
        
        if short_std(j)/long_std(k)<=0.33
            HOLD_FLAG(i)=1;
            window(l)=j;
        k=k+1;   
        if length(window)>=2 && window(l)-window(l-1)==1
            tot_time=[tot_time wind_time(l-1):wind_time(l)];
           % tot_time(1:end)=1;
            final_digi(tot_time)=0;
        end
%if length(window)>=2 & window(l-1)==j-1
%                 temp_window(g)=j;
%                 g=g+1;
%                 end
            l=l+1;
        end
        j=j+1;
%         if ismember(j-1,window)
%             idx=find(diff(window)==1);
%             result=find(diff(idx)==1);
%         end
            
        
%         if mod(length(short_std),4)==0
%             long_std(k)=median(short_std);
%             k=k+1;
    end
    
       
     
    if i>=2
        con=(x(i)-x(i-1));
        d(i)=sign(con);
        if d(i)~=d(i-1)
            if d(i)==-1
                if x(i-1)>mean(x) & VALLEY_FLAG==0 
                    peak(y)=i-1;
                    y=y+1;
                    PEAK_FLAG=0;
                    VALLEY_FLAG=1;
                end
            else if d(i)==1
                    if x(i-1)<mean(x) & PEAK_FLAG==0 & VALLEY_FLAG==1
                        valley(z)=i-1;
                        PEAK_FLAG=1;
                        VALLEY_FLAG=0;
                        z=z+1;
                    end
                end
            end
        end
    end
end

figure(7)
plot(t,x,'LineWidth',3)
hold on
for i=1:length(window)
    xline(t(wind_time(window(i))),'LineWidth',3)  
    %wind_time(window(i))=1;
%     if wind_time(i)~=1
%         wind_time(i)=0
%     end
end
% for i=1:length(wind_time)
%     xline(wind_time(i),'LineWidth',1);
%     %wind_time(i)=1;
% end

figure(8)
plot(t,x,'LineWidth',3)
hold on
plot(t(peak),x(peak),'x','LineWidth',3,'MarkerSize',7)
% plot(t(valley),x(valley),'o','LineWidth',3,'MarkerSize',7)    
% peak=t(peak);
% valley=t(valley);
%%
figure(3)
plot(final_digi)

digi_sig=zeros(298,1);
len=1;
wind_count=10;

% for i=1:length(wind_time)
%     for j=1:9
%         if wind_count~=1
%             
%             tot_len=wind_time(i)-wind_count;
%             wind_count=wind_count-1;
%             
%             digi_sig(len)=tot_len;
%             len=len+1;
%         else 
%             wind_count=9;
%         end
%         end
% end
new_wind=wind_time(window);
final_wind=zeros(length(window));
digi_sig=1:length(x);
% for i=1:length(new_wind)
%     for j=1:10
%         if wind_count~=1
%             tot_wind=new_wind(i)-wind_count;
%             wind_count=wind_count-1;
%             final_wind(len)=tot_wind;
%             len=len+1;
%         else
%             wind_count=10;
%         end
%     end
% end
final_wind(final_wind==0)=[];
digi_sig(final_wind)=0;
digi_sig(digi_sig~=0)=1;
subplot(211)
plot(t,x,'LineWidth',3)
xlim([0 60])
subplot(212)
plot(t,digi_sig)
ylim([0 2])
xlim([0 60])

t1=linspace(0,50,length(y))
figure(5)
subplot(311)
plot(t1,y)
yyaxis right
plot(t,x)
subplot(312)
plot(t,x)
subplot(313)
plot(t,digi_sig)
ylim([0 1.5])


% all_length=zeros(1,298);
% for i=1:length(digi_sig)
%     plot(digi_sig(i),'LineWidth',2)
%     all_length((digi_sig(i)))=1;
% end