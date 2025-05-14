%[y,Fs]=audioread('Signal.wav'); % the signal is in y and sampling freq in Fs
%sound(y,Fs); pause(15); % Play the original sound
%Ts = 1/Fs;
%L = length(y);
%t = [0:L-1]*Ts;
%plot(t,y); grid;
%alpha = 0.9; D = 8000; % Echo parameters
%b = [1,zeros(1,D),alpha]; % Filter parameters
%x = filter(b,1,y); % Generate sound plus its echo
%sound(x,Fs); % Play sound with echo

%x = linspace(0, 4*pi, 10);
%y = sin(x);

%p0 = polyfit(x,y,0);
%x1 = linspace(0,4*pi);
%y1 = polyval(p0,x1);
%figure; plot(x,y,'o');title('0th-Degree Polynomial Fit'); xlabel('x');
%ylabel('sin(x)');
%legend('Data Points', '0th-Degree Fit');hold on;plot(x1,y1); hold off;

%p1 = polyfit(x,y,1);
%x2 = linspace(0,4*pi);
%y2 = polyval(p1,x2);
%figure; plot(x,y,'o');hold on;plot(x2,y2); hold off;

%p7 = polyfit(x,y,7);
%x3 = linspace(0,4*pi);
%y3 = polyval(p7,x3);
%figure; plot(x,y,'o');hold on;plot(x3,y3); hold off;

[y,Fs]=audioread('Signal.wav');
Sig_Chunk = y(1:10000);
t = (0:length(Sig_Chunk)-1)/Fs;
%plot(t,Sig_Chunk,'o');xlabel('Time (s)');ylabel('Amplitude');
%title('Samples of the Signal Chunk');grid on;
p = polyfit((0:length(Sig_Chunk)-1), Sig_Chunk, 2);
y1 = polyval(p,(0:length(Sig_Chunk)-1)');
fitted_y1 = fit((0:length(Sig_Chunk)-1)', Sig_Chunk, 'smoothingspline');
y2 = feval(fitted_y1, (0:length(Sig_Chunk)-1)');

figure;
subplot(211);plot(t, Sig_Chunk, 'o', 'DisplayName', 'Original Samples');  % Original data
hold on;
subplot(212);plot(t, y2, 'r-', 'DisplayName', 'Fitted Spline');  % Fitted spline
xlabel('Time (s)');ylabel('Amplitude');title('Signal Chunk with Fitted Spline');

