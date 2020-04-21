sys = tf([0.2],[1 0 0]);
lead = tf([168 0],[1 8])
lag = tf([1 1.547],[1 0.0065]);

% % Plot root locus
% figure(1);
% hold on;
% rlocus(sys)
% rlocus(lead*sys)
% rlocus(lag*lead*sys)
% hold off;


% % Plot bode plots
% figure(1);
% hold on;
% bode(sys)
% bode(lead*sys)
% bode(lag*lead*sys)
% hold off;


syscl = feedback(lag*lead*sys, 1);

% % Plot ramp response
% t = 0:0.1:10;  % the time vector
% input = t;
% [y,t] = lsim(syscl, input, t);
% figure(1);
% hold on;
% plot(t,transpose(input)-y);
% hold off;

% % Plot step response
% step(syscl)
% stepinfo(syscl)

% % Discretise controller using ZOH 
% cdz = c2d(lag*lead,0.02)