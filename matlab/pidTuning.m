d = sim("pidtuningcontroller").simout;
y = d.Data;
t = d.Time;

stepinfo(y, t, 1.0)