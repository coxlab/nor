
r1=transpose(X3_30_Rs(:,1)+1);
r2=transpose(X3_30_Rs(:,2)+1);

Q = [columnA columnF columnG columnI columnJ columnL columnN]; % [ FrameNumber Headx Heady Tailx Taily Centroidx Centroidy]

for i = 1:length(r1)
    X = Q(r1(i):r2(i),2:3);
    Y = Q(r1(i):r2(i),4:5);
    Q(r1(i):r2(i),2:3) = Y(:,:);
    Q(r1(i):r2(i),4:5) = X(:,:);
end

Final_m = [Q(1:7500,2:3), Q(1:7500,6:7), Q(1:7500,2)-Q(1:7500,6), Q(1:7500,3)-Q(1:7500,7)];
FramNum = transpose(Q(1:7500,1));
posit_direc=transpose(Final_m); % compiles relevant data in one matrix (frame number ,position, then head direction by row)
xaxis = [ 1,0];
%procframnum = zeros(1, length(posit_direc(1,:))); % number of the frame (which comes from modified SOS processing)
FvB = zeros(1, length(posit_direc(1,:))); % 1 = facing; -1 = back to; 0 = noise/left/right 
Side = zeros(1, length(posit_direc(1,:))); % 1 = left; -1 = right; 0 = midline

for i=1:length(posit_direc(1,:)) % Run through every processed frame 

    theta = rad2deg(atan2(xaxis(1)*posit_direc(6,i)-posit_direc(5,i)*xaxis(2),xaxis(1)*posit_direc(5,i)+xaxis(2)*posit_direc(6,i))); %calculate angle of head direction relative to pre defined x-axis
% Facing stimulus
if theta < 0 
    if -170 < theta && theta < -10 % accounting for noise at the midline (could be looking left or right)
        FvB(i) = 1;
    else 
        FvB(i) = 0;
    end
%Back to stimulus    
elseif theta > 0 
    if 10 < theta && theta < 170 % accounting for noise at the midline (could be looking left or right)
        FvB(i) = -1;
    end
else
    FvB(i) = 0;
end

%What side ofthe box is animal on? (using head point, which theoretically,
%is the tip of the nose
if posit_direc(1,i) < 390 % rat's head is on left side of box
    Side(i) = 1;
elseif posit_direc(1,i) > 390 % rat's head is on right side of box
    Side(i) = -1;
else %rat's head is on center line
    if posit_direc(3,i) < 390 % rat's centroid is on left -> rat is on left
        Side(i) = 1;
    elseif posit_direc(3,i) > 390 % rat's centroid is on right -> rat is on right
        Side(i) = -1;
    else % rat's head and centroid are on center line -> can't determine side
        Side(i) = 0;
    end
end

end
% Check if Head was within 2 centimeter region 

X2cmL = [193 220 220 347 347 337]'; % x-coordinates of 2cm on left
Y2cmL = [91 145 154 127 119 65]'; % y coordinates of 2cm box on left
X2cmR = [428 424 424 555 555 584]'; % x coordinates of 2cm on left
Y2cmR = [62 120 126 145 141 90]'; % y coordinates of 2cm box on right
inL2cm = inpolygon(Final_m(:,1),Final_m(:,2),X2cmL,Y2cmL); % Check to see if head coordinate is in the 2cm region
inR2cm = inpolygon(Final_m(:,1),Final_m(:,2),X2cmR,Y2cmR); % Check to see if head coordinate is in the 2cm region

head_direction_data = [FramNum; FvB; Side];

%FPS ~= 25 so each frame is ~1/25 of a second long

LookAt = (sum(FvB == 1)) / 25; % Time spent looking at stimulus in seconds
LookAw = (sum(FvB == -1)) / 25; % Time spent looking away from stimulus in seconds
LookN = (sum(FvB == 0)) / 25; % Time lost to noise/ loking at nothing
OnLeft = (sum(Side == 1)) / 25; % Time spent on left side of box in seconds
OnRight = (sum(Side == -1)) / 25; % Time spent on right side of box in seconds
Midline = (sum(Side == 0)) / 25; % Time spent on the midline of box in seconds
in2cmL = (sum(inL2cm))/25; % Time spent with head in 2cm box on left 
in2cmR = (sum(inR2cm))/25; % Time spent with head in 2cm box on right 
TotTime = length(FramNum)/(25);

OnLeft_Looking = 0; 
%OnLeft_LookingRight = 0;
OnRight_Looking = 0;
%OnRight_LookingRight = 0;
for n = 1:length(head_direction_data(1,:))
    if Side(n) == 1 && FvB(n) == 1
        OnLeft_Looking = OnLeft_Looking +1;
    %elseif Side(n) == 1 && FvB(n) == -1
        %OnLeft_LookingRight = OnLeft_LookingRight +1;
    elseif Side(n) == -1 && FvB(n) == 1
        OnRight_Looking = OnRight_Looking +1;
    %elseif Side(n) == -1 && FvB(n) == -1
        %OnRight_LookingRight = OnRight_LookingRight +1;
    end
end
OnLeft_Looking = OnLeft_Looking/25; % Time spent looking on left side of box in seconds
%OnLeft_LookingRight = OnLeft_LookingRight/25; % Time spent looking left on right side of box in seconds
OnRight_Looking = OnRight_Looking/25; % Time spent looking on right side of box in seconds
%OnRight_LookingRight = OnRight_LookingRight/25; % Time spent looking right on right side of box in seconds

FillinOutput = [OnLeft, OnLeft_Looking, NaN , OnRight, NaN, OnRight_Looking, NaN, NaN, in2cmL, in2cmR, TotTime]; 
copy(FillinOutput);
