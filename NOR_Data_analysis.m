r1=transpose(X3_13_Rs(:,1)+1);
r2=transpose(X3_13_Rs(:,2)+1);

Q = [columnA columnF columnG columnI columnJ columnL columnR]; % [ FrameNumber Headx Heady Tailx Taily Centroidx Centroidy]

for i = 1:length(r1)
    X = Q(r1(i):r2(i),2:3);
    Y = Q(r1(i):r2(i),4:5);
    Q(r1(i):r2(i),2:3) = Y(:,:);
    Q(r1(i):r2(i),4:5) = X(:,:);
end

Final_m = [Q(:,2:3), Q(:,6:7), Q(:,2)-Q(:,4), Q(:,3)-Q(:,5)];
FramNum = transpose(Q(:,1));
posit_direc=transpose(Final_m); % compiles relevant data in one matrix (frame number ,position, then head direction by row)
xaxis = [ 1,0];
%procframnum = zeros(1, length(posit_direc(1,:))); % number of the frame (which comes from modified SOS processing)
LvR = zeros(1, length(posit_direc(1,:))); % 1 = left; -1 = right; 0 = noise/forward/backward 
Side = zeros(1, length(posit_direc(1,:))); % 1 = left; -1 = right; 0 = midline

for i=1:length(posit_direc(1,:)) % Run through every processed frame 
%    procframnum(i) = i;
    theta = rad2deg(atan2(xaxis(1)*posit_direc(6,i)-posit_direc(5,i)*xaxis(2),xaxis(1)*posit_direc(5,i)+xaxis(2)*posit_direc(6,i))); %calculate angle of head direction relative to pre defined x-axis
% looking to the left
if theta > 0 
    if 10 < theta && theta < 170 % accounting for noise at the midline (could be looking forward or backward)
        LvR(i) = 1;
    else 
        LvR(i) = 0;
    end
%looking to the right    
elseif theta < 0 
    if -170 < theta && theta < -10 % accounting for noise at the midline (could be looking forward or backward)
        LvR(i) = -1;
    end
else
    LvR(i) = 0;
end

%What side ofthe box is animal on? (using head point, which theoretically,
%is the tip of the nose
if posit_direc(2,i) < 250 % rat's head is on left side of box
    Side(i) = 1;
elseif posit_direc(2,i) < 250 % rat's head is on right side of box
    Side(i) = -1;
else %rat's head is on center line
    if posit_direc(4,i) < 250 % rat's centroid is on left -> rat is on left
        Side(i) = 1;
    elseif posit_direc(4,i) > 250 % rat's centroid is on right -> rat is on right
        Side(i) = -1;
    else % rat's head and centroid are on center line -> can't determine side
        Side(i) = 0;
    end
end

end

head_direction_data = [FramNum; LvR; Side];

%FPS ~= 25 so each frame is ~1/25 of a second long

LookL = (sum(LvR == 1)) / 25; % Time spent looking left in seconds
LookR = (sum(LvR == -1)) / 25; % Time spent looking right in seconds
LookN = (sum(LvR == 0)) / 25; % Time lost to noise
OnLeft = (sum(Side == 1)) / 25; % Time spent on left side of box in seconds
OnRight = (sum(Side == -1)) / 25; % Time spent on right side of box in seconds
Midline = (sum(Side == 0)) / 25; % Time spent on the midline of box in seconds
TotTime = length(FramNum)/(25);

OnLeft_LookingLeft = 0; 
OnLeft_LookingRight = 0;
OnRight_LookingLeft = 0;
OnRight_LookingRight = 0;
for n = 1:length(head_direction_data(1,:))
    if Side(n) == 1 && LvR(n) == 1
        OnLeft_LookingLeft = OnLeft_LookingLeft +1;
    elseif Side(n) == 1 && LvR(n) == -1
        OnLeft_LookingRight = OnLeft_LookingRight +1;
    elseif Side(n) == -1 && LvR(n) == 1
        OnRight_LookingLeft = OnRight_LookingLeft +1;
    elseif Side(n) == -1 && LvR(n) == -1
        OnRight_LookingRight = OnRight_LookingRight +1;
    end
end
OnLeft_LookingLeft = OnLeft_LookingLeft/25; % Time spent looking left on left side of box in seconds
OnLeft_LookingRight = OnLeft_LookingRight/25; % Time spent looking left on right side of box in seconds
OnRight_LookingLeft = OnRight_LookingLeft/25; % Time spent looking right on left side of box in seconds
OnRight_LookingRight = OnRight_LookingRight/25; % Time spent looking right on right side of box in seconds

FillinOutput = [OnLeft, OnLeft_LookingLeft, OnLeft_LookingRight, OnRight, OnRight_LookingLeft, OnRight_LookingRight, LookL, LookR, TotTime]; 
copy(FillinOutput);
