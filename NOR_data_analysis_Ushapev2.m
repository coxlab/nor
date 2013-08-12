
r1=transpose(X1_30_Rs(:,1)+1);
r2=transpose(X1_30_Rs(:,2)+1);

Q = [columnA columnC columnD columnF columnG columnL columnN]; % [ FrameNumber Headx Heady Tailx Taily Centroidx Centroidy]

for i = 1:length(r1)
    X = Q(r1(i):r2(i),2:3);
    Y = Q(r1(i):r2(i),4:5);
    Q(r1(i):r2(i),2:3) = Y(:,:);
    Q(r1(i):r2(i),4:5) = X(:,:);
end

Final_m = [Q(:,2:3), Q(:,6:7), Q(:,2)-Q(:,6), Q(:,3)-Q(:,7)];
FramNum = transpose(Q(:,1));
posit_direc=transpose(Final_m); % compiles relevant data in one matrix (frame number ,position, then head direction by row)

xaxis = [ 1,0];
%procframnum = zeros(1, length(posit_direc(1,:))); % number of the frame (which comes from modified SOS processing)
FvB = zeros(1, length(posit_direc(1,:))); % 1 = facing; -1 = back to; 0 = noise/left/right 
Side = zeros(1, length(posit_direc(1,:))); % 1 = left; -1 = right; 0 = midline
P1 = [221,146]; %Left side of left monitor
P2 = [348,108]; %Right side of left monitor
P3 = [431,108]; %Left side of right monitor
P4 = [561,146]; %Right side of right monitor
V1 = [0,0]; % one side of "peripheral" vision of stimulus on left
V2 = [0,0]; % other side of "peripheral" vision of stimulus on left
V3 = [0,0]; % one side of "peripheral" vision of stimulus on right 
V4 = [0,0]; % other side of "peripheral" vision of stimulus on right
for i= 1:length(posit_direc(1,:)) % Run through every processed frame 
%    procframnum(i) = i;
%What side of the box is animal on? (using head point, which theoretically,
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

theta = rad2deg(atan2(xaxis(1)*posit_direc(6,i)-posit_direc(5,i)*xaxis(2),xaxis(1)*posit_direc(5,i)+xaxis(2)*posit_direc(6,i))); %calculate angle of head direction relative to pre defined x-axis
% looking on the left side of box
if theta < 0 && Side(i) == 1
    V1(1) = P1(1) - posit_direc(3,i); % vector1 x coordinate
    V1(2) = P1(2) - posit_direc(4,i); % vector1 y coordinate
    V2(1) = P2(1) - posit_direc(3,i); % vector2 x coordinate
    V2(2) = P2(2) - posit_direc(4,i); % vector2 y coordinate
    V1_theta = rad2deg(atan2(xaxis(1)*V1(2)-V1(1)*xaxis(2),xaxis(1)*V1(1)+xaxis(2)*V1(2)));
    V2_theta = rad2deg(atan2(xaxis(1)*V2(2)-V2(1)*xaxis(2),xaxis(1)*V2(1)+xaxis(2)*V2(2)));
    if V1_theta < theta && theta < V2_theta || V2_theta < theta && theta < V1_theta % if head direction is between the two vectors, animal is looking at object
        FvB(i) = 1;
    else FvB(i) = 0; %facing, but not looking 
    end
%looking on the right side of box    
elseif theta < 0 && Side(i) == -1
    V3(1) = P3(1) - posit_direc(3,i); % vector1 x coordinate
    V3(2) = P3(2) - posit_direc(4,i); % vector1 y coordinate
    V4(1) = P4(1) - posit_direc(3,i); % vector2 x coordinate
    V4(2) = P4(2) - posit_direc(4,i); % vector2 y coordinate
    V3_theta = rad2deg(atan2(xaxis(1)*V3(2)-V3(1)*xaxis(2),xaxis(1)*V3(1)+xaxis(2)*V3(2)));
    V4_theta = rad2deg(atan2(xaxis(1)*V4(2)-V4(1)*xaxis(2),xaxis(1)*V4(1)+xaxis(2)*V4(2)));
    if V3_theta < theta && theta < V4_theta || V4_theta < theta && theta < V3_theta % if head direction is between the two vectors, animal is looking at object
        FvB(i) = 1;
    else FvB(i) = 0; % facing, but not looking
    end
elseif theta > 0 
    FvB(i) = -1; % Back to stimulus
end

end

head_direction_data = [FramNum; FvB; Side];

%FPS ~= 25 so each frame is ~1/25 of a second long

LookAt = (sum(FvB == 1)) / 25; % Time spent looking left in seconds
LookAw = (sum(FvB == -1)) / 25; % Time spent looking right in seconds
LookN = (sum(FvB == 0)) / 25; % Time lost to noise/ Not looking 
OnLeft = (sum(Side == 1)) / 25; % Time spent on left side of box in seconds
OnRight = (sum(Side == -1)) / 25; % Time spent on right side of box in seconds
Midline = (sum(Side == 0)) / 25; % Time spent on the midline of box in seconds
TotTime = length(FramNum)/(25);

OnLeft_Looking = 0; 
%OnLeft_LookingRight = 0;
OnRight_Looking = 0;
%OnRight_LookingRight = 0;
for n = 1:length(head_direction_data(1,:))
    if Side(n) == 1 && FvB(n) == 1
        OnLeft_Looking = OnLeft_Looking +1;
    %elseif Side(n) == 1 && FvB(n) == -1
    %   OnLeft_LookingRight = OnLeft_LookingRight +1;
    elseif Side(n) == -1 && FvB(n) == 1
        OnRight_Looking = OnRight_Looking +1;
    %elseif Side(n) == -1 && FvB(n) == -1
    %    OnRight_LookingRight = OnRight_LookingRight +1;
    end
end
OnLeft_Looking = OnLeft_Looking/25; % Time spent looking left on left side of box in seconds
%OnLeft_LookingRight = OnLeft_LookingRight/25; % Time spent looking left on right side of box in seconds
OnRight_Looking = OnRight_Looking/25; % Time spent looking right on left side of box in seconds
%OnRight_LookingRight = OnRight_LookingRight/25; % Time spent looking right on right side of box in seconds

FillinOutput = [OnLeft, OnLeft_Looking, NaN, OnRight, NaN, OnRight_Looking, NaN, NaN, TotTime]; 
copy(FillinOutput);
