% =========================================================================
% PURPOSE: This script inserts calibration data into the data table.
%
% STEP 1: Load PVCtests data table (not scripted here).
% STEP 2: use TABLE = sortrows(TABLE, COLUMNS); to sort data by:
%             1 - configuration number
%             2 - filename
%             3 - stalk number
% STEP 3: Run this script
% =========================================================================


PVCtests = sortrows(PVCtests,"Filename2","ascend");
N = height(PVCtests);
P1preOffset = 5; 

% ====================== CALIBRATION COEFFICIENTS =========================
% The coefficients below were used from 06/17/24 to 4/1/25 (Configs 1-3)
kAx = 0.1265;
kAy = 0.109;
kBx = 0.1227;
kBy = 0.0997;

dAx = 0.0634;
dAy = 0.0157;
dBx = 0.015;
dBy = 0.0647;

K = [kAx, kAy, kBx, kBy];
D = [dAx, dAy, dBx, dBy];
% =========================================================================


% ===== CREATE CALIBRATION VARIABLES IN THE TABLE (if needed) =============
VarNames = PVCtests.Properties.VariableNames;
if ~ismember('K', VarNames) || ~ismember('D', VarNames) || ~ismember('C', VarNames)
    temp = single(nan(N,4));
    PVCtests.K = temp;
    PVCtests.D = temp;
    PVCtests.C = temp;
end
% =========================================================================

    
% ============== LOOP TO CALCULATE TEST-SPECIFIC "C" values ===============
for i = 1:1000
    % ---------- PULL OUT DATA FOR EASIER ACCESS --------------------------
    StrainAx = PVCtests.StrainAx{i};
    StrainBx = PVCtests.StrainBx{i};
    StrainAy = PVCtests.StrainAy{i};
    StrainBy = PVCtests.StrainBy{i};
   
    % Create Pi variables from table data:
    for j = 1:5
        eval(['P',num2str(j), ' = PVCtests.P',num2str(j),'(i);'])
    end

    % =========== COMPUTE "C" CALIBRATION VALUES ======================
    cAx = nanmean(StrainAx(1:P1-P1preOffset));     % Take the first several data points to set c values
    cBx = nanmean(StrainBx(1:P1-P1preOffset));
    cAy = nanmean(StrainAy(1:P1-P1preOffset));
    cBy = nanmean(StrainBy(1:P1-P1preOffset));
    C = [cAx, cAy, cBx, cBy];
   
    % STORE CALIBRATION VALUES:
    PVCtests.K(i,:) = K;
    PVCtests.D(i,:) = D;
    PVCtests.C(i,:) = C;
    % =================================================================

end
% ================== CALCULATION LOOP =====================================


% ====================== CALIBRATION COEFFICIENTS =========================
% The coefficients below were used from 04/29/25 - ??? (Configs 4-??)
kAx = 0.0615;
kAy = 0.0615;
kBx = 0.0610;
kBy = 0.0619;

dAx = 0.0370;
dAy = 0.0366;
dBx = 0.0893;
dBy = 0.0900;

K = [kAx, kAy, kBx, kBy];
D = [dAx, dAy, dBx, dBy];
% =========================================================================


for i = 1000:N
    % ---------- PULL OUT DATA FOR EASIER ACCESS --------------------------
    StrainAx = PVCtests.StrainAx{i};
    StrainBx = PVCtests.StrainBx{i};
    StrainAy = PVCtests.StrainAy{i};
    StrainBy = PVCtests.StrainBy{i};
   
    % Create Pi variables from table data:
    for j = 1:5
        eval(['P',num2str(j), ' = PVCtests.P',num2str(j),'(i);'])
    end

    % =========== COMPUTE "C" CALIBRATION VALUES ======================
    cAx = nanmean(StrainAx(1:P1-P1preOffset));     % Take the first several data points to set c values
    cBx = nanmean(StrainBx(1:P1-P1preOffset));
    cAy = nanmean(StrainAy(1:P1-P1preOffset));
    cBy = nanmean(StrainBy(1:P1-P1preOffset));
    C = [cAx, cAy, cBx, cBy];
   
    % STORE CALIBRATION VALUES:
    PVCtests.K(i,:) = K;
    PVCtests.D(i,:) = D;
    PVCtests.C(i,:) = C;
    % =================================================================
end