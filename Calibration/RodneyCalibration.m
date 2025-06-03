
T = readtable("05_01_Testing_combined.xlsx", "Sheet", "Data");
Ax = T.Location == "Ax";
Ay = T.Location == "Ay";
Bx = T.Location == "Bx";
By = T.Location == "By";

x = [10 12 14 16 18 10 12 14 16 18 10 12 14 16 18 10 12 14 16 18 10 12 14 16 18]'; % cm
mass = [.25 .25 .25 .25 .25 0.5 0.5 0.5 0.5 0.5 1 1 1 1 1 1.5 1.5 1.5 1.5 1.5 2.0 2.0 2.0 2.0 2.0]'; % kg
F = mass*9.81; % N

cal1 = T.Mass_kg_== mass(1) & T.Distance_cm_ == x(1);
cal2 = T.Mass_kg_== mass(2) & T.Distance_cm_ == x(2);
cal3 = T.Mass_kg_== mass(3) & T.Distance_cm_ == x(3);
cal4 = T.Mass_kg_== mass(4) & T.Distance_cm_ == x(4);
cal5 = T.Mass_kg_== mass(5) & T.Distance_cm_ == x(5);
cal6 = T.Mass_kg_== mass(6) & T.Distance_cm_ == x(6);
cal7 = T.Mass_kg_== mass(7) & T.Distance_cm_ == x(7);
cal8 = T.Mass_kg_== mass(8) & T.Distance_cm_ == x(8);
cal9 = T.Mass_kg_== mass(9) & T.Distance_cm_ == x(9);
cal10 = T.Mass_kg_== mass(10) & T.Distance_cm_ == x(10);
cal11 = T.Mass_kg_== mass(11) & T.Distance_cm_ == x(11);
cal12 = T.Mass_kg_== mass(12) & T.Distance_cm_ == x(12);
cal13 = T.Mass_kg_== mass(13) & T.Distance_cm_ == x(13);
cal14 = T.Mass_kg_== mass(14) & T.Distance_cm_ == x(14);
cal15 = T.Mass_kg_== mass(15) & T.Distance_cm_ == x(15);
cal16 = T.Mass_kg_== mass(16) & T.Distance_cm_ == x(16);
cal17 = T.Mass_kg_== mass(17) & T.Distance_cm_ == x(17);
cal18 = T.Mass_kg_== mass(18) & T.Distance_cm_ == x(18);
cal19 = T.Mass_kg_== mass(19) & T.Distance_cm_ == x(19);
cal20 = T.Mass_kg_== mass(20) & T.Distance_cm_ == x(20);
cal21 = T.Mass_kg_== mass(21) & T.Distance_cm_ == x(21);
cal22 = T.Mass_kg_== mass(22) & T.Distance_cm_ == x(22);
cal23 = T.Mass_kg_== mass(23) & T.Distance_cm_ == x(23);
cal24 = T.Mass_kg_== mass(24) & T.Distance_cm_ == x(24);
cal25 = T.Mass_kg_== mass(25) & T.Distance_cm_ == x(25);


% Half Bridges
V_Ax_cal = [T.SampleAverage(Ax & cal1) T.SampleAverage(Ax & cal2) T.SampleAverage(Ax & cal3) T.SampleAverage(Ax & cal4) T.SampleAverage(Ax & cal5) T.SampleAverage(Ax & cal6) T.SampleAverage(Ax & cal7) T.SampleAverage(Ax & cal8) T.SampleAverage(Ax & cal9) T.SampleAverage(Ax & cal10) T.SampleAverage(Ax & cal11) T.SampleAverage(Ax & cal12) T.SampleAverage(Ax & cal13) T.SampleAverage(Ax & cal14) T.SampleAverage(Ax & cal15) T.SampleAverage(Ax & cal16) T.SampleAverage(Ax & cal17) T.SampleAverage(Ax & cal18) T.SampleAverage(Ax & cal19) T.SampleAverage(Ax & cal20) T.SampleAverage(Ax & cal21) T.SampleAverage(Ax & cal22) T.SampleAverage(Ax & cal23) T.SampleAverage(Ax & cal24) T.SampleAverage(Ax & cal25)]';
V_Bx_cal = [T.SampleAverage(Bx & cal1) T.SampleAverage(Bx & cal2) T.SampleAverage(Bx & cal3) T.SampleAverage(Bx & cal4) T.SampleAverage(Bx & cal5) T.SampleAverage(Bx & cal6) T.SampleAverage(Bx & cal7) T.SampleAverage(Bx & cal8) T.SampleAverage(Bx & cal9) T.SampleAverage(Bx & cal10) T.SampleAverage(Bx & cal11) T.SampleAverage(Bx & cal12) T.SampleAverage(Bx & cal13) T.SampleAverage(Bx & cal14) T.SampleAverage(Bx & cal15) T.SampleAverage(Bx & cal16) T.SampleAverage(Bx & cal17) T.SampleAverage(Bx & cal18) T.SampleAverage(Bx & cal19) T.SampleAverage(Bx & cal20) T.SampleAverage(Bx & cal21) T.SampleAverage(Bx & cal22) T.SampleAverage(Bx & cal23) T.SampleAverage(Bx & cal24) T.SampleAverage(Bx & cal25)]';
V_Ay_cal = [T.SampleAverage(Ay & cal1) T.SampleAverage(Ay & cal2) T.SampleAverage(Ay & cal3) T.SampleAverage(Ay & cal4) T.SampleAverage(Ay & cal5) T.SampleAverage(Ay & cal6) T.SampleAverage(Ay & cal7) T.SampleAverage(Ay & cal8) T.SampleAverage(Ay & cal9) T.SampleAverage(Ay & cal10) T.SampleAverage(Ay & cal11) T.SampleAverage(Ay & cal12) T.SampleAverage(Ay & cal13) T.SampleAverage(Ay & cal14) T.SampleAverage(Ay & cal15) T.SampleAverage(Ay & cal16) T.SampleAverage(Ay & cal17) T.SampleAverage(Ay & cal18) T.SampleAverage(Ay & cal19) T.SampleAverage(Ay & cal20) T.SampleAverage(Ay & cal21) T.SampleAverage(Ay & cal22) T.SampleAverage(Ay & cal23) T.SampleAverage(Ay & cal24) T.SampleAverage(Ay & cal25)]';
V_By_cal = [T.SampleAverage(By & cal1) T.SampleAverage(By & cal2) T.SampleAverage(By & cal3) T.SampleAverage(By & cal4) T.SampleAverage(By & cal5) T.SampleAverage(By & cal6) T.SampleAverage(By & cal7) T.SampleAverage(By & cal8) T.SampleAverage(By & cal9) T.SampleAverage(By & cal10) T.SampleAverage(By & cal11) T.SampleAverage(By & cal12) T.SampleAverage(By & cal13) T.SampleAverage(By & cal14) T.SampleAverage(By & cal15) T.SampleAverage(By & cal16) T.SampleAverage(By & cal17) T.SampleAverage(By & cal18) T.SampleAverage(By & cal19) T.SampleAverage(By & cal20) T.SampleAverage(By & cal21) T.SampleAverage(By & cal22) T.SampleAverage(By & cal23) T.SampleAverage(By & cal24) T.SampleAverage(By & cal25)]';

%V1_cal = [T.SampleAverage(green & cal1) T.SampleAverage(green & cal2) T.SampleAverage(green & cal3) T.SampleAverage(green & cal4) T.SampleAverage(green & cal5) T.SampleAverage(green & cal6) T.SampleAverage(green & cal7) T.SampleAverage(green & cal8) T.SampleAverage(green & cal9) T.SampleAverage(green & cal10) T.SampleAverage(green & cal11) T.SampleAverage(green & cal12) T.SampleAverage(green & cal13) T.SampleAverage(green & cal14) T.SampleAverage(green & cal15) T.SampleAverage(green & cal16) T.SampleAverage(green & cal17) T.SampleAverage(green & cal18) T.SampleAverage(green & cal19) T.SampleAverage(green & cal20) T.SampleAverage(green & cal21) T.SampleAverage(green & cal22) T.SampleAverage(green & cal23) T.SampleAverage(green & cal24) T.SampleAverage(green & cal25) T.SampleAverage(green & cal26) T.SampleAverage(green & cal27) T.SampleAverage(green & cal28)]';
%V2_cal = [T.SampleAverage(yellow & cal1) T.SampleAverage(yellow & cal2) T.SampleAverage(yellow & cal3) T.SampleAverage(yellow & cal4) T.SampleAverage(yellow & cal5) T.SampleAverage(yellow & cal6) T.SampleAverage(yellow & cal7) T.SampleAverage(yellow & cal8) T.SampleAverage(yellow & cal9) T.SampleAverage(yellow & cal10) T.SampleAverage(yellow & cal11) T.SampleAverage(yellow & cal12) T.SampleAverage(yellow & cal13) T.SampleAverage(yellow & cal14) T.SampleAverage(yellow & cal15) T.SampleAverage(yellow & cal16) T.SampleAverage(yellow & cal17) T.SampleAverage(yellow & cal18) T.SampleAverage(yellow & cal19) T.SampleAverage(yellow & cal20) T.SampleAverage(yellow & cal21) T.SampleAverage(yellow & cal22) T.SampleAverage(yellow & cal23) T.SampleAverage(yellow & cal24) T.SampleAverage(yellow & cal25) T.SampleAverage(yellow & cal26) T.SampleAverage(yellow & cal27) T.SampleAverage(yellow & cal28)]';

%--------------- Calibrate via Regression --------------
% V = kF(x-d)+c = kFx - kdF + c
x = x./100;
A = [F.*x -F];

% Bridge Ax

lm_Ax = fitlm(A, V_Ax_cal);
c_Ax = lm_Ax.Coefficients.Estimate(1);
k_Ax = lm_Ax.Coefficients.Estimate(2);
d_Ax = lm_Ax.Coefficients.Estimate(3)/k_Ax;

% Bridge Bx

lm_Bx = fitlm(A, V_Bx_cal);
c_Bx = lm_Bx.Coefficients.Estimate(1);
k_Bx = lm_Bx.Coefficients.Estimate(2);
d_Bx = lm_Bx.Coefficients.Estimate(3)/k_Bx;

% Bridge Ay

lm_Ay = fitlm(A, V_Ay_cal);
c_Ay = lm_Ay.Coefficients.Estimate(1);
k_Ay = lm_Ay.Coefficients.Estimate(2);
d_Ay = lm_Ay.Coefficients.Estimate(3)/k_Ay;

% Bridge By

lm_By = fitlm(A, V_By_cal);
c_By = lm_By.Coefficients.Estimate(1);
k_By = lm_By.Coefficients.Estimate(2);
d_By = lm_By.Coefficients.Estimate(3)/k_By;


% ------------ Save Calibration Data -------------
% save CalibrationData_round3_2.mat
save CalibrationData_Rodney.mat

% ---------- Compare Regression Values -----------
r2_cal_1 = [lm_Ax.Rsquared.Ordinary lm_Ay.Rsquared.Ordinary];
r2_cal_2 = [lm_Bx.Rsquared.Ordinary lm_By.Rsquared.Ordinary];
r2_cal_array = [r2_cal_1; r2_cal_2];

lm_Ax.Rsquared.Ordinary
lm_Ay.Rsquared.Ordinary
lm_Bx.Rsquared.Ordinary
lm_By.Rsquared.Ordinary

rows = {'1','2'};
cols = {'x', 'y'};
r2_cal_table = array2table(r2_cal_array, 'RowNames', rows, 'VariableNames', cols);

imagesc(r2_cal_array); %
colormap(summer)
colorbar
textStrings = num2str(r2_cal_array(:), '%0.8f');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x_grid, y_grid] = meshgrid(1:2, 1:2);

text(x_grid(:), y_grid(:), textStrings(:),'HorizontalAlignment', 'center'); % Plot the strings
title('R^2 Values for Calibration')
xticklabels({' ', 'X', ' ', 'Y'})
yticks([1,2])
yticklabels({'A','B'})
ylabel('Bridge Location')
xlabel('Bridge Configuration')
saveas(gcf,'rodney_r2values.png')
