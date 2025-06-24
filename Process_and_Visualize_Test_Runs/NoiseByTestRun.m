% =========================================================================
% PURPOSE: This script calculates the average noise in a single test run
% and generates a plot of noise as a function of test run. Noise is
% quantified by the mean absolute error from the median value for each
% individual stalk.
%
% PREREQUISITES:
%   1 - "renamingFilenames.m" should be run before this script is run.
%   2 - must have "MedEI.mat" in the folder to run this script
%
%
% NOTES: This approach is a bit dangerous as it must choose some reference
% values from which to calculate the noise.
% =========================================================================

load MedEI.mat                                  % load reference data set

testRuns = unique(PVCtests.Filename2);          % get all the unique test runs (requires that Filename2 variable exists in PCVtests - see "renamingFilenames.m")
Low = categorical("Low");
Medium = categorical("Medium");


for i = 1:length(testRuns)                               % Loop through all the test runs

    index = ismember(PVCtests.Filename2, testRuns(i));   % find all data matching this test run
    PVCtests.TestRun(index) = i;                         % Add TestRun variable to PVCtests   
    testType = PVCtests.PVC(index);                      % get the testType (Med or Low)   
    
    % --------------- CALCULATE NOISE VALUES ------------------------------
    if testType(1) == Medium
        noise(i) = nanmean(abs(PVCtests.EI(index) - MedEI.MedStalksEI))/nanmean(MedEI.MedStalksEI)*100;
    elseif testType(1) == Low && i <=171
        noise(i)  = nanmean(abs(PVCtests.EI(index) - MedEI.LowStalksEI))/nanmean(MedEI.LowStalksEI)*100;
    elseif testType(1) == Low && i > 171
        LowEI= MedEI.LowStalksEI;
        medianLowEI(2) = NaN;
        noise(i)  = nanmean(abs(PVCtests.EI(index) - LowEI))/nanmean(LowEI)*100;
    end
    % ---------------------------------------------------------------------

    yawVals = PVCtests.Yaw(index);                      % Get the Yaw, Offset, and Config values
    offsetVals = PVCtests.Offset(index);
    configVals = PVCtests.RodneyConfig(index);
    
    Yaw(i) = yawVals(1);                                % Get the Yaw, Offset, and Config values
    StalkType(i) = testType(1);
    Offset(i) = offsetVals(1);
    Config(i) = configVals(1);


end
    
% ============== CREATE DATA TABLE of RUN DATA ============================
Yaw = Yaw(:);
Offset = Offset(:);
StalkType = StalkType(:);
Offset = Offset(:);
Config = Config(:);
noise = noise(:);
RunData = table(testRuns, Config, StalkType, Yaw, Offset, noise);
% =========================================================================



% ==================== PLOTTING ===========================================
xlimits = [40 200];             % sets x-limits for all charts
% -------------------------------------------------------------------------
figure('Units','normalized','Position',[0 0 1 1]);
tiledlayout(6,1)
% -------------------------------------------------------------------------

nexttile([1,1]) % ---------------------------------------------------------
plot(RunData.StalkType == categorical("Medium"),'LineWidth',3)
ylim([-0.5 1.5])
title('Stalk Stiffness (Med/Low)')
xlim(xlimits)

nexttile([3,1])% ---------------------------------------------------------
plot(RunData.noise,'.k','MarkerFaceColor','k')
xline([74.5, 82.5, 111.5 141.5, 171.5 174.5])
hold on
% patch([0 99 99 0], [0 0 200 200],'b','FaceAlpha', 0.05,'EdgeColor','none')
% plot([1 99], [1 1 ]*10, 'k','LineWidth',3)
ylim([0 80])
text(50,50,'Config 1', 'FontSize',14)           % Original from capstone and 2024 field tests - rickety mounting, bad guide
text(75.5,50,'Config 2', 'FontSize',14)         % Straight section added
text(92,50,'Config 3', 'FontSize',14)           % Straight section improved
text(120,50,'Config 4', 'FontSize',14)          % New housing, flimsy guide
text(155,50,'Config 5', 'FontSize',14)          % Sturdyier guide
text(172,50,'Config 6')                         % Sanded the edges of the guide
text(175,45,'Config 7')                         % Christian ground the edges and profile of the stiffer guide
yline(10,'Color',[1 1 1]*0.75)
ylabel('Run Noise Level (%)', 'FontSize',14)
xlim(xlimits)

nexttile([1,1])% ---------------------------------------------------------
yline([10:10:30],'Color',[1 1 1]*0.75)
hold on
plot(RunData.Yaw,'LineWidth',3)
ylim([0 35])
ylabel('Yaw Angle', 'FontSize',14)
xlim(xlimits)

nexttile([1,1])% ---------------------------------------------------------
plot(RunData.Offset,'LineWidth',3)
ylim([15 35])
ylabel('Offset')
xlabel('Run Number', 'FontSize',14)
xlim(xlimits)
% =========================================================================
