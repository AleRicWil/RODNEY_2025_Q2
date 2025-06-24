% =========================================================================
% PURPOSE: This script is designed to be run by Maeson after performing
% each set of tests. It calculates EI values and plots noise as a function
% of test run.
%
% FILE PREREQUISITE - Must have the following files in the folder:
%   1 - renamingFilenames.m         
%   2 - InsertCalibrationData.m     
%   3 - ParameterizedEIcalcs.m 
%   4 - NoiseByTestRun.m 
%   5 - CookEIcalcs.m
%   6 - NoiseByTestRun.m 
%   7 - CookEIcalcs.m
%   8 - Strains2ForcesLocs.m 
%   9 - MedEI.mat  
% =========================================================================


[file, location] = uigetfile('*.mat','Choose MASTER TABLE to process');
file = fullfile(location, file);
load(file);


% ========== RENAME FILES AND INSERT CALIBRATION DATA =====================
renamingFilenames               % Renames all files using the Filename2 format with no single digit test runs
InsertCalibrationData           % inserts calibration data for each test
% =========================================================================


% ===== CREATE EI VARIABLES IN THE TABLE (if needed) =============
N = height(PVCtests);
VarNames = PVCtests.Properties.VariableNames;
if ~ismember('EI', VarNames) || ~ismember('D', VarNames) || ~ismember('C', VarNames)
    temp = single(nan(N,1));
    PVCtests.EI = temp;
end
% =========================================================================


% ================ PROCESSING OPTIONS =====================================
ProcParam = struct();                           % create empty structure

ProcParam.RegionPoints = [2, 3];                % Default: [2,3].  These are the "pick" points used in the clicking routing (P1, P2, P3, etc.). But just enter the numbers, not the "P"s.
ProcParam.RegionPointOffsets = [0, 0];          % Default: [0,0].  Offsets from the pick points, if needed. Use negative numbers for a backwards offset
ProcParam.RegionIndices = [nan, nan];           % Default: [0,0].  These are not inputs, but upon completion, they will contain the starting and ending indices for the selected regions. 

ProcParam.StrainSmoothingSpan = 30;             % Default: [30].    Span to be used for smoothing the strain
ProcParam.StrainSmoothingMethod = 'loess';      % Default: 'loess'. Method to use for smoothing
ProcParam.StrainSmoothingReps = 2;              % Default: [2].     Number of smoothing iterations 

ProcParam.ForcePositionSmoothingSpan = 30;         % Default: [30].    Span to be used for smoothing the force and position data
ProcParam.ForcePositionSmoothingMethod = 'loess';  % Default: 'loess'. Method to use for smoothing
ProcParam.ForcePositionSmoothingReps = 2;          % Default: [2].     Number of smoothing iterations 
% =========================================================================


% ================ CALCULATE EI VALUES ====================================
[PVCtests] = ParameterizedEIcalcs(PVCtests,ProcParam);      % calculate all the EI values
% =========================================================================


% ========= PLOT NOISE AS A FUNCTION OF TEST NUMBER =======================
NoiseByTestRun                                              % Creates a figure showing noise by test run.
% =========================================================================


% ========= SAVE THE LATEST MASTER TABLE FILE =============================
oldFolder = cd(location);
SaveAppendFilename = ['PVC_MASTER_TABLE_', char(datetime(),'yyyy-MM-dd_HH.mm.ss'), '.mat'];
save(SaveAppendFilename, 'PVCtests')
cd(oldFolder);
% =========================================================================

