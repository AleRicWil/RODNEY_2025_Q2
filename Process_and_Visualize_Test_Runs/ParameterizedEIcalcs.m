function [PVCtests] = ParameterizedEIcalcs(PVCtests,ProcParam)
% =========================================================================
% PURPOSE: calculates EI values from PVCtests data table. Stores the results
% as well as the processing options used (useful for not losing track of
% processing options).
%
% INPUTS:   PVCtests - table of PVCtest data (MUST contain calibration data!)
%           ProcParam - structure of all processing options (see example below)
% 
%                    ProcParam has the following structure:
%                    -----------------------------------------------------  
%                    ProcParam = struct();                           % create empty structure
%                    ---------------------------------------------------------------------------
%                    ProcParam.RegionPoints = [2, 3];                % Default: [2,3].  These are the "pick" points used in the clicking routing (P1, P2, P3, etc.). But just enter the numbers, not the "P"s.
%                    ProcParam.RegionPointOffsets = [0, 0];          % Default: [0,0].  Offsets from the pick points, if needed. Use negative numbers for a backwards offset
%                    ProcParam.RegionIndices = [nan, nan];           % Default: [0,0].  These are not inputs, but upon completion, they will contain the starting and ending indices for the selected regions. 
%                    ---------------------------------------------------------------------------
%                    ProcParam.StrainSmoothingSpan = 30;             % Default: [30].    Span to be used for smoothing the strain
%                    ProcParam.StrainSmoothingMethod = 'loess';      % Default: 'loess'. Method to use for smoothing
%                    ProcParam.StrainSmoothingReps = 2;              % Default: [2].     Number of smoothing iterations 
%                    ---------------------------------------------------------------------------
%                    ProcParam.ForcePositionSmoothingSpan = 30;         % Default: [30].    Span to be used for smoothing the force and position data
%                    ProcParam.ForcePositionSmoothingMethod = 'loess';  % Default: 'loess'. Method to use for smoothing
%                    ProcParam.ForcePositionSmoothingReps = 2;          % Default: [2].     Number of smoothing iterations 
%                    ---------------------------------------------------------------------------
%
% OUTPUTS:  updated PVCtests data table
%
% NOTES:
%   1 - This function assumes that PVCtests already contain calibration data.
%   2 - Processes ALL of the tests using the same processing options. 
% 
% =========================================================================


N = height(PVCtests);

% If PVCtests does not yet contain a ProcParam variable, add it
if ~ismember('ProcParam', PVCtests.Properties.VariableNames)
    PVCtests.ProcParam = cell(N,1);
end

% Wipe all previous results and/or define variables if needed.
PVCtests.EI = nan(N,1);
PVCtests.StrainAxSm = cell(N,1);
PVCtests.StrainAySm = cell(N,1);
PVCtests.StrainBxSm = cell(N,1);
PVCtests.StrainBySm = cell(N,1);
PVCtests.FxSm = cell(N,1);
PVCtests.FySm = cell(N,1);
PVCtests.FSm = cell(N,1);
PVCtests.ZSm = cell(N,1);
PVCtests.ZxSm = cell(N,1);
PVCtests.ZySm = cell(N,1);
PVCtests.ThetaSm = cell(N,1);
PVCtests.deflection = cell(N,1);
PVCtests.FitCoeffs = nan(N,2);

% =================== Loop through all tests ==============================
for i = 1:N

    % ---------- PULL OUT ALL DATA FOR EASIER ACCESS ----------------------
    StrainAx = PVCtests.StrainAx{i};
    StrainBx = PVCtests.StrainBx{i};
    StrainAy = PVCtests.StrainAy{i};
    StrainBy = PVCtests.StrainBy{i};
    K = PVCtests.K(i,:);
    D = PVCtests.D(i,:);
    C = PVCtests.C(i,:);
    % ---------------------------------------------------------------------

    % =================== DEFINE REGION TO USE ============================
    for j = 1:2
        k = ProcParam.RegionPoints(j);
        eval(['R',num2str(j), ' = PVCtests.P',num2str(k),'(i);'])       % this forms an R1 and R2 variables based on the "points" indicated in "RegionPoints".
    end
    
    R1offset = ProcParam.RegionPointOffsets(1);     % Add offset values
    R2offset = ProcParam.RegionPointOffsets(2);

    region = (R1+R1offset):(R2 + R2offset);         % the actual indexed region to analyze
    ProcParam.RegionIndices = [R1, R2];             % store the indexed region beginning and end points
    % =================================================================

    % ============= EXTRACT THE TEST DATA TO ANALYZE =====================
    Time = PVCtests.Time{i}(region);
    StrainAx = PVCtests.StrainAx{i}(region);
    StrainBx = PVCtests.StrainBx{i}(region);
    StrainAy = PVCtests.StrainAy{i}(region);
    StrainBy = PVCtests.StrainBy{i}(region);
    h = PVCtests.Height(i)/100;
    % =====================================================================

    % ============ SMOOTH THE STRAIN DATA =================================
    span = ProcParam.StrainSmoothingSpan;
    method = ProcParam.StrainSmoothingMethod;

    for j = 1:ProcParam.StrainSmoothingReps
        StrainAxSm = smooth(StrainAx, span, method);
        StrainBxSm = smooth(StrainBx, span, method);
        StrainAySm = smooth(StrainAy, span, method);
        StrainBySm = smooth(StrainBy, span, method);
    end
    % =====================================================================

    % ============ CALCULATE FORCES AND POSITIONS =========================
    [FxSm, FySm, FSm, ZxSm, ZySm, ZSm] = Strains2ForcesLocs(K,D,C,StrainAxSm, StrainBxSm, StrainAySm, StrainBySm);
    % =====================================================================
   
    % ============ SMOOTH FORCE/POSITION DATA =============================
    % NOTE: THIS SMOOTHING STEP MAY NOT HAVE MUCH INFLUENCE ON THE RESULTS       
    span = ProcParam.ForcePositionSmoothingSpan;
    method = ProcParam.ForcePositionSmoothingMethod;

    for j = 1:ProcParam.ForcePositionSmoothingReps
        FxSm2 = smooth(FxSm, span, method);
        FySm2 = smooth(FySm, span, method);
        ZxSm2 = smooth(ZxSm, span, method);
        ZySm2 = smooth(ZySm, span, method);
    end
    % =====================================================================

    % ============ CALCULATE EI =======================================
    [EIc, FdefRsquared, deflection, ~, FitCoeffs] = CookEIcalcs(FxSm2, FySm2, ZxSm2, ZySm2, PVCtests.Yaw(i), h);
    % =================================================================
      
    % ============ STORE DATA INTO TABLE ==================================
    PVCtests.EI(i) = EIc;                       % store results 
    PVCtests.FdefRsquared(i) = FdefRsquared;
    PVCtests.deflection{i} = deflection;
    PVCtests.FitCoeffs(i,:) = FitCoeffs';

    PVCtests.ProcParam{i} = ProcParam;          % store the Processing Parameters just used
    
    PVCtests.StrainAxSm{i} = StrainAxSm;        % store the individual data segments used
    PVCtests.StrainAySm{i} = StrainAySm;
    PVCtests.StrainBxSm{i} = StrainBxSm;
    PVCtests.StrainBySm{i} = StrainBySm;
    PVCtests.FxSm{i} = FxSm;
    PVCtests.FySm{i} = FySm;
    PVCtests.FSm{i} = FSm;
    PVCtests.ZSm{i} = ZSm;
    PVCtests.ZxSm{i} = ZxSm;
    PVCtests.ZySm{i} = ZySm;
    PVCtests.ThetaSm{i} =  atan2(FySm,FxSm);
    % =================================================================

end
