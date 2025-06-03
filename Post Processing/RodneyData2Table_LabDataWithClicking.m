% function [DataByStalk] = RodneyRunAndTimestamp2Table_LabData(dataFileFullPath)

% =========================================================================
% This function takes in two data files and produces a single data table called DataByStalk
% where each row is an individual stalk. The two files are:
%       - data_file (contains raw test data)
%       - timestamp_file (contains timestamps that are used to separate individual stalks
%
% Key variables include the following:
%       - RunData (a table containing the raw test data for a single test run)
%       - TimeStamps (a table containing the information on timestamps identified by humans)
%       - IndexMatrix (a matrix that contains the indices used to separate the stalks)
%       - SingleStalk (a table containing data from a single stalk)
%       - DataByStalk (a table containing all the data from "data_file", but now each row is an individual stalk
%   
% =========================================================================


% ============ STEP 0 - USER INPUT AND SETUP ==============================
Nstalks = 9;

% PROMPT FOR FILE OR FOLDER -----------------------------------------------
answer = questdlg('Processing a single file or all files from a folder?', ...
	'Processing Options', ...
	'Single File','Folder','Cancel','Folder');

switch answer
    case 'Single File'
        [file,folder] = uigetfile('.csv');
        cd(folder)
        files = dir(file);
    case 'Folder'
        folder = uigetdir();
        cd(folder)
        files = dir('*.csv');
       
    case 'Cancel'
        exit
end
%--------------------------------------------------------------------------

% OPERATOR & CLICKER ------------------------------------------------------
operator = questdlg('Who ran these tests?', ...
	'Test Operator', ...
	'CS','MH','DC','MH');

clicker = questdlg('Who is processing the data?', ...
	'Test Processor', ...
	'CS','MH','DC','MH');
%--------------------------------------------------------------------------

% PROMPT FOR SESSION FILENAME ---------------------------------------------
prompt = {'Filename for saving session progress:'};
dlgtitle = 'Session Filename';
fieldsize = [1 60];
definput = {files(1).name};
SessionFilename = inputdlg(prompt,dlgtitle,fieldsize,definput);
SessionFilename = [SessionFilename{1,1}, '.mat'];
%--------------------------------------------------------------------------


% CREATE EMPTY TABLE FOR STORING DATA WHICH HAS ONE ROW OF DATA FOR EACH STALK AND INCLUDES ALL THE DATA FOR THE ENTIRE TEST RUN 
mkdir('ProcessedFiles'),
VariableNames = {'Note', 'RodneyConfig', 'PVC', 'Height', 'Yaw', 'Pitch', 'Roll', 'RateOfTravel', 'AngleOfTravel', 'Offset', 'Stalk', 'Time', 'StrainAx', 'StrainBx', 'StrainAy', 'StrainBy', 'DataStartOG', 'P1', 'P2', 'P3', 'P4', 'P5', 'DataEndOG', 'Operator', 'Clicker', 'Filename', 'Directory'};
C = cell(1,1);
Cat = categorical({''});
SingleTestRun = table(C, Cat, Cat, 0, 0, 0, 0, 0, 0, 0, 0, C, C, C, C, C, 0, 0, 0, 0, 0, 0, 0, C, C, C, C, 'VariableNames',VariableNames);  % 
MultiTestRun = table(C, Cat, Cat, 0, 0, 0, 0, 0, 0, 0, 0, C, C, C, C, C, 0, 0, 0, 0, 0, 0, 0, C, C, C, C, 'VariableNames',VariableNames);  
    
% TURN OFF WARNINGS THAT DO NOT AFFECT PERFORMANCE
VariableNameWarning = 'MATLAB:table:ModifiedAndSavedVarnames';      % WARNING MESSAGE:  'Column headers from the file were modified to make them valid MATLAB identifiers before creating variable names for the table. The original column headers are saved in the VariableDescriptions property. Set 'VariableNamingRule' to 'preserve' to use the original column headers as table variable names.'
warning('off', VariableNameWarning);
AddingRowsWarning = 'MATLAB:table:RowsAddedExistingVars';           % WARNING MESSAGE    'The assignment added rows to the table, but did not assign values to all of the table's existing variables. Those variables are extended with rows containing default values.'   
warning('off', AddingRowsWarning);
% =========================================================================


% =================== LOOP THROUGH FILE(S) ================================
for j = 1:length(files)
    
    k = 1;
    SingleTestRun = table(C, Cat, Cat, 0, 0, 0, 0, 0, 0, 0, 0, C, C, C, C, C, 0, 0, 0, 0, 0, 0, 0, C, C, C, C, 'VariableNames',VariableNames);  % 

    % ============ STEP 1 - READ IN THE DATA FILE==========================
    % Note - MATLAB doesn't allow spaces in table column names. It converts any
    % spaces to underscores and generates warnings when it does this. The
    % warnings are harmless and can be ignored.
    
    % IMPORT OPTIONS:
    %opts = detectImportOptions(dataFileFullPath);                   % create a MATLAB import options object
    %opts.VariableTypes = {'single','single','single', 'single','single','char'};     % read in data as single-precision instead of double-precision (makes the data file half the size and the extra precision is not useful)
    % Note: apparently MATLAB doesn't allow import "opts" with the
    % "NumHeaderLines" option, and "opts" doesn't have a "NumHeaderLines"
    % options, so these options are incompatible. 
    filename = files(j).name;    
    RawDataTable = readtable(filename, 'NumHeaderLines', 11);    % read the data file in as a table
    RawDataTable = removevars(RawDataTable, "CurrentTime");                   % remove the current time variable
    RawDataTable.Properties.VariableNames(1) = "Time";                   % rename "Time_Microseconds" to just "Time" because they don't look like microseconds
    RawDataTable.Time = single(RawDataTable.Time);
    RawDataTable.StrainAx = single(RawDataTable.StrainAx);
    RawDataTable.StrainBx = single(RawDataTable.StrainBx);
    RawDataTable.StrainAy = single(RawDataTable.StrainAy);
    RawDataTable.StrainBy = single(RawDataTable.StrainBy);
    
    HeaderData = readcell(filename);          % Read the file as a cell
    % NOTE: This is extremely inefficient, but MATLAB's 'Range' option never works so instead of reading just part of the file, we have to read in the ENTIRE file. 
    HeaderData = HeaderData(1:10,:);
    [hh, ww] = size(HeaderData);
    for h = 1:hh, for w = 1:ww
            if ismissing(HeaderData{h,w}), HeaderData{h,w} = []; end
    end, end
    % =====================================================================
    
    
    % ============ STEP 2 - PREP DATA FOR VISUALIZATION ===================
    nbegin = 100;                               % number of beginning points to sample
    C1 = nanmean(RawDataTable{1:nbegin,2:5});   % mean value of beginning sample    
    S1 = nanstd(RawDataTable{1:nbegin,2:5});    % stdev of beginning sample
    
    dat = RawDataTable{:,2:5};          % data matrix for easy access
    index = dat < C1 + 4*S1;            % identify the "no stalk signal" regions
    dat(index) = nan;                   % turn those to nans
    C2 = nanmean(dat);                  % the mean of the stalk signal data
    S2 = nanstd(dat);                   % and stdev
    
    dat = RawDataTable{:,2:5};          % reset the data matrix
    dat = (dat - C1)./(C2 + 3*S2);      % scaling the data from 0 to 1 for better plotting
    dat = dat + [0 1 2 3];              % shifting each signal upward for better visualization
    % =====================================================================


    % ============ STEP 3 - CLICK ON CUT POINTS BETWEEN TESTS =============
    
    % FIRST CUT OUT DATA AT THE BEGINNING AND END -------------------------
    check = 0; 
    while check == 0                % While loop to allow fixing clicks
    
        % PLOTTING 
        figure('units','normalized','outerposition',[0 0 1 1])
        plot(dat)
        h1 = text(500, -0.25, 'Click Twice to Trim the ends of the data.','FontSize',20);
        [x, ~] = ginput(2);
        x = uint16(x);
        Dstart = x(1);  % where the first stalk starts
        Dend = x(2);    % where the data ends 
        
        % CONFIRMATION STEP
        xline(x)
        delete(h1)
        xlimits = xlim();
        text(xlimits(1) + 100, -0.25, 'Click white to continue, gray to repeat.','FontSize',20)
        [x, y] = ginput(1);
        ylimits = ylim(); 
        xlimits = xlim();
        if x > xlimits(1) && x < xlimits(2) && y > ylimits(1) && y < ylimits(2)
            check = 1;
        else
            clf
            text(500, 1.5, 'Repeating....','FontSize',20)
            pause(0.5)
        end
        clf
    end
    %----------------------------------------------------------------------

    % CUT BETWEEN STALKS --------------------------------------------------
    check = 0;
    while check == 0            % While loop to allow fixing clicks

        % PLOTTING 
        index = Dstart:Dend;
        plot(index, dat(index,:))
        xlimits = ylim();
        ylim([-0.5, ylimits(2)])
        h1 = text(500, -0.25, 'Click 8 times to cut between stalks.','FontSize',20);
        [x, ~] = ginput(Nstalks - 1);
        x = uint16(x);
        
        % CONFIRMATION STEP
        cuts = [Dstart; x(:); Dend];
        xline(cuts,'LineWidth',3)
        delete(h1)                          
        ylimits = ylim();
        text(xlimits(1) + 100, -0.25, 'Click white to continue, gray to repeat.','FontSize',20)
        [x, y] = ginput(1);
        ylimits = ylim(); 
        xlimits = xlim();
        if x > xlimits(1) && x < xlimits(2) && y > ylimits(1) && y < ylimits(2)
            check = 1;
        else
            clf
            text(500, 1.5, 'Repeating....','FontSize',20)
            pause(0.5)
        end
        clf

    end
    % =========================================================================
    
    
    % ============ STEP 4 - LABELING  =====================================
    dat = dat - [0 1 2 3];              % remove the vertical shift
    CUTS = nan(Nstalks,7);              % create CUTS, an Nstalks x 7 matrix that holds all the stalk cut and label indices
    CUTS(:,1) = cuts(1:Nstalks);        % the first point for each stalk
    CUTS(:,7) = cuts(2:end);            % the last point for each stalk (note that last for one is first for the next)
    
    % LOOP THROUGH THE Nstalks
    for i = 1:Nstalks 
    
        % PLOTTING WITH CHECKS
        check = 0;
        while check == 0
            
            % INITIAL PLOT AND CLICK --------------------------------------
            clf
            index = CUTS(i,1):CUTS(i,7);
            plot(index(:), dat(index(:),:))         % note that the horizontal axis is in terms of indices. This keeps everything in the original global reference frame, reducing the possibility of mistakes
            h1 = text(CUTS(i,1) + 10, 1.5, 'Click 5 times to label data.','FontSize',20);
            title(['Stalk ', num2str(i)])
            [x, ~] = ginput(5);
            x = uint16(x);            
            CUTS(i,2:6) = x(:)';            
            hold on

            % CONFIRMATION STEP -------------------------------------------
            plot(CUTS(i,:), dat(CUTS(i,:),:),'o','MarkerSize',8, 'MarkerFaceColor','r') % plot red dots
            delete(h1)
            text(CUTS(i,1) + 10, 1.5, 'Click on white to continue.','FontSize',20)
            [x, y] = ginput(1);
            ylimits = ylim();
            xlimits = xlim();
            if x > xlimits(1) && x < xlimits(2) && y > ylimits(1) && y < ylimits(2)
                check = 1;
            else
                clf
                text(0.1, 0.5, 'Repeating....','FontSize',20)
                pause(0.5)
            end
        end
    
    end
    % =========================================================================


    % =============== STEP 5 - CONVERT DATA TO TABLE ==========================
    for i = 1:height(CUTS)     % Loop through all the stalks 
        index = CUTS(i,1):CUTS(i,7);
        SingleTestRun.Stalk(k) = i;                                        % Create the DataByStalk table which has one row of data for each stalk.
        SingleTestRun.Time(k) = {RawDataTable.Time(index)};                % Add all the relevant data.....
        SingleTestRun.StrainAx(k) = {RawDataTable.StrainAx(index)};
        SingleTestRun.StrainBx(k) = {RawDataTable.StrainBx(index)};
        SingleTestRun.StrainAy(k) = {RawDataTable.StrainAy(index)};
        SingleTestRun.StrainBy(k) = {RawDataTable.StrainBy(index)};
        SingleTestRun.DataStartOG(k) = CUTS(i,1);
        SingleTestRun{k,18:22} = CUTS(i,2:end-1) - CUTS(i,1);              % ***IMPORTANT***: This is one of the few places were I use INDICES rather that named variable columns
        SingleTestRun.DataEndOG(k) = CUTS(i,7);
        k = k + 1;
    end
    % =========================================================================
    
    
    % ======== STEP 6 - ADD METADATA FROM HEADER =======================================      
    SingleTestRun.Note(1:k-1) = HeaderData(1,2);
    SingleTestRun.RodneyConfig(1:k-1) = categorical(HeaderData(2,2));
    SingleTestRun.PVC(1:k-1) = categorical(HeaderData(3,2));
    SingleTestRun.Height(1:k-1) = single(HeaderData{4,2});
    SingleTestRun.Yaw(1:k-1) = single(HeaderData{5,2});
    SingleTestRun.Pitch(1:k-1) = single(HeaderData{6,2});
    SingleTestRun.Roll(1:k-1) = single(HeaderData{7,2});
    SingleTestRun.RateOfTravel(1:k-1) = single(HeaderData{8,2});
    SingleTestRun.AngleOfTravel(1:k-1) = single(HeaderData{9,2});
    SingleTestRun.Offset(1:k-1) = single(HeaderData{10,2});
    SingleTestRun.Operator(1:k-1) = {operator};
    SingleTestRun.Clicker(1:k-1) = {clicker};
    SingleTestRun.Filename(1:k-1) = {filename};
    SingleTestRun.Directory(1:k-1) = {folder};
    % =========================================================================


    % ======== STEP 7 - SAVE AND CONSOLIDATE DATA, MOVE PROCESSED FILES ============
   
    % SAVE DATA FROM THIS TEST RUN ----------------------------------------
    RunSaveFile = [filename,'.mat'];                % create filename
    save(RunSaveFile, 'SingleTestRun');             % save the file
    movefile(filename,'ProcessedFiles');            % move both files to the processed folder
    movefile(RunSaveFile,'ProcessedFiles');

    % APPEND TEST RUN DATA TO MULTI-RUN TABLE 
    if j == 1
        MultiTestRun = SingleTestRun; 
    else
        MultiTestRun = [MultiTestRun; SingleTestRun];
    end
    save(SessionFilename,'MultiTestRun') 
    % ---------------------------------------------------------------------

    % OPTION TO STOP PROCESSING -------------------------------------------
    keepGoing = questdlg('Continue?', ...
	    'Continue', ...
	    'Continue','Quit Labeling, Go to Append','Continue');

    switch keepGoing
        case 'Quit Labeling, Go to Append'
            break
    end
    % =====================================================================

    close all

end




% ======== STEP 7 - WRAPPING UP ===========================================
Nprocessed = height(MultiTestRun);

% APPEND QUERY ------------------------------------------------------------
append = questdlg('Append this data to an existing data table?', ...
	    'Append data?', ...
	    'Yes','No, Quit','Yes');

switch append
    case 'Yes'
       [AppendFile,folder] = uigetfile('.mat');     % Ask the user to identify the file for appending
       cd(folder)                                   % move to folder
       load(AppendFile);                        % Alternative: Use the T = load() syntax which loads contents into a structure. 
       % FieldNames = fieldnames(S);
       NstalksPrev = height(PVCtests);                    
       PVCtests = [PVCtests; MultiTestRun];
       NstalksNew = height(PVCtests);
       Nincreased = NstalksNew - NstalksPrev;

       % CONFIRMATION CHECK
       ConfirmationMessage = {['Stalks processed this session:    ', num2str(Nprocessed)], ...
                              ['Master Table increased by         ', num2str(Nincreased)], ...
                              ['    '], ...
                              ['Ok to save results?']}';                               
       confirm = questdlg(ConfirmationMessage, ...
	        'Append Confirmation', ...
	        'NO SAVE','Yes, Save','Yes, Save');

       if strcmp(confirm,  'Yes, Save') == 1
            SaveAppendFilename = ['PVC_MASTER_TABLE_', char(datetime(),'yyyy-MM-dd_HH.mm.ss'), '.mat'];
            save(SaveAppendFilename, 'PVCtests')
       end

end 

close all
