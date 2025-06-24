% =========================================================================
% PURPOSE: This script renames filenames so that they can be sorted into the
% the order of performance. 
%
% EXPLANATION: Filenames are currently named as "DATE_1.csv" instead of 
% "DATE_01.csv". The latter will sort properly but the former will not.
% This script creates filenames with leading zeros as needed.
% =========================================================================

% Create the "Filename2" variable if it does not already exist.
if ~ismember('Filename2', PVCtests.Properties.VariableNames)
    PVCtests.Filename2 = PVCtests.Filename;
end



for i = 1:height(PVCtests)

    filename = PVCtests.Filename{i};           % get the original filename
    index = regexp(filename, '_\d\.csv');       % search for the pattern '_#.csv', where # is any single digit

    % if index is a scalar value:
    if isscalar(index) == 1
        filename = [filename(1:index), '0', filename(index+1), '.csv'];     % create the new filename
        PVCtests.Filename2{i} = filename;                                  % store the new filename in Filename2
    end

    
end


