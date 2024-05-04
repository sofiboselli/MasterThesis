function subArrays = splitArrayFromCell(cellArray, n)
    % Validate the input arguments    
    % Extract the double array from the cell
    
    % Ensure the array is 2D and of the expected size
    [totalRows, totalColumns] = size(cellArray);
    
    % Ensure that the number of rows can be evenly divided by n
    if mod(totalRows, n) ~= 0
        error('The number of rows in the array must be evenly divisible by n');
    end
    
    % Calculate the number of rows in each sub-array
    rowsPerSubArray = totalRows / n;
    
    % Initialize a cell array to store the sub-arrays
    subArrays = cell(1, n);
    
    % Split the original double array into sub-arrays
    for i = 1:n
        startRow = (i - 1) * rowsPerSubArray + 1;
        endRow = i * rowsPerSubArray;
        
        % Extract the sub-array
        subArrays{i} = cellArray(startRow:endRow, :);
    end
end
