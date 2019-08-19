 
clear;
clc;



mazeName = 'b.txt';



mazeFile = importdata(mazeName, '\t', 1);
maze = mazeFile.data;

% Junctions have 3 or more neighbors
junctionNodes = (find(sum(maze(:, 4:7) == -1, 2) < 2)) - 1


% Powerpill nodes
powerpillNodes = maze(find(maze(:,9) > -1),1)