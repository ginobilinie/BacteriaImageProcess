%this script is written to normalize a 3-d matrix to [0,1]
function m=normalizeMatrix(mat)
xmax=max(mat(:));
xmin=min(mat(:));
m=(mat-xmin)/(xmax-xmin);
end