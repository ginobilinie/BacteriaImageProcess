function [] = writelogfile (str)

global logfile;

if isempty(logfile)
    fprintf('%s\n',str);
    return;
end

fid = fopen(logfile,'a');

if (fid ~= -1)
    fprintf(fid, '%s\n', str);
    fclose(fid);
else
    fprintf('%s\n',str);
end

end
