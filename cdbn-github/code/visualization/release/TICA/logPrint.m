function [] = logPrint(varargin)
        global logFile
        if isempty(logFile)
		sprintf('\n')
		sprintf(varargin{:})
        else
                [fid,msg] = fopen(logFile,'a');
		fprintf(fid,'\n');
		fprintf(fid,varargin{:});
        end


	if ~isempty(logFile)
		fclose(fid);
	end	

end

