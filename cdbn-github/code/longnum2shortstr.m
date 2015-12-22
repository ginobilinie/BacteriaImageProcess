%This script is written to transform a long num to a short num, and keep n
%digits after . position
%params:
%t:the long num, note it is num instead of str
%n:the number of digits after . position
function x=longnum2shortstr(t,n)
if nargin==1
    n=2
end
num=floor(t);
str=num2str(num);
len=length(str);
str1=num2str(t);
nlen=min(length(str1),len+n+1);
x=str1(1:nlen);
return