function [like,grad]=normalMLE(b,restrMat,Y,X,W,d)
%NORMALMLE estimates a linear regression model with potentially hetero-
%   skedastic error variances.
%   
%   LIKE = NORMALMLE(B,RESTRMAT,Y,X,D) 
%   estimates a linear regression model with errors assumed to be normal. 
%   The user can specify heteroskedasticity in these error variances.
%   Parameter restrictions are constructed in RESTRMAT.
%   
%   For estimation without restrictions, set RESTRMAT to be an empty matrix. 
%   
%   RESTRMAT is an R x 5 matrix of parameter restrictions. See APPLYRESTR
%   for more information in using this feature. If no parameter
%   restrictions are desired, RESTRMAT should be passed as an empty matrix
%   Y is an N x 1 vector of outcomes. 
%   X is an N x K matrix of covariates.
%   W is an N x 1 vector of weights.
%   D is an N x 1 vector of integers that indicates which variance group
%   an observation falls into (heteroskedasticity case). If homoskedastic
%   errors are assumed, D may be left unpassed or passed as an empty matrix
%   B is the parameter vector, with K + numel(unique(D)) elements
%   
%   This function does *not* automatically include a column of ones in X.
%   It also does *not* automatically drop NaNs
 
% Copyright 2014 Tyler Ransom, Duke University
% Revision History: 
%   September 25, 2014
%     Created 
%   September 29, 2014
%     Added code for weighted estimation 
%==========================================================================
 
% error checking
assert(size(X,1)==size(Y,1),'X and Y must be the same length');
if nargin==5
    d = ones(size(Y));
elseif nargin==6 && isempty(d)
    d = ones(size(Y));
end
J = numel(unique(d));
assert(  min(d)==1 && max(d)==J   ,'d should contain integers numbered consecutively from 1 through J');
assert(size(X,2)+J==size(b,1),'parameter vector has wrong number of elements');
if ~isempty(W)
    assert(isvector(W) && length(W)==length(Y),'W must be a column vector the same size as Y');
else
    W = ones(size(Y));
end
 
% apply restrictions as defined in restrMat
if ~isempty(restrMat)
    b = applyRestr(restrMat,b);
end
 
% slice parameter vector
beta      = b(1:end-J);
wagesigma = b(end-(J-1):end);
n         = length(Y);
 
% log likelihood
likemat = zeros(n,J);
dmat    = zeros(n,J);
for j=1:J
    dmat(:,j) = d==j;
    likemat(:,j) = -.5*(log(2*pi)+log(wagesigma(j)^2)+((Y-X*beta)./wagesigma(j)).^2);
end
like = -W'*sum(dmat.*likemat,2);
 
% analytical gradient
grad = zeros(size(b));
for j=1:J
    grad(1:end-J) = -X'*(W.*(d==j).*(Y-X*beta)./(wagesigma(j).^2)) + grad(1:end-J);
end
for j=1:J
    k=length(b)-(J-1)+j-1;
    temp = 1./wagesigma(j)-((Y-X*beta).^2)./(wagesigma(j).^3);
    grad(k) = sum(W.*(d==j).*temp);
end
 
end