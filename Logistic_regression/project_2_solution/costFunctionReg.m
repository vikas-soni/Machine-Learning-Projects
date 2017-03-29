function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=sigmoid((theta'*X'))';
temp1=y.*log(h);
temp2=(1-y).*log(1-h);
theta_square = theta.^2;
theta_square = theta_square(2:size(theta,1));
temp3 = (sum(theta_square)*lambda)/(2*m);
J=((sum(-1*(temp1+temp2)))/m)+temp3;
grad_1 = (sum((h-y).*X)/m)';
grad_2=grad_1+(lambda/m)*theta;
grad=[grad_1(1);grad_2(2:size(theta))];




% =============================================================

end
