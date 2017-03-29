function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h=((theta'*X'))';

J=(1/(2*m)).*sum((h-y).^2);

theta_square = theta.^2;
theta_square = theta_square(2:size(theta,1));
temp3 = (sum(theta_square)*lambda)/(2*m);

J = J + temp3;

grad_1 = (sum((h-y).*X)/m)';
grad_2=grad_1+(lambda/m)*theta;
grad=[grad_1(1);grad_2(2:size(theta))];

% =========================================================================

grad = grad(:);

end
