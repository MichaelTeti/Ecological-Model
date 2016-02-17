function [J grad]=cost(X_norm, Theta1, Theta2, y, m, theta, lambda, d1, d2);
% Feedforward 
z2=X_norm*reshape(theta(1:200), 50, 4)';
a2=sigmoid(z2);
a2=[ones(size(a2, 1), 1) a2];
z3=a2*reshape(theta(201:end), 1, 51)';
h=sigmoid(z3);


y1=-y'*log(h);
y0=(1-y)'*log(1-h);
J=(y1-y0)/m;

Theta1_reg=theta(51:end);
Theta2_reg=theta(202:end);
theta_all=[Theta1_reg(:); Theta2_reg(:)];
reg=sum(theta.^2)*lambda/(2*m);   % regularization of cost function
J=J+reg; % Cost function with regularization

if nargout > 1
%Backpropagation
  for t=1:m
    a_1=X_norm(t, :)';
    z_2=reshape(theta(1:200), 50, 4)*a_1;
    a_2=sigmoid(z_2);
    a_2=[1; a_2];
    z_3=reshape(theta(201:end), 1, 51)*a_2;
    a_3=sigmoid(z_3);
    err_out=zeros(m, 1);
    err_out=a_3-y(t);
    err_2=reshape(theta(201:end), 1, 51)'*err_out;
    err_2=err_2(2:end).*sigmoidGradient(z_2);
    d2=d2+err_out*a_2';
    d1 = d1 + err_2 * a_1';
  end;
  
  Theta1_temp = [zeros(size(Theta1,1),1) reshape(theta(51:200), 50, 3)];
  Theta2_temp = [zeros(size(Theta2,1),1) reshape(theta(202:end), 1, 50)];
  Theta1_grad = 1 / m * d1 + lambda/m * Theta1_temp;
  Theta2_grad = 1 / m * d2 + lambda/m * Theta2_temp;
  
 
  grad=[Theta1_grad(:); Theta2_grad(:)];    % Gradient 
 end;
end;
