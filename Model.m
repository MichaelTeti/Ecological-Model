% author=Michael Teti, FAU Center for Complex Systems

% Load Data and Initialize Parameters
data=load('ClutchData.txt');
X=data(:, 3:end);
X=[data(:, 1) X];
y=data(:, 2);
input_layer_size=4;
hidden_layer_size=50;
output_layer_size=1;
lambda=.5;
num_labels=1;
m=length(y);   % 681 training examples


% feature scaling
X_norm=featureScaling(X);

% randomly initialize weights
L_1in=input_layer_size;
L_1out=hidden_layer_size;
epsilon1=sqrt(6)/(sqrt(L_1in+L_1out));
L2_in=hidden_layer_size+1;
L2_out=num_labels;
epsilon2=sqrt(6)/(sqrt(L2_in+L2_out));
Theta1=rand(hidden_layer_size, input_layer_size)*2*epsilon1-epsilon1;  % Weights 1
Theta2=rand(num_labels, hidden_layer_size+1)*2*epsilon2-epsilon2;    % Weights 2
theta=[Theta1(:); Theta2(:)];   %weights to train


% Feedforward 
%z2=X_norm*Theta1';
%a2=sigmoid(z2);
%a2=[ones(size(a2, 1), 1) a2];
%z3=a2*Theta2';
%h=sigmoid(z3);
%
%
%y1=-y'*log(h);
%y0=(1-y)'*log(1-h);
%J=(y1-y0)/m;
%
%Theta1_reg=Theta1(:, 2:end);
%Theta2_reg=Theta2(:, 2:end);
%theta_all=[Theta1_reg(:); Theta2_reg(:)];
%reg=sum(theta.^2)*lambda/(2*m);   % regularization of cost function
%J=J+reg % Cost function with regularization


% Backpropagation
d1=zeros(size(Theta1));
d2=zeros(size(Theta2));

%for t=1:m
%  a_1=X_norm(t, :)';
%  z_2=Theta1*a_1;
%  a_2=sigmoid(z_2);
%  a_2=[1; a_2];
%  z_3=Theta2*a_2;
%  a_3=sigmoid(z_3);
%  err_out=zeros(m, 1);
%  err_out=a_3-y(t);
%  err_2=Theta2'*err_out;
%  err_2=err_2(2:end).*sigmoidGradient(z_2);
%  d2=d2+err_out*a_2';
%  d1 = d1 + err_2 * a_1';
%  end;
%
%Theta1_temp = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
%Theta2_temp = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
%Theta1_grad = 1 / m * d1 + lambda/m * Theta1_temp;
%Theta2_grad = 1 / m * d2 + lambda/m * Theta2_temp;
%
%
%grad=[Theta1_grad(:); Theta2_grad(:)];   % Gradient 
[J grad]=cost(X_norm, Theta1, Theta2, y, m, theta, lambda, d1, d2);

%theta=theta+.1*grad; 

options=optimset('GradObj', 'on', 'MaxIter', 100);
func=@(theta)cost(X_norm,Theta1,Theta2,y,m,theta,lambda,d1,d2);
[v fval]=fminunc(func, theta, options);


Theta1=reshape(theta(1:numel(Theta1)), 50, 4);
Theta2=reshape(theta(numel(Theta1)+1:end), 1, 51);



% predict
a2test=sigmoid(X_norm*Theta1');
a2test=[ones(size(a2test, 1), 1) a2test];
htest=sigmoid(a2test*Theta2');

p=ones(m, 1);
for j=1:m;
  if htest(j)=y(j)
    p(j)=0;
  end;
end;

e=sum(p==0);
percent_correct=(e/m)*100


MPCR_Stochastic_Gradient(J, grad);


