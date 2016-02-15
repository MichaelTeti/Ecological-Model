function X_norm=featureScaling(X);
  % Feature Scaling
mu=zeros(1, size(X, 2));
sigma=zeros(1, size(X, 2));
for i=1:size(mu, 2);
  mu(1, i)=mean(X(:, i));
  sigma(1, i)=std(X(:, i));
end;

X_norm=(X-mu)./sigma;
X_norm=[ones(size(X_norm, 1), 1) X_norm];
end;
