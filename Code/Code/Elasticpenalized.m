function [vrais]=Elasticpenalized(eps)

global q
global lambda
global r_
global X
global T
global alpha

    %W=[ones(T-1,1) normalize(X(:,2:end))];
    p1=q-(r_<=X*eps);
    p2=r_-X*eps;
    p3=sum(p1.*p2)/(T-1);
    p4=(lambda*(q*(1-q))^0.5/(T-1))*((1-alpha)*sum(eps.^2)+alpha*sum(abs(eps)));
    vrais=p3+p4;
end

