function var = VaR_EVT(ret_i,estimateur_hill,p,k)
%Var_EVT calculate the VaR using the Hill Estimator
    n=size(ret_i,1);
    var = ((k / (n * (1 - p))) ^ estimateur_hill) * ret_i(n - k + 1);
end

