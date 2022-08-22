function [rlt] = GEV(ret_i,estimateur_hill,k)
%calculate the GEV to know the law judging from Hill estimator
    if estimateur_hill == 0
        rlt = exp(-exp(ret_i(k)));
    elseif estimateur_hill < 0 
        rlt = exp(-(1+(estimateur_hill*ret_i(1)^(-1/estimateur_hill))));
    else
        rlt = exp(-(1+(estimateur_hill*ret_i(end)^(-1/estimateur_hill))));
    end
end

