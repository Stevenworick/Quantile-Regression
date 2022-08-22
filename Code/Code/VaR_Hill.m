function [var_h] = VaR_Hill(ret,q)
%Var_Hill calculate the VaR using the Hill Estimator 
    var_gauche = zeros(size(ret,2),1);
    var_droite = zeros(size(ret,2),1);
    for i = 1:size(ret,2)
       ret_i = ret(:,i);
       gain_i = sort(ret_i(ret_i>=0));
       perte_i = sort(ret_i(ret_i<0));

       estimateur_gain = zeros(size(gain_i));

       for k = 1:size(gain_i,1)
           estimateur_g = sum(log(gain_i(end-k+1:end)/gain_i(end-k+1)))/k;
           estimateur_gain(k) = estimateur_g;
       end

       estimateur_perte = zeros(size(perte_i));
       for k = 1:size(perte_i,1)
           estimateur_p = sum(log(perte_i(end-k+1:end)/perte_i(end-k+1)))/k;
           estimateur_perte(k) = estimateur_p;
       end

       k_perte = round(sqrt(size(perte_i,1)));
       k_gain = round(sqrt(size(gain_i,1)));

       estimateur_hill_perte = estimateur_perte(k_perte);
       estimateur_hill_gain = estimateur_gain(k_gain);

       var_gauche(i) = VaR_EVT(perte_i,estimateur_hill_perte,q,k_perte);
       var_droite(i) = VaR_EVT(gain_i,estimateur_hill_gain,1-q,k_gain);

    end 
    var_h = [var_gauche var_droite];
end

