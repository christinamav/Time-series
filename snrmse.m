function y = snrmse(trueV, predV, sV)
    % y = nrmse(trueV,predV, sV) 
    % Computes the seasonal normalized root mean square error 
    % using 1/(N-1) for the computation of SD.
    %
    % INPUTS
    %  trueV: Vector of correct values
    %  predV: Vector of predicted values
    %  sV   : seasonal component 

    vartar = sum((trueV - sV).^2);
    varpre = sum((trueV - predV).^2);
    y = sqrt(varpre / vartar);

end