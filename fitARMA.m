function [nrmseV, phiV, thetaV, SDz, aicS, fpeS, armamodel, xpreM, residuals] = fitARMA(xV, p, q, Tmax)
    % [nrmseV,phiV,thetaV,SDz,aicS,fpeS,armamodel,xpreM, residuals] = fitARMA(xV,p,q,Tmax)
    % Fits an autoregressive moving average (ARMA) model and
    % computes the fitting error (normalized root mean square error) for a
    % given number of steps ahead. 
    % The ARMA model has the form
    % x(t) = phi(0) + phi(1)*x(t-1) + ... + phi(p)*x(t-p) + 
    %        +z(t) - theta(1)*z(t-1) + ... - theta(q)*z(t-q), 
    % z(t) ~ WN(0,sdnoise^2)
    %
    % INPUTS:
    %  xV      : vector of the scalar time series
    %  p       : the order of the AR part of the model
    %  q       : the order of the MA part of the model
    %  Tmax    : the prediction horizon, the fit error is computed for
    %            T=1...Tmax steps ahead
    % OUTPUT: 
    %  nrmseV  : vector of length Tmax, the nrmse of the fit for T-mappings,
    %            T=1...Tmax. 
    %  phiV    : the coefficients of the estimated AR part, of length
    %            (p+1), with phi(0) as first component
    %  thetaV  : the coefficients of the estimated MA part (of length q)
    %  SDz     : the standard deviation of the noise term
    %  aicS    : the AIC value for the model
    %  fpeS    : the FPE value for the model
    %  armamodel : the model structure (contains all the above apart from
    %               nrmseV)
    %  xpreM   : matrix of Tmax columns, where each T column is the vector 
    %            of T-time step ahead in-sample-predicted values.
    %            Note that the number of rows is n, whereas the first p+T-1 
    %            values of each T column are close to the true observations for  
    %            times 1,2,...,p+T-1 (new versions of Matlab estimate these
    %            values as well).
    % residuals: residuals of the estimated ARMA model 
    
    % time-series length
    n = length(xV);
    % sample mean of time-series 
    mx = mean(xV);
    % subtract mean from the time-series 
    xxV = xV-mx;
    
    % construct ARMA(p,q) model for the time-series with zero mean 
    armamodel = armax(xxV,[p q]);

    % extract coefficients of the estimated AR part 
    if p==0
        phiV = [];
    else
        % constant term 
        phi0 = (1+sum(armamodel.a(2:p+1)))*mx;
        % phi(0), ..., phi(p) coefficients 
        phiV = [phi0 -armamodel.a(2:p+1)];
        
        % roots of characteristic polynomial of AR part 
        rootarV = roots(armamodel.a);
        % if any root is outside the unitary circle, the AR part is not
        % stationary 
        if any(abs(rootarV)>=1)
            fprintf('The estimated AR(%d) part of the model is not stationary.\n',p);
        end
    end

    % extract coefficients of the estimated MA part 
    if q==0
        thetaV = [];
    else
        % theta(1), ..., theta(q) coefficients 
        thetaV = -armamodel.c(2:end);
        
        % roots of characteristic polynomial of MA part 
        rootmaV = roots(armamodel.c);
        %  if any root is outside the unitary circle, the MA part is not
        %  reversible 
        if any(abs(rootmaV)>=1)
            fprintf('The estimated MA(%d) part of the model is not reversible.\n',q);
        end
    end
    
    % compute noise variance 
    SDz = sqrt(armamodel.NoiseVariance);

    % compute AIC of the estimated model 
    aicS = aic(armamodel);
    % compute FPE of the estimated model 
    fpeS = armamodel.EstimationInfo.FPE;

    % initialise NRMSE vector of length Tmax 
    nrmseV = NaN*ones(Tmax,1);
    % initialise array of predictions of size n by Tmax 
    xpreM = NaN(n,Tmax);

    % make in-sample predictions and compute the NRMSE for T=1, ...,Tmax
    % steps ahead 
    for T=1:Tmax
        % predict n values of the time-series using estimated ARMA model 
        tmpS = predict(armamodel,xxV,T);
        % add mean value to the predictions 
        xpreV = tmpS + mx;
        
        % compute NRMSE for T steps ahead 
        nrmseV(T) = nrmse(xV(q+1:n), xpreV(q+1:n));
        
        % assign predictions to the T-th column of output vector 
        xpreM(:,T) = xpreV; 
    end
    
   % compute residuals of estimated ARMA model; these are the differences
   % between the original time-series and the 1-step ahead predictions made
   % by the estimated model 
   residuals = xV - xpreM(:, 1);

end