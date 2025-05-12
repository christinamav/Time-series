function [nrmseV,preM,phiV,thetaV] = predictARMAnrmse(xV,p,q,Tmax,nlast,tittxt)
    % [nrmseV,preM,phiV,thetaV] = predictARMAnrmse(xV,p,q,Tmax,nlast,tittxt)
    % Makes predictions with an ARMA(p,q) model on a last part
    % of a given time series and computes the prediction error (NRMSE measure)
    % for T-step ahead predictions. The model is
    % x(t) = phi(0) + phi(1)*x(t-1) + ... + phi(p)*x(t-p) + 
    %        +z(t) - theta(1)*z(t-1) + ... + theta(q)*z(t-q), 
    % z(t) ~ WN(0,sdnoise^2)
    % Note that if q=0, ARMA(p,q) reduces to AR(p) (autoregressive model of
    % order p), and if p=0, ARMA(p,q) reduces to MA(q) (moving average model of
    % order q).
    %
    % INPUTS:
    %  xV      : vector of the scalar time series
    %  p       : the order of AR part of the model
    %  q       : the order of MA part of the model
    %  Tmax    : the predictions in the test set are repeated for each of the 
    %            prediction steps T=1...Tmax
    %  nlast   : the size of the test set to compute the prediction error on
    %          : If not specified, it is half the length of the time series
    %  tittxt  : string to be displayed in the title of the figure.
    %            If not specified, no plot is made.
    % OUTPUT: 
    %  nrmseV  : vector of length Tmax, the nrmse for the predictions for time
    %            steps T=1...Tmax, on the test set
    %  preM    : matrix of nlast rows and Tmax columns, having the T-ahead 
    %            predictions at column T, T=1...Tmax. The cell at row i and 
    %            column T, has the T-step ahead prediction for the target at 
    %            time i-T, x(i-T), that is the notation of the prediction is 
    %            x_{i-T}(T) (to be compared to the true value x(i)).
    %  phiV    : the coefficients of the estimated AR time series (of length
    %            (p+1) with phi(0) as first component
    %  thetaV  : the coefficients of the estimated MA time series (of length q)
    
    % length of input time-series 
    n = length(xV);

    
    if nargin==5
        tittxt = [];
    elseif nargin==4
        tittxt = [];
        nlast = round(n/2);
    end

    if isempty(nlast)
        nlast = round(n/2);
    end

    if nlast>=n-2*q
        error('test set is too large for the given time series!')
    end

    % size of training set
    n1 = n-nlast; 

    % training set
    x1V = xV(1:n1); 
    
    % sample mean of training set 
    mx1 = mean(x1V); 

    % set mean of the training set to zero
    xx1V = x1V - mx1; 
    
    % construct an ARMA(p,q) model for the time-series centered at zero,
    % using only the training set 
    armamodel = armax(xx1V,[p q]);
    
    % extract coefficients of the estimated AR part 
    if p==0
        phiV = [];
    else
        % constant term 
        phi0 = (1+sum(armamodel.a(2:1+p)))*mx1;
        % phi(0), ..., phi(p) coefficients 
        phiV = [phi0;-armamodel.a(2:p+1)']; % Note that the AR coefficients are for the centered time series.
    end
    
    % extract coefficients of the estimated MA part 
    if q==0
        thetaV = [];
    else
        % theta(1), ..., theta(q) coefficients 
        thetaV = -armamodel.c(2:q+1)'; % Note that the MA coefficients are for the centered time series.
    end
    
    % initialise prediction array of size n by Tmax 
    % for simplicity use the indices for the whole time series, the first
    % n1 rows will be ignored
    preM = NaN(n,Tmax); 

    % subtract the mean of the training set from the whole time-series 
    xxV = xV - mx1;

    % make predictions for T=1,...,Tmax steps ahead for the whole
    % time-series 
    for T=1:Tmax
        % predict n values of the time-series using estimated ARMA model, 
        % for T steps ahead 
        tmpS = predict(armamodel,xxV,T);

        % add mean value of training set to the predictions 
        preM(:,T) = tmpS + mx1;   
    end  
    
    % compute NRMSE for T=1,...,Tmax steps ahead 
    nrmseV = ones(Tmax,1);
    for T=1:Tmax
        nrmseV(T) = nrmse(xV(n1+T:n),preM(n1+T:n,T));
    end
    
    % keep only the last (n-n1) rows of the predictions, which correspond
    % to the predictions of the test set 
    preM = preM(n1+1:n,:);
    
    if ~isempty(tittxt)
	    figno = gcf;
	    figure(figno)
	    clf
	    plot((1:Tmax),nrmseV,'.-k')
	    hold on
	    plot([1 Tmax],[1 1],'y')
	    xlabel('prediction time T')
	    ylabel('NRMSE(T)')
        if q==0
    	    title(sprintf('%s, NRMSE(T) for AR(%d) prediction, n=%d, nlast=%d',...
                tittxt,p,n,nlast))
        elseif p==0
            title(sprintf('%s, NRMSE(T) for MA(%d) prediction, n=%d, nlast=%d',...
               tittxt,q,n,nlast))
        else
            title(sprintf('%s, NRMSE(T) for ARMA(%d,%d) prediction, n=%d, nlast=%d',...
                tittxt,p,q,n,nlast))
        end
    end

end