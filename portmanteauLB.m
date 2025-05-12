function [hV,pV,QV,xautV] = portmanteauLB(serV,thesize,alpha, tittxt)
    % [hV,pV,QV,xautV] = portmanteauLB(serV,thesize,alpha,tittxt)
    % Performs Portmanteau hypothesis test (H0) for independence of time series:
    % tests jointly that several autocorrelations are zero.
    % It computes the Ljung-Box statistic of the modified sum of 
    % autocorrelations up to a maximum lag, for maximum lags 
    % {1,2,...,maxtau}. 
    %
    % INPUTS:
    % - serV    : a vector that can represent either
    %             (a) a scalar time series of length n, or
    %             (b) the autocorrelation function at lags {1,2,...,maxtau}
    % - thesize : a positive integer that denotes either 
    %             (a) the maximum lag 'maxtau' to compute autocorrelation for
    %             if the first argument 'serV' is a time series of length 
    %             'n' > 'thesize', or
    %             (b) the length of the time series 'n' if the first argument
    %             'serV' is the autocorrelation for lags up to 'maxtau' <
    %             'thesize'
    % - alpha   : significance level (default 0.05)
    % - tittxt  : specific title used in plots
    % 
    % OUTPUT:
    % - hV      : vector of length 'maxtau' of test decision values {0,1}
    %             for the given significance level maximum lags
    %             {1, 2, ..., maxtau}
    %             h=0 -> "do not reject H0", h=1 -> "reject H0"
    % - pV      : vector of length 'maxtau' of the corresponding p-values
    % - QV      : vector of length 'maxtau' of the corresponding Q statistics
    %             which follow a Chi-square distribution under H0.
    % - xautV   : vector of length 'maxtau' of the corresponding autocorrelation
    %             at lags {1, 2, ..., maxtau}
    % References:
    % Ljung, G. and Box, GEP (1978) "On a measure of lack of fit in time 
    % series models", Biometrika, Vol 66, 67-72.
    
    % default value
    if nargin == 2
        alpha = 0.05;
    end

    % error checks
    if (numel(alpha)>1), error('ALPHA must be a scalar.'); end
    if (alpha<=0 || alpha>=1), error('ALPHA must be between 0 and 1'); end
    
    %----------------------------------------------------------------------

    if length(serV) > thesize
        % input arguments are for a time series
        
        % time-series length
        n = length(serV);
        % maximum lag for which to compute the test statistic 
        maxtau = thesize;
        
        % compute autocorrelation at lags {1, 2, ..., maxtau}
        tmpV = autocorrelation(serV, maxtau, 0,"");
        xautV = tmpV(2:end);
    else
        % input arguments are for autocorrelation

        % time-series length
        n = thesize;
        % maximum lag for which to compute the test statistic 
        maxtau = length(serV);
        
        % autocorrelation at lags {1, 2, ..., maxtau}
        xautV = serV;
    end  

    %----------------------------------------------------------------------

    % initialise p-value, test statistic, test results and critical value 
    % vectors
    pV = NaN*ones(maxtau, 1);
    QV = NaN*ones(maxtau, 1);
    hV = NaN*ones(maxtau, 1);
    critical_values = NaN*ones(maxtau, 1);

    % compute squared autocorrelation values at lags {1, 2, ..., maxtau}
    xautsqV = xautV.^2;
    
    % iterate through lag values and compute the test statistic, p-values 
    % and test results at each lag up to the maximum lag 
    sumxautsq = 0;
    for t= 1:maxtau
        sumxautsq = sumxautsq + xautsqV(t)/ (n - t);

        % test statistic  value at maximum lag t; follows chi-square
        % distribution with t degrees of freedom 
        QV(t) = n * (n + 2) * sumxautsq;

        % p-value of hypothesis test 
        pV(t) = 1 - chi2cdf(QV(t),t);

        % critical value of hypothesis test; value of random variable that
        % follows a chi-square distribution with t degrees of freedom and
        % has a probability equal to (1 - alpha)
        critical_values(t) = chi2inv(1 - alpha, t);
        
        % hypothesis test results 
        hV(t) = (QV(t) > critical_values(t));
    end

    %----------------------------------------------------------------------
    
    % plot p-values with respect to each lag up to maxtau
    figure;
    plot(1:maxtau, pV, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    hold on;
    plot([0 maxtau+1],alpha*[1 1],'--r', 'LineWidth', 1.5);
    xlabel('lag k', 'FontSize', 15);
    ylabel('p-value','FontSize', 15);
    title(['Ljung-Box Portmanteau test', tittxt], 'FontSize', 15);
    axis([0 maxtau+1 0 1]);
    hold off;

    % plot test statistic and critical values with respect to each lag up
    % to maxtau
    figure;
    plot(1:maxtau, QV, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    hold on;
    plot(1:maxtau, critical_values, '-oc', 'LineWidth', 1.5, 'MarkerFaceColor', 'c');
    hold off;
    xlabel('lag k', 'FontSize', 15);
    legend('test statistic Q', 'critical value \chi^2_{k, 1-\alpha}', 'FontSize', 15, 'location', 'best');
    title(['Ljung-Box Portmanteau test', tittxt], 'FontSize', 15);

end
