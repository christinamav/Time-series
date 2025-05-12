function sV = seasonal_components(yV,period) 
    % sV = seasonal_components(yV,period)
    % Computes the periodic time series comprised of repetetive
    % patterns of seasonal components given a time series and the season
    % (period).
    %
    % INPUTS 
    % - yV      : vector of length 'n' of the time series
    % - period  : the season (period)
    % OUTPUTS
    % - sV      : vector of length 'n' of the time series of seasonal components
    
    % time-series length
    n = length(yV);

    % initialise vector of periodic function with length of one time period 
    monV = NaN*ones(period,1);
    
    % initialise vector of seasonality component with length that of the
    % time-series 
    sV = NaN*ones(n,1);

    % compute elements of the periodic function using the means of
    % time-series elements in periodic intervals 
    for j=1:period
        monV(j) = mean(yV(j:period:n));
    end
    
    % remove mean of periodic function elements 
    monV = monV - mean(monV);

    % extract seasonal component of the whole time-series by replicating 
    % periodic elements 
    for j=1:period
        sV(j:period:n) = monV(j)*ones(length(j:period:n),1);
    end

end
