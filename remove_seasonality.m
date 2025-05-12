function [xV, sV] = remove_seasonality(yV, period)
    % [xV, sV]  = remove_seasonality(yV, period)
    % Computes the periodic time series of seasonal components and removes
    % them from the original time-series.
    %
    % INPUTS 
    % - yV      : vector of length 'n' of the time series
    % - period  : the season (period)
    % OUTPUTS
    % - xV      : vector of length 'n' of the time series where the
    % seasonality component is removed
    % - sV      : seasonality components of original timeseries 
    
    % find the seasonal components of the time-series 
    sV = seasonal_components(yV, period);
    
    % remove seasonal components from the original non-stationary
    % time-series 
    xV = yV - sV;

end