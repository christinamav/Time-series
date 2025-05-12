function [rhoV, lags, significant_lags] = autocorrelation(xV, maxtau, display, tittxt)
    % [rhoV, lags, significant_lags] = autocorrelation(xV, maxtau, display, tittxt)
    % Computes and plots the autocorrelation of a time series.
    %
    % INPUTS:
    % - xV      : vector of a scalar time series
    % - maxtau  : largest delay time to compute autocorrelation for
    % - display : if 1 then show a graph of the autocorrelation (if
    %             omitted no figure will be generated)
    % - tittxt   : text to display in the title of the figure (if display is
    %             zero, argument value will be ignored)
    % OUTPUT:
    % - rhoV             : vector of length (maxtau + 1) containing the
    %                      autocorrelation at lags {0, 1, ..., maxtau}
    % - lags             : lags at which the autocorrelation is computed;
    %                      {0, 1, ..., maxtau}                    
    % - significant_lags : lags at which the autocorrelation function is    
    %                      statistically significant 
    
    % lags at which the autocorrelation is computed 
    lags = (0:maxtau);
    
    % sample mean value of time-series 
    xm = mean(xV);
    
    % subtract mean value from time-series 
    yV = xV - xm;
    
    % compute the autocorrelation function of time-series at lags 
    % {-maxtau, ..., maxtau}
    tmpV = xcorr(yV, maxtau, 'normalized');
    
    % keep autocorrelation values at lags {0, ..., maxtau}
    rhoV = tmpV(maxtau+1:2*maxtau+1);
    
    % time-series length
    n = length(xV);

    % rejection region boundaries for statistical significance testing 
    boundary = 2/sqrt(n);
    
    % plot autocorrelation function 
    if display
        figure;
        stem(lags, rhoV, 'filled', 'LineWidth', 1.5);

        xlabel('lag \tau', 'FontSize', 15)
        ylabel('r(\tau)', 'FontSize', 15)

        yline(boundary, 'LineWidth', 1, 'Color', 'r', 'LineStyle', '--'); 
        yline(-boundary, 'LineWidth', 1, 'Color', 'r', 'LineStyle', '--'); 

        title(tittxt, 'FontSize', 15);
    end
    
    % find indices where autocorrelation is statistically significant 
    indices = abs(rhoV) >= boundary;
    % find corresponding lag values 
    significant_lags = lags(indices);
    
    % remove first element that corresponds to lag 0, since autocorrelation
    % at 0 is always equal to 1
    significant_lags = significant_lags(2:end);
    
end 