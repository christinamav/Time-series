function [pacfV, lags, significant_lags] = acf2pacf(rhoV, n, display, tittxt)
    % [pacfV, lags, significant_lags] = acf2pacf(rhoV,display)
    % Computes the partial autocorrelation (pacf) function for a 
    % given autocorrelation function (for lags up to 'p').
    %
    % INPUTS 
    % - rhoV    : array of size 'p x 1' of the autocorrelation for the first p
    %             lags (omitting autocorrelation at lag 0)
    % - n       : length of time-series, used for statistical significance
    %             testing
    % - display : if 1 then show a graph of the autocorrelation (if
    %             omitted no figure will be generated)
    % - tittxt  : text to display in the title of the figure (if display is
    %             zero, argument value will be ignored
    % OUTPUTS
    % - pacfV            : the array of size 'p x 1' of the partial autocorrelation values
    % - lags             : lags at which the partial autocorrelation is computed;
    %                      {0, 1, ..., maxtau}                    
    % - significant_lags : lags at which the partial autocorrelation function is    
    %                      statistically significant 
    
    % rejection region boundaries for statistical significance testing 
    boundary = 2/sqrt(n);
    
    % maximum lag at which autocorrelation is computed 
    p = length(rhoV);
    
    % lags at which the autocorrelation is computed 
    lags = (1:p);

    % initialise partial autocorrelation function as a vector of length p 
    pacfV = NaN*ones(p,1);

    % partial autocorrelation at lag 1 is equal to autocorrelation at lag 1
    pacfV(1) = rhoV(1);

    % compute partial autocorrelation for lags {2, ..., p}, by solving the 
    % Yule-Walker equations using Kramer-Rao method 
    for i=2:p
        % determine denominator and numernator matrices 
        denomM = toeplitz([1;rhoV(1:i-1)]);
        numerM = [denomM(:,1:i-1) rhoV(1:i)];
        % partial autocorrelation function is the fraction of the
        % determinants of the matrices 
        pacfV(i) = det(numerM)/det(denomM);
    end

   % plot partial autocorrelation function 
    if display
        figure;
        stem(lags, pacfV, 'filled', 'LineWidth', 1.5);

        xlabel('lag \tau', 'FontSize', 15)
        ylabel('$\hat{\phi}_{\tau, \tau}$', 'Interpreter', 'latex', 'FontSize', 15)

        yline(boundary, 'LineWidth', 1, 'Color', 'r', 'LineStyle', '--'); 
        yline(-boundary, 'LineWidth', 1, 'Color', 'r', 'LineStyle', '--'); 

        title(tittxt, 'FontSize', 15);
    end
    
    % find indices where partial autocorrelation is statistically significant 
    indices = abs(pacfV) >= boundary;
    % find corresponding lag values 
    significant_lags = lags(indices);
    
end