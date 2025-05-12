clear; clc; close all;

% IMPORT AND EXTRACT DATA 

data = readtable("train.csv");

% extract dates 
dates = table2array(data(:,1));

% select specific station (team 12) 
station_sel = 12;

% extract measurements 
solar_energy = table2array(data(:, station_sel));

% extract station names 
station_name = data.Properties.VariableNames(station_sel+1);

% number of observations 
N = size(solar_energy, 1);

%--------------------------------------------------------------------------

% VISUALISATION 

% visualise time-series 
figure;
plot(solar_energy);
title("Daily solar energy time-series data in station " + station_name, 'FontSize', 15);

%--------------------------------------------------------------------------

% PREPROCESSING 

% find consecutive constant values in time-series data
% first order differences of time-series 
dyV = diff(solar_energy);
% indices where first order difference is zero 
zero_indices = find(dyV == 0);

% assume a 365-day year (seasonality length) 
season_len = 365;

% adjust consecutive constant values 
adjusted_solar_energy = solar_energy;

for i=1:size(zero_indices, 1)
    % measurement took place in the first year of the time-series 
    if zero_indices(i) <= season_len
        % replace measurement with measuremnt of following year 
        adjusted_solar_energy(zero_indices(i)) = solar_energy(zero_indices (i) + season_len);

    % measurement took place in the last year of the time-series
    elseif zero_indices (i) >= N - season_len + 1
        % replace measurement with measurement of previous year 
        adjusted_solar_energy(zero_indices(i)) = solar_energy(zero_indices (i) - season_len);

    else
        % replace measurement with the mean measurement of previous and
        % following year 
        adjusted_solar_energy(zero_indices(i)) = mean([solar_energy(zero_indices(i) - season_len), solar_energy(zero_indices(i) + season_len)]);
    end
end

% visualise adjusted time-series 
figure;
plot(adjusted_solar_energy);
title("Adjusted daily solar energy time-series data in station " + station_name, 'FontSize', 15); 

%--------------------------------------------------------------------------

% SEASONALITY REMOVAL (step 1)

% remove seasonality from the time series
% assuming the time-series has no trend component, the resulting
% time-series can be assumed stationary 
[xV, sV]  = remove_seasonality(adjusted_solar_energy, season_len);
% visualise seasonality components of non-stationary time-series 
figure; 
plot(adjusted_solar_energy);
hold on;
plot(sV + mean(xV));
hold off;
legend("Non-stationary time-series", "Seasonality component", 'FontSize', 15);
title("Adjusted daily solar energy time-series data in station " + station_name, 'FontSize', 15); 

% visualise residuals 
figure;
plot(xV);
hold on;
yline(mean(xV), '--r', 'LineWidth', 1.5);
hold off;
legend("", "\mu = " + num2str(mean(xV), '%e'), 'FontSize', 15);
title("Residual time-series after seasonality component removal", 'FontSize', 15);

%--------------------------------------------------------------------------

% STATISTICAL INDEPENCY TEST (step 2)

% check if the stationary time series is white noise

% maximum lag 
maxtau = 20;

% compute autocorrelation for lags 1,...,max_tau
[~, lags, significant_lags] = autocorrelation(xV, maxtau, 1, "Autocorrelation of stationary time-series");

% compute percentage of statistically significant autocorrelation function
% values, assuming significance level of alpha = 0.05 
percentage = length(significant_lags)/maxtau;
fprintf("Statistically significant autocorrelation function values at lags:\n");
disp(significant_lags);
fprintf("Percentage of statistically significant autocorrelation function values: %.4f\n\n", percentage);

% significance level for Pormanteau hypothesis testing 
alpha = 0.05;

% perform Portmanteau hypothesis testing 
[hV, pV, QV, xautV] = portmanteauLB(xV, maxtau, alpha, "for stationary time-series");

%--------------------------------------------------------------------------

% FIT LINEAR STOCHASTIC PROCESS MODEL (step 3)

% investigate most suitable linear model for stationary time-series 

% compute the partial autocorrelation function 
[partial_aut, ~, significant_lags] = acf2pacf(xautV, N, 1, "Partial autocorrelation of stationary time-series");

fprintf("Statistically significant partial autocorrelation function values at lags:\n");
disp(significant_lags);


% compute AIC values for ARMA(p, q) models for different combinations of
% order values 
% p in {0, 1, ..., 7}
% q in {0, 1, ..., 7}

% initialise an array of size p_len by q_len
% each row represents p, each column represents q 
% p values will range from 0 to (p_len - 1), q values will range from 0 to
% (q_len - 1)
p_len = 5;
q_len = 5;
aics = zeros(p_len, q_len);
fpes = zeros(p_len, q_len);

% iterate through p, q values 
for p= 0:(p_len-1) 
    for q= 0:(q_len-1)
        % fit ARMA(p, q) model and return AIC and FPE value 
        [~, ~, ~, ~, aicS, fpeS, ~, ~, ~] = fitARMA(xV, p, q, 1);
        aics(p+1, q+1) = aicS;
        fpes(p+1, q+1) = fpeS;
    end
end    


% plot AIC vs. p for different q values
figure;
hold on;
colors = lines(q_len); % different colors for each q
for j = 1:q_len
    plot(0:(p_len-1), aics(:, j), '-', 'Color', colors(j,:), 'DisplayName', sprintf('q=%d', j-1), 'LineWidth', 1.5);
end
hold off;

xlabel('p (AR order)', 'FontSize', 15);
ylabel('AIC', 'FontSize', 15);
title('AIC for different ARMA(p,q) models', 'FontSize', 15);
legend('show', 'FontSize', 15);
grid on;


% plot FPE vs. p for different q values
figure;
hold on;
colors = lines(q_len); % different colors for each q
for j = 1:q_len
    plot(0:(p_len-1), fpes(:, j), '-', 'Color', colors(j,:), 'DisplayName', sprintf('q=%d', j-1), 'LineWidth', 1.5);
end
hold off;

xlabel('p (AR order)', 'FontSize', 15);
ylabel('FPE', 'FontSize', 15);
title('FPE for different ARMA(p,q) models', 'FontSize', 15);
legend('show', 'FontSize', 15);
grid on;


% by examining the AIC and FPE plots, we conclude to select an ARMA(2,2)
% model for predictions 
p_best = 2;
q_best = 2;

%--------------------------------------------------------------------------

% diagnose model suitability by examining if the residuals of the best
% fitted model come from a white noise process  

% fit ARMA(2,2) model 
[best_nrmseV, best_phiV, best_thetaV, best_SDz, best_aicS, best_fpeS, best_armamodel, best_xpreM, best_residuals] = fitARMA(xV, p_best, q_best, 1);

% plot original stationary time-series along with estimated ARMA model
% values 
figure;
plot(xV);
hold on;
plot(best_xpreM(:,1));
hold off;
title("Original stationary time-series and estimated ARMA(2,2) model", 'FontSize', 15);
legend("original time-series", "estimated ARMA model", 'FontSize', 15);

% compute autocorrelation for lags 0, 1,...,max_tau
[rhoV_best, ~, ~] = autocorrelation(best_residuals, maxtau, 1, "Autocorrelation of residuals");

% perform Portmanteau hypothesis testing 
[h_res_best, ~, ~, ~] = portmanteauLB(best_residuals, maxtau, alpha, "for residuals");

%--------------------------------------------------------------------------

% PREDICTION WITH LINEAR MODEL (step 4) 

% define the training set as the measurements made in years 1994-2006
train_indices = (dates < 20070101);
solar_energy_train = adjusted_solar_energy(train_indices);

% length of training set
n1 = length(solar_energy_train);

% extract seasonal component only from the training set 
[x_train1, s_train1] = remove_seasonality(solar_energy_train, season_len);

% expand to match the length of the whole time-series 
r = mod(n1, season_len);
extrap_seasonalV = [s_train1(season_len-r+1:season_len); s_train1(1:(season_len-r))];
s_train = [s_train1; extrap_seasonalV];

% subtract estimated seasonal component from the whole time-series 
x_train = adjusted_solar_energy - s_train;


% prediction horizon 
Tmax = 30;

% make predictions for 1, ..., Tmax steps ahead for all days in year 2007,
% using the ARMA model selected above 
[~, predictions_arma, ~, ~] = predictARMAnrmse(x_train, p_best, q_best, Tmax, N-n1);

% add seasonality component to the predictions of the stationary
% time-series to extract predictions for the original time-series 
predictions_arma = predictions_arma + repmat(extrap_seasonalV, 1, Tmax);


% calculate NRMSE with respect to the number of steps ahead, on the
% predictions of the original non-stationary time-series 
nrmseV = zeros(Tmax,1);
for T= 1:Tmax
    nrmseV(T) = nrmse(adjusted_solar_energy(n1+T:N), predictions_arma(T:end, T));
end

% plot predicted along with true values in year 2007 for 1 step ahead
% predictions
figure; 
plot(adjusted_solar_energy(n1+1:end)); 
hold on;
plot(predictions_arma(:, 1));
hold off;
txt = sprintf("Predictions using the ARMA(%d,%d) model (1 step ahead)", p_best, q_best);
title(txt, 'FontSize', 15);
legend("actual time-series", "predictions", 'FontSize', 15);

% plot NRMSE values with respect to the number of steps ahead 
figure; 
plot(1:Tmax, nrmseV, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
txt = sprintf("NRMSE of the predictions using ARMA(%d,%d) model", p_best, q_best);
title(txt, 'FontSize', 15); 
xlabel("T", 'FontSize', 15); 
ylabel("NRMSE(T)", 'FontSize', 15);

%--------------------------------------------------------------------------

% PREDICTION WITH SEASONALITY COMPONENT EXTRAPOLATION (step 5)

% predictions using the seasonal component only
% we expect the two predictions to converge to one another since the ARMA
% model will converge to the mean of the time series as T grows 
predictions_seasonal = extrap_seasonalV + mean(x_train1);

% plot predicted along with true values in year 2007 
figure; 
plot(adjusted_solar_energy(n1+1:end));
hold on; 
plot(predictions_seasonal);
hold off; 
title("Predictions using the seasonal component s(t)", 'FontSize', 15);

% calculate the nrmse for the seasonal prediction
nrmse_seasonal = nrmse(adjusted_solar_energy(n1+1:end), predictions_seasonal);

fprintf("ARMA(2,2) model predictions: NRMSE(1) = %.6f\n", nrmseV(1));
fprintf("Seasonality component extrapolation predictions: NRMSE(1) = %.6f\n\n", nrmse_seasonal);

%---------------------------------------------------------------------------

% SNRMSE CALCULATION (step 6)

% calculate SNRMSE with respect to the number of steps ahead 
snrmseV = zeros(Tmax, 1); 

for T= 1:Tmax
    snrmseV(T) = snrmse(adjusted_solar_energy(n1+T:N), predictions_arma(T:end, T), extrap_seasonalV(T:end));
end

% plot SNRMSE along with NRMSE for the selected ARMA(2,2) model 
figure; 
plot(1:Tmax, snrmseV, '-o', 'Color', "#D95319", 'LineWidth', 1.5, 'MarkerFaceColor', "#D95319");
hold on;
plot(1:Tmax, nrmseV, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold off;
txt = sprintf("NRMSE of the predictions using ARMA(%d,%d) model", p_best, q_best);
title(txt, 'FontSize', 15);
xlabel("T", 'FontSize', 15); 
legend("SNRMSE", "NRMSE", 'FontSize', 15, 'Location', 'best');

%---------------------------------------------------------------------------