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

%--------------------------------------------------------------------------

% SEASONALITY REMOVAL 

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

% visualise the stationary component 
figure;
plot(xV);
hold on;
yline(mean(xV), '--r', 'LineWidth', 1.5);
hold off;
legend("", "\mu = " + num2str(mean(xV), '%e'), 'FontSize', 15);
title("Residual time-series after seasonality component removal", 'FontSize', 15);

%--------------------------------------------------------------------------

% FIT NON-LINEAR MODEL (step 7)

% we are going to fit a non-linear model to the stationary time-series 
% for this purpose, we will use state space reconstruction 

% DELAY ESTIMATION 
% compute and plot the mutual information of the stationary time series 
mut_info = mutualinformation(xV, 20, [], 'Mutual information of stationary time-series');
% compute and plot autocorrelation function of the stationary time-series 
[~, ~, ~] = autocorrelation(xV, 10, 1, "Autocorrelation of stationary time-series");
% plot horizontal line with value 1/e
yline(1/exp(1), '--c', 'LineWidth', 1.5);
legend("", "", "",  "r(\tau) = 1/e", 'FontSize', 15);

% selected lag for the state space reconstruction
tau = 2; 

% EMBEDDING DIMENSION ESTIMATION 
% false nearest neighbors for the calculation of the embedding dimension 
fnnM = falsenearest(xV, tau, 10, 10, 0, "False Nearest Neighbors"); 

% set threshold value; select the value of m for which fnn is below that
% threshold, if no such value exists, select the value of m corresponding
% to the last non- NaN value of fnn 
threshold = 0.01;
i = find(fnnM(:, 2) < threshold, 1, 'first');
if ~isempty(i)
    m = i;
else
    m = find(isnan(fnnM(:, 2)), 1, 'first') - 1;
end

%--------------------------------------------------------------------------

% FIT LOCAL LINEAR MODEL 

% k: number of nearest neighbors taken into consideration for local model
% estimation 
% we are going to choose the hyperparameter k in order to minimise the
% NRMSE of the fitted model 

% k values for which to fit local linear model 
k_vals = 8:2^4:300;
% initialise vector of size k_vals to hold NRMSE for each k 
nrmse1 = zeros(size(k_vals));

% iterate through values of k 
for i = 1:length(k_vals)
    fprintf("Fitting for number of neighbors k = %i\n", k_vals(i));
    % fit local linear model 
    [nrmseV, ~] = localfitnrmse(xV, tau, m, 1, k_vals(i), m);
    nrmse1(i) = nrmseV(1);
end
fprintf("\n");

% plot NRMSE(1) with respect to k (number of nearest neigbors) 
figure;
plot(k_vals, nrmse1, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel("k", 'FontSize', 15);
ylabel('NRMSE(1)', 'FontSize', 15);
title("NRMSE of fitted Local Linear model", 'FontSize', 15);

% select threshold value; the first value of k at which nrmse is below this
% threshold is selected 
threshold = 0.94;
idx = find(nrmse1 < threshold, 1);
best_k_llp = k_vals(idx);

fprintf("Best value of k for Local Linear Model: %d\n\n", best_k_llp);

% fit local linear model with best value of k 
[~, ~, residuals_llp] = localfitnrmse(xV, tau, m , 1, best_k_llp, m); 

%--------------------------------------------------------------------------

% diagnose model suitability by examining if the residuals of the best
% fitted model come from a white noise process 

% maximum lag
maxtau = 20; 
% significance level for Pormanteau hypothesis testing 
alpha = 0.05;

% compute autocorrelation for lags 0, 1,...,max_tau
[~, ~, ~] = autocorrelation(residuals_llp, maxtau, 1, "Autocorrelation of residuals of Local Linear Model");

% perform Portmanteau hypothesis testing 
[~, ~, ~, ~] = portmanteauLB(residuals_llp, maxtau, alpha, "on residuals of Local Linear Model");

%--------------------------------------------------------------------------

% FIT LOCAL AVERAGE MODEL 

% k: number of nearest neighbors taken into consideration for local model
% estimation 
% we are going to choose the hyperparameter k in order to minimise the
% NRMSE of the fitted model 

% k values for which to fit local linear model 
k_vals = 8:2^4:300;
% initialise vector of size k_vals to hold NRMSE for each k 
nrmse1 = zeros(size(k_vals));

% iterate through values of k 
for i = 1:length(k_vals)
    fprintf("Fitting for number of neighbors k = %i\n", k_vals(i));
    % fit local average model 
    [nrmseV, ~] = localfitnrmse(xV, tau, m, 1, k_vals(i));
    nrmse1(i) = nrmseV(1);
end
fprintf("\n");

% plot NRMSE(1) with respect to k (number of nearest neigbors) 
figure;
plot(k_vals, nrmse1, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel("k", 'FontSize', 15);
ylabel('NRMSE(1)', 'FontSize', 15);
title("NRMSE of fitted Local Average Model", 'FontSize', 15);

% select value of k which minimises the nrmse 
idx = find(nrmse1 == min(nrmse1));
best_k_lam = k_vals(idx);

fprintf("Best value of k for Local Average Model: %d\n", best_k_lam);

% fit local average model with best value of k 
[~, ~, residuals_lam] = localfitnrmse(xV, tau, m , 1, best_k_lam); 

%--------------------------------------------------------------------------

% diagnose model suitability by examining if the residuals of the best
% fitted model come from a white noise process 

% maximum lag
maxtau = 20; 
% significance level for Pormanteau hypothesis testing 
alpha = 0.05;

% compute autocorrelation for lags 0, 1,...,max_tau
[~, ~, ~] = autocorrelation(residuals_lam, maxtau, 1, "Autocorrelation of residuals of Local Average Model");

% perform Portmanteau hypothesis testing 
[h_res, p_res, Q_res, xaut_res] = portmanteauLB(residuals_lam, maxtau, alpha, "on residuals of Local Average Model");

%--------------------------------------------------------------------------

% PREDICTION WITH NON-LINEAR MODEL (step 8)

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

% we are going to use the local linear model for predictions since the
% residuals of the local average model are not iid

% predict with the best local linear model
[~, pred_llpV] = localpredictnrmse(x_train, N - n1, tau, m, Tmax, best_k_llp, m);


% pred_llpV is a matrix whose first column contains the test set time indices,
% and columns 2:end are the T-step ahead predictions for the stationary series.
% To recover the full (non-stationary) predictions, add back the seasonal component.
% Use extrapolated seasonal component, estimated from the training set.
predictions = pred_llpV;
predictions(:,2:end) = pred_llpV(:,2:end) + repmat(extrap_seasonalV, 1, Tmax);

% plot predicted along with true values in year 2007 for 1 step ahead
% predictions
figure; 
plot(adjusted_solar_energy(n1+1:end)); 
hold on; 
plot(predictions(:,2));
hold off;
title("LLP model predictions", 'FontSize', 15);
legend("actual time-series", "predictions", 'FontSize', 15);

% calculate NRMSE with respect to the number of steps ahead, on the
% predictions of the original non-stationary time-series 
nrmseV_llp = zeros(Tmax, 1);
for T = 1:Tmax
    % for each target index in the predictions, the true value for T-step ahead is:
    % adjusted_solar_energy(test_index + T)
    trueVals = adjusted_solar_energy((n1+T):end);

    % the T-step ahead predictions are in column (T+1) of predictions_local.
    predVals = predictions(T:end, T+1);
    nrmseV_llp(T) = nrmse(trueVals, predVals);
end

% plot NRMSE values with respect to the number of steps ahead 
figure; 
plot(1:Tmax, nrmseV_llp, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel("T", 'FontSize', 15); 
ylabel("NRMSE(T)", 'FontSize', 15);
title("NRMSE(T) with LLP model, K = " + int2str(best_k_llp), 'FontSize', 15); 

%--------------------------------------------------------------------------