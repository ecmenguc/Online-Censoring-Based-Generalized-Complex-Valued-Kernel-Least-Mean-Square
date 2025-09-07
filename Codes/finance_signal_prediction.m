%% =========================================================
%% Code for: A Novel Online Censoring-Based Generalized 
%% Complex-Valued Kernel Least Mean Square
%% Authors: Buket Çolak Güvenç and Engin Cemal Mengüç
%% Published in: IEEE Signal Processing Letters
%% =========================================================
%% This code implements Experiment B: Financial Signal Prediction
%% It compares gCKLMS (v ≠ 0) and OC-gCKLMS (v ≠ 0) algorithms.
%% =========================================================
clc; clear; close all;

%% ==================== Data Loading and Preprocessing ====================
veri = xlsread('JPM_2024.xlsx'); 
high = veri(:, 1);          
closing = veri(:, 2);       
stock = closing + 1i * high; 
N = length(stock);          
d = transpose(stock);       
d = d / max(abs(d));        

delay = 1;        
x = d(delay:end); 
y = d;

%% ==================== Parameters ====================
pruebas = 1;     % Number of trials
L = 5;           % Filter length
bestSig3 = 1.73;
bestSig4 = 0.58;
bestSig5 = 1.11;
bestv = 0.18;
mu = 0.2;        % Step-size for gCKLMS (v≠0)
mu1 = 0.2;       % Step-size for OC-gCKLMS (v≠0)
mu2 = mu1;
beta = 0.95;
mu_tau = 0.1;

%% Initialize error tracking variables
e_gCKLMS_vT = zeros(1, N - L);
e_OC_gCKLMS_vT = zeros(1, N - L);

%% ==================== Trials Loop ====================
for t = 1:pruebas
    fprintf('--- Trial %d ---\n', t);

    % Input-output pairs (time-delayed vectors)
    z = [x(1:N-L).' x(2:N-L+1).' x(3:N-L+2).' x(4:N-L+3).' x(5:N-L+4).'];
    Nz = size(z, 1);

    %% ==================== Kernel Matrix Calculation ====================
    XX = zeros(Nz, Nz, 'single');
    for i = 1:Nz
        for j = i:Nz
            XX(i,j) = sum(abs(z(i,:) - z(j,:)).^2);
            XX(j,i) = XX(i,j);
        end
    end

    %% ==================== Kernel Matrices ====================
    K3 = exp(-XX / (bestSig3^2)); 
    K4 = exp(-XX / (bestSig4^2));
    K5 = exp(-XX / (bestSig5^2));

    %% ==================== CASE 1: gCKLMS (v ≠ 0) ====================
    e_gCKLMS_v = zeros(1, N-L);
    y_gCKLMS_v = e_gCKLMS_v;
    a_gCKLMS_v = e_gCKLMS_v;
    b_gCKLMS_v = e_gCKLMS_v;
    c_gCKLMS_v = e_gCKLMS_v;

    e_gCKLMS_v(1) = y(1);
    y_gCKLMS_v(1) = y(1);
    a_gCKLMS_v(1) = 2*mu*real(e_gCKLMS_v(1)) / (abs(K3(1,1)) + 1e-6);
    b_gCKLMS_v(1) = 2*1i*mu*imag(e_gCKLMS_v(1)) / (abs(K4(1,1)) + 1e-6);
    c_gCKLMS_v(1) = bestv*mu*(imag(e_gCKLMS_v(1)) + 1i*real(e_gCKLMS_v(1))) / (abs(K5(1,1)) + 1e-6);

    tic
    for k1 = 2:N-L
        y_gCKLMS_v(k1) = a_gCKLMS_v(1:k1-1)*K3(1:k1-1,k1) + ...
                         b_gCKLMS_v(1:k1-1)*K4(1:k1-1,k1) + ...
                         c_gCKLMS_v(1:k1-1)*K5(1:k1-1,k1);
        e_gCKLMS_v(k1) = y(k1) - y_gCKLMS_v(k1);
        a_gCKLMS_v(k1) = 2*mu*real(e_gCKLMS_v(k1)) / (abs(K3(k1,k1)) + 1e-6);
        b_gCKLMS_v(k1) = 2*1i*mu*imag(e_gCKLMS_v(k1)) / (abs(K4(k1,k1)) + 1e-6);
        c_gCKLMS_v(k1) = bestv*mu*(imag(e_gCKLMS_v(k1)) + 1i*real(e_gCKLMS_v(k1))) / (abs(K5(k1,k1)) + 1e-6);
    end
    t_gCKLMS_v = toc;
    e_gCKLMS_vT = e_gCKLMS_vT + (real(e_gCKLMS_v).^2 + imag(e_gCKLMS_v).^2);

    %% ==================== CASE 2: OC-gCKLMS (v ≠ 0) ====================
    e_OC_gCKLMS_v = zeros(1, N-L);
    y_OC_gCKLMS_v = zeros(1, N-L);

    a_OC = [];
    b_OC = [];
    c_OC = [];
    idx = [];
    update = 0;
    tau = 0;
    var_e = abs(e_OC_gCKLMS_v(1))^2;

    e_OC_gCKLMS_v(1) = y(1);
    y_OC_gCKLMS_v(1) = y(1);

    num_inf_data = N-L - 0.5*(N-L);
    Pce = (N-L - num_inf_data)/(N-L);

    tic
    for k1 = 2:N-L
        if ~isempty(idx)
            y_OC_gCKLMS_v(k1) = a_OC*K3(idx,k1) + b_OC*K4(idx,k1) + c_OC*K5(idx,k1);
            e_OC_gCKLMS_v(k1) = y(k1) - y_OC_gCKLMS_v(k1);
        else
            e_OC_gCKLMS_v(k1) = y(k1);
        end

        var_e = beta*var_e + (1-beta)*abs(e_OC_gCKLMS_v(k1))^2;
        phi = tau*sqrt(var_e);

        if abs(e_OC_gCKLMS_v(k1)) >= phi
            a_OC = [a_OC, 2*mu1*real(e_OC_gCKLMS_v(k1))/(abs(K3(k1,k1))+1e-6)];
            b_OC = [b_OC, 1i*2*mu2*imag(e_OC_gCKLMS_v(k1))/(abs(K4(k1,k1))+1e-6)];
            c_OC = [c_OC, bestv*mu*(imag(e_OC_gCKLMS_v(k1)) + 1i*real(e_OC_gCKLMS_v(k1)))/(abs(K5(k1,k1))+1e-6)];
            tau = tau + mu_tau*Pce;
            idx = [idx, k1];
            update = update + 1;
        else
            tau = tau - mu_tau*(1-Pce);
        end
    end
    t_OC_gCKLMS_v = toc;
    e_OC_gCKLMS_vT = e_OC_gCKLMS_vT + (real(e_OC_gCKLMS_v).^2 + imag(e_OC_gCKLMS_v).^2);

    Pup(t) = update;
end

%% ==================== Performance Metrics ====================
% MSE
K = 50;
MSE_gCKLMS_v_dB = 10*log10(sum_mean_val2(e_gCKLMS_vT,K));
MSE_OC_gCKLMS_v_dB = 10*log10(sum_mean_val2(e_OC_gCKLMS_vT,K));

% Time reduction
reduction_time = (t_gCKLMS_v - t_OC_gCKLMS_v)/t_gCKLMS_v*100;
fprintf('OC-gCKLMS (v≠0) reduces computation time by %.2f%%\n', reduction_time);

% Update statistics
update_OC = mean(Pup)/Nz;
actual_update = 1-Pce;

%% ==================== Plots ====================
figure;
plot(MSE_gCKLMS_v_dB,'b','LineWidth',2); hold on;
plot(MSE_OC_gCKLMS_v_dB,'r','LineWidth',2);
legend('gCKLMS (v≠0)','OC-gCKLMS (v≠0)');
title('MSE Comparison');
grid on;

figure;
plot(abs(d(delay:end)),'k','LineWidth',2); hold on;
plot(abs(y_gCKLMS_v),'b','LineWidth',2);
plot(abs(y_OC_gCKLMS_v),'r','LineWidth',2);
legend('Actual Data','gCKLMS (v≠0)','OC-gCKLMS (v≠0)');
title('Algorithm Outputs vs. Financial Data');
xlabel('Time'); ylabel('Magnitude');
grid on;
