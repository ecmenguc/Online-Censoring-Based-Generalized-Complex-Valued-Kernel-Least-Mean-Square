% =========================================================================
% MATLAB Code for:
% "A Novel Online Censoring-Based Generalized Complex-Valued Kernel LMS"
% Authors: Buket Çolak Güvenç and Engin Cemal Mengüç
% Journal: IEEE Signal Processing Letters
%
% Experiment B: A Random Process Filtered
% This script compares gCKLMS and the proposed OC-gCKLMS.
% =========================================================================

clc
clear all
close all

%% Random seed
semilla = 99;
rng(semilla)

%% Grid and process setup
T = 100;
Np = T^2;
limiteXmn = -5;
limiteXmx = 5;

Xr = linspace(limiteXmn,limiteXmx,T);
Xi = linspace(limiteXmn,limiteXmx,T);
Xmt = Xr'*ones(1,T) + 1i*ones(T,1)*Xi;

x = reshape(Xmt,Np,1);          
rndIDX = randperm(T^2);
x = x(rndIDX);

%% Channel / filter parameters
g1 = 3;
g2 = 0.5;
v1 = 2;
v2 = 1;

H1 = v1*exp(-(abs(Xmt).^2)/g1) + v2*1i*exp(-(abs(Xmt).^2)/g2);
H1 = H1/norm(H1);

%% Error accumulators
e2T = zeros(1,Np);
e4T = e2T;

%% Number of trials
pruebas = 1;

%% Loop over trials
for t = 1:pruebas
    disp('-------------------------------------------------------------------');
    disp(['TEST NUMBER : ', num2str(t)]);
    disp('-------------------------------------------------------------------');

    %% Generate random input
    sr = randn(T,T);
    Y = conv2(H1,sr,'same');

    %% Visualization of the process
    figure(1)
    surf(Xr',Xi',real(Y))
    title('Real part of the process')

    figure(2)
    surf(Xr',Xi',imag(Y))
    title('Imaginary part of the process')

    %% Desired output
    y = reshape(Y,Np,1);
    d = y(rndIDX);

    %% Additive noise
    SNR = 15;
    vn = var(d)/(10^(SNR/10));
    noise = sqrt(vn/2)*(randn(length(y),1)+1j*randn(length(y),1));
    y = d + noise;

    %% Kernel hyperparameters
    bestSig3 = 1.73;
    bestSig4 = 0.58;
    bestSig5 = 1.11;
    bestv   = 0.18;

    %% Step-sizes
    mu1 = 0.1;
    mu2 = 0.1;
    mu3 = 0.1;

    %% Kernel matrix
    X = kron(x,ones(1,Np));
    XX = abs(X-X.').^2;

    %% =========================
    %% CASE 2: gCKLMS (v ≠ 0)
    %% =========================
    K3 = exp(-XX/(bestSig3^2)); 
    K4 = exp(-XX/(bestSig4^2));
    K5 = exp(-XX/(bestSig5^2));

    e2 = zeros(1,Np);
    y3 = e2;
    a = e2;
    b = e2;
    c = e2;

    e2(1) = y(1);
    y3(1) = y(1);
    a(1) = 2*mu1*real(e2(1))/abs(K3(1,1)+1e-6);
    b(1) = 2*1i*mu2*imag(e2(1))/abs(K4(1,1)+1e-6);
    c(1) = bestv*mu3*(imag(e2(1))+1i*real(e2(1)))/abs(K5(1,1)+1e-6);

    tic
    for k1 = 2:Np
       y3(k1) = (a(1:k1-1)*K3(1:k1-1,k1) + ...
                 b(1:k1-1)*K4(1:k1-1,k1) + ...
                 c(1:k1-1)*K5(1:k1-1,k1));
       e2(k1) = y(k1) - y3(k1);
       a(k1) = 2*mu1*real(e2(k1))/abs(K3(k1,k1)+1e-6);
       b(k1) = 2*1i*mu2*imag(e2(k1))/abs(K4(k1,k1)+1e-6);
       c(k1) = bestv*mu3*(imag(e2(k1))+1i*real(e2(k1)))/abs(K5(k1,k1)+1e-6);
    end
    t2(t) = toc;
    e2T = e2T + (real(e2).^2 + imag(e2).^2);

    %% =========================
    %% CASE 4: OC-gCKLMS (v ≠ 0)
    %% =========================
    K6 = exp(-XX/(bestSig3^2)); 
    K7 = exp(-XX/(bestSig4^2));
    K8 = exp(-XX/(bestSig5^2));

    e4 = zeros(1,Np);
    e4(1) = d(1);

    a(1) = 2*mu1*real(e4(1))/abs(K6(1,1)+1e-6);
    b(1) = 1i*2*mu2*imag(e4(1))/abs(K7(1,1)+1e-6);
    c(1) = bestv*mu3*(imag(e4(1))+1i*real(e4(1)))/abs(K8(1,1)+1e-6);

    %% Online censoring parameters
    beta   = 0.95;
    var_e  = abs(e4(1))^2;
    a = []; b = []; c = [];
    update = 0;
    idx = [];
    mu_tau = 0.1;
    tau    = 0;
    Pce    = 0.5;

    tic
    for k1 = 2:Np
        if ~isempty(idx)
            e4(k1) = d(k1) - (a*K6(idx,k1) + b*K7(idx,k1) + c*K8(idx,k1));
        else
            e4(k1) = d(k1);
        end

        var_e = beta*var_e + (1-beta)*abs(e4(k1))^2;
        phi   = tau*sqrt(var_e);

        if abs(e4(k1)) >= phi
            a = [a, 2*mu1*real(e4(k1))/abs(K6(k1,k1)+1e-6)];
            b = [b, 1i*2*mu2*imag(e4(k1))/abs(K7(k1,k1)+1e-6)];
            c = [c, bestv*mu3*(imag(e4(k1))+1i*real(e4(k1)))/abs(K8(k1,k1)+1e-6)];
            tau = tau + mu_tau*Pce;
            idx = [idx, k1];
            update = update + 1;
        else
            tau = tau - mu_tau*(1-Pce);
        end
    end
    t4(t) = toc;
    e4T = e4T + (real(e4).^2 + imag(e4).^2);
    Pup(t) = update;
end

%% =========================
%% Post-processing
%% =========================

Pce

e2T = e2T / pruebas;
e4T = e4T / pruebas;

t_gCKLMS_v   = sum(t2)/pruebas;
tOC_gCKLMS_v = sum(t4)/pruebas;

reduction_in_time = t_gCKLMS_v - tOC_gCKLMS_v;
reduction_in_time_percent = (reduction_in_time / t_gCKLMS_v) * 100;
fprintf('OC stratejisi süreyi %.2f%% azaltmıştır.\n', reduction_in_time_percent)

update_OC_gCKLMS = (sum(Pup)/pruebas)/Np
actual_update     = 1 - Pce



%% MSE calculation
K = length(e2T);
MSE_gCKLMS_v_dB    = 10*log10(sum_mean_val2(abs(e2T).^2,K));
MSE_OC_gCKLMS_v_dB = 10*log10(sum_mean_val2(abs(e4T).^2,K));

%% Plot MSE comparison
figure(3)
plot(MSE_gCKLMS_v_dB,'k','LineWidth',2);
hold on
plot(MSE_OC_gCKLMS_v_dB,'r','LineWidth',2);
legend('gCKLMS','OC-gCKLMS');
xlabel('Number of samples');
ylabel('Averaged MSE (dB)');
grid on

%% Final MSE values
MSE_gCKLMS_v_dB_final    = MSE_gCKLMS_v_dB(end);
MSE_OC_gCKLMS_v_dB_final = MSE_OC_gCKLMS_v_dB(end);

fprintf('\n--- Ortalama MSE (dB, kümülatif ortalama ile) ---\n');
fprintf('gCKLMS    %.4f dB\n', MSE_gCKLMS_v_dB_final)
fprintf('OC-gCKLMS (v ≠ 0):   %.4f dB\n', MSE_OC_gCKLMS_v_dB_final)
