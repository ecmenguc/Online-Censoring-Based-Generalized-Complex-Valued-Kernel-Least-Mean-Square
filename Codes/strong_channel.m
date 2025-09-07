% =========================================================================
% MATLAB Code for:
% "A Novel Online Censoring-Based Generalized Complex-Valued Kernel LMS"
% Authors: Buket Çolak Güvenç and Engin Cemal Mengüç
% Journal: IEEE Signal Processing Letters
%
% This script compares gCKLMS and the proposed OC-gCKLMS algorithm.
% It reproduces the main experiments of the paper.
% =========================================================================

%% Experiment Settings
% Signal: unbalanced QAM with amplitudes
%   sigOut1 = .2
%   sigOut2 = 1
% Channel: Strong nonlinear channel (5 taps, Bouboulis 2012)
% Trials: 30   (set to 100 for final experiments)
% SNR: 15 dB
% N = 20000 samples
% Outputs: MSE, circularity, update ratio, and runtime comparisons

clc
clear all
close all
%semilla=9;
%rng(semilla)


% Amplitudes.
sigOut1=.2; %Real
sigOut2=1; %Imag

% Samples
N=20000; 

% Preallocation for error storage
e1T=zeros(1,N-5);
e1T=e1T;
e1Tr=e1T;
e1Ti=e1T;
e2T=e1T;
e2Tr=e1T;
e2Ti=e1T;
pruebas=30; % Set to 100 for final test

for t=1:pruebas
    disp('-------------------------------------------------------------------');
    disp(['TEST NUMBER : ', num2str(t)]);
    disp('-------------------------------------------------------------------');
    %% Output - signal to be learned

     %% Generate input signal (unbalanced QAM)
    y=(sign(randn(1,N)))*sigOut1+1j*(sign(randn(1,N)))*sigOut2;

 
    % Channel Boboulis 2012 --> Strong Nonlinear Channel with 5 taps
    h = [(-0.9 + 0.8i), (0.6 - 0.7i), (-0.4 + 0.3i), (0.3 + 0.2i), (-0.1 - 0.2i)]; 
    x = h * [y(6:end); y(5:end-1); y(4:end-2); y(3:end-3); y(2:end-4)];
    x = [0 0 0 0 0 x];
    % Nonlinearity
    x = x + (0.2 + 1j * 0.25) * x.^2 + (0.08 + 1j * 0.09) * x.^3; % Strong Nonlinear Channel 



   %% Add noise
    SNR=15;
    vn=var(y)/(10^(SNR/10));
    noise=sqrt(vn/2)*(randn(1,length(x))+1j*randn(1,length(x)));
    x=x+noise;
    disp(['SNR =', num2str(SNR), 'dB']);
    disp(['SNR_numerical =', num2str(10*log10(var(y)/var(noise))), 'dB']);
    disp('-------------------------------------------------------------------');

    % Filter
    L=5; % length
    D=2; % delay

     z=[x(1:N-L).' x(2:N-L+1).' x(3:N-L+2).' x(4:N-L+3).' x(5:N-L+4).'];
     d=y(L-D:N-D-1).';

     % Kernel parameters
     bestSig1=0.59;
     bestSig2=1.63;

      % Test
      Nz=size(z,1);

     
    %% Compute kernel distance matrix (symmetric)
    XX = zeros(Nz, Nz, 'single');
    for i = 1:Nz
        % Only calculate the part needed for each row of XX
        for j = i:Nz
            XX(i, j) = sum(abs(z(i, :) - z(j, :)).^2); % Calculate row-wise differences for symmetry
            XX(j, i) = XX(i, j); % Exploit symmetry
        end
    end


      
    %% gCKLMS algorithm
    g_K1=exp(-XX/(bestSig1^2)); 
    g_K2=exp(-XX/(bestSig2^2));
    e1=zeros(1,Nz);
    g_a=e1;
    g_b=e1;
    g_mu1=0.5;
    g_mu2=g_mu1;
    e1(1)=d(1);
    g_a(1)=2*g_mu1*real(e1(1))/abs(g_K1(1,1)+1e-6);
    g_b(1)=1i*2*g_mu2*imag(e1(1))/abs(g_K2(1,1)+1e-6);
    tic
    for k1=2:Nz
       e1(k1)=d(k1)-(g_a(1:k1-1)*g_K1(1:k1-1,k1)+g_b(1:k1-1)*g_K2(1:k1-1,k1));
       g_a(k1)=2*g_mu1*real(e1(k1))/abs(g_K1(k1,k1)+1e-6);
       g_b(k1)=1i*2*g_mu2*imag(e1(k1))/abs(g_K2(k1,k1)+1e-6);
    end
    err1i=(imag(e1).^2);
    err1r=(real(e1).^2);
    err1=err1i+err1r;
    e1Ti=e1Ti+err1i;
    e1Tr=e1Tr+err1r;
    e1T=e1T+err1;
    
    t1(t) = toc;
      

    %% Proposed OC-gCKLMS algorithm
    K1=exp(-XX/(bestSig1^2));
    K2=exp(-XX/(bestSig2^2));

    e2=zeros(1,Nz);
    mu1=0.5;
    mu2=mu1;
    e2(1)=d(1);

    a(1)=2*mu1*real(e2(1))/abs(K1(1,1)+1e-6);
    b(1)=1i*2*mu2*imag(e2(1))/abs(K2(1,1)+1e-6);
    num_inf_data = Nz-0.5*Nz;
    Pce = (Nz-num_inf_data)/Nz;
   % tau = sqrt(log(1/(1-Pce)));

    beta = 0.95;
    var_e = abs(e2(1))^2;
    a = [];
    b = [];
    update = 0;
    idx = []; 
    tic;
    mu_tau = 0.1;
    tau = 0;

    for k1 = 2:Nz
        if ~isempty(idx)
            e2(k1) = d(k1) - (a * K1(idx, k1) + b * K2(idx, k1));
        else
            e2(k1) = d(k1);
        end
  
        var_e = beta * var_e + (1 - beta) * abs(e2(k1))^2;
        phi = tau * sqrt(var_e);

        if abs(e2(k1)) >= phi
            a = [a, 2 * mu1 * real(e2(k1)) / abs(K1(k1, k1) + 1e-6)];
            b = [b, 1i * 2 * mu2 * imag(e2(k1)) / abs(K2(k1, k1) + 1e-6)];
            tau = tau + mu_tau*Pce;
            idx = [idx, k1];
            update=update+1;
        else
            a = a;
            b = b;
            tau = tau - mu_tau*(1-Pce);
        end
    end
    err2i=(imag(e2).^2);
    err2r=(real(e2).^2);
    err2=err2i+err2r;
    e2Ti=e2Ti+err2i;
    e2Tr=e2Tr+err2r;
    e2T=e2T+err2;
  
    Pup(t)=update;
   
    t2(t)= toc;
end


%% Averaging results
Pce
e1Tr=e1Tr/pruebas;  %gCKLMS
e1Ti=e1Ti/pruebas;
e1T=e1T/pruebas;


e2Tr=e2Tr/pruebas;   %OC-gCKLMS
e2Ti=e2Ti/pruebas;
e2T=e2T/pruebas;

%% Performance metrics
update_OC_gCKLMS    = (sum(Pup)/pruebas)/Nz   %OC_gCKLMS
actual_update       = 1-Pce   


t_gCKLMS            = sum(t1)/pruebas
tOC_gCKLMS          = sum(t2)/pruebas


reduction_in_time = t_gCKLMS - tOC_gCKLMS;
reduction_in_time_percent = (reduction_in_time / t_gCKLMS) * 100;
reduction_in_time_percent
fprintf('OC stratejisi süreyi %.2f%% azaltmıştır.\n', reduction_in_time_percent);

%% MSE comparison
K=50;
gCKMLS_MSE         = mean(10*log10(sum_mean_val2(e1T(end-1000:end),1000)))
OC_gCKMLS_MSE         = mean(10*log10(sum_mean_val2(e2T(end-1000:end),1000)))
mse_change_percent = ((gCKMLS_MSE - OC_gCKMLS_MSE) / gCKMLS_MSE) * 100;


%% Plot MSE
K = 50;

figure(1);
plot(10*log10(sum_mean_val2(e1T,K)),'c','LineWidth',2);  % gCKLMS için MSE
hold on;
plot(10*log10(sum_mean_val2(e2T,K)),'k','LineWidth',2);  % OC-gCKLMS için MSE
legend('gCKLMS','OC-gCKLMS');
title('MSE Comparison with Online Censoring');






















