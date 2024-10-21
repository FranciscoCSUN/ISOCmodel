% 03/31/24  Changing Gaussian Channel
% Written by Joshua Mathis - Spring 2024
% Maintained by Francisco Hernandez - Fall 2024
%% 

clear variables;
close all;
clc;

% Basic Specs and Bit Generation

%% PRBS-31 generator
cinit   = randi([0,(2^31)-1],1,1);
bits    = 223 * 8;                  % RS Message length (223) * Bits per symbol (8)
prbs    = nrPRBS(cinit,bits);

freq    = 500E6;
fs      = 10*freq;                  % sampling rate (nyquist theorem)
T       = 1/freq;                   % sampling period
lambda  = 1550 * 10^-9;             % Wavelength in meters (1550 nm)
dist    = 1000 * 10^3;              % Distance in meters 10^3 = km (starlink’s longest is 5,300 km) fast vs far
PL      = fspl(dist,lambda);        % free space path loss in dB
PL_lin  = db2mag(PL);               % linear path loss
h       = 6.626E-34;                % Planck's constant
c       = 299792458;                % Speed of light
q       = 1.602E-19;                % Charge of an electron
k_bolt  = 1.38064852E-23;           % Boltzman's constant
Temp    = 273;                      % Temp in Kelvin ( space temperature is hard to find )


Rx_diameter    = 0.3;               % Receiver lens diameter ( model )

AC_amp  = 0.4;                      % AC amplitude 
DC_amp  = 1.8;                      % DC amplitude
I_AVG   = 12;                       % Average operating Current
eta_eff = 0.5;                      % Power conversion efficiency %50 efficiency
theta   = 0.0382*(pi/180);          % Full cone angle (Half power beam width) in Radians from F260SMA-1550, could have someone to mount properly with money 
W0      = 1.5E-3;                   % beam waist radius also on product page



% Trial Laser stats % other laser, trying out different values
% AC_amp = 3;
% DC_amp = 6;
% I_AVG  = 9;
% eta_eff= 0.19;


%% System Objects        
enc     = comm.RSEncoder(BitInput = true, CodewordLength = 255, MessageLength = 223); % RS bit Encoder object RS (255, 223)
dec     = comm.RSDecoder(BitInput = true, CodewordLength = 255, MessageLength = 223); % RS bit Decoder object RS (255, 223)
con_enc = comm.ConvolutionalEncoder;                                        % CON_Encoder object  
con_dec = comm.ViterbiDecoder('InputFormat','Hard');                        % CON_Decoder object     hard?
error   = comm.ErrorRate('ComputationDelay',3,'ReceiveDelay',34);           % Error checking object  
%hpFilt  = designfilt('highpassiir', 'FilterOrder', 8, ...                   
%            'HalfPowerFrequency', 1e6, 'SampleRate', fs);                  % Filter Object design filter function in matlab, high pass, 
dcBlock = dsp.DCBlocker('Algorithm','Subtract mean');                       % DC Block Object


%% Encode Message
con_data = con_enc(prbs);                   % 50 overhead, 1 message bit , 2 overhead bits
enc_data = step(enc,con_data);              % 4080? same ratio k to n ratio 223 to 255 is same as 3568 is to 4080
                                            % why does the convolutional encoder triple the number of bits

%% Modulate the Encoded Message
num_T = length(enc_data)/2;                 % number of needed periods for encoded message
t   = linspace(0,num_T * T,1000*num_T);     % 500 x more elements for enc data 500 per symbol
sig = zeros(length(t),1)';

doub_enc_data = double(enc_data);
% On / off Keying modulation happens here , with DC offest 
for i = 1 : length(doub_enc_data) % each data point for encoded data set
    for i2 = 1 + ((length(t)/length(doub_enc_data))*(i-1)) : (length(t)/length(doub_enc_data))*i % 5000 / 500  1 500
    if doub_enc_data(i) == 0
        sig(i2) = -AC_amp + DC_amp;
    elseif doub_enc_data(i) == 1
        sig(i2) = AC_amp + DC_amp;
    end
    end
end

% plot modulated electrical signal
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(t,sig)
hold on;
yline(DC_amp,'--','Color','g');
hold off;
xlim([0 10*T])
ylim([min(sig)-(abs(min(sig)*0.1)) max(sig)+(abs(max(sig)*0.1))]);
title('Modulated Signal at TX')

%% Convert Electrical Signal to Optical Power
sig_pwr = sig .* I_AVG;                                 % P = VI Using average power                    % signal is an amplitude times current , give average power
sig_opt = sig_pwr .* eta_eff;                           % convert to optical power using conversion     % signal * power efficiency -> represents conversion from electrical power to optical power
                                                        % efficiency this is from the HHL1550 Laser
                                                        % from Seminex

% Display Electrical Power signal
% figure;
% plot(t,sig_pwr)
% xlim([0 10*T])
% ylim([min(sig_pwr)-0.5 max(sig_pwr)+0.5])

% Display Optical Power signal
% figure;
% plot(t,sig_opt)
% xlim([0 10*T])
% ylim([min(sig_opt)-0.5 max(sig_opt)+0.5])

%% Optical Transmitter Gain - going through the lenses          ( dbW, at 0dBW - 0dBW is 1 watt, 0dBm is one milliWatt )
sig_opt_dB   = pow2db(sig_opt);                         % Optical Signal in dBW                                     gain in db adds, gain in linear multiplies ( dBW vs dBm 
Omega        = 2*pi*(1 - cos(theta/2));                 % Solid angle of beam in Stradians                          radians for spheres - area of the cone, 4pi stradians = theta from lens model
Directivity  = (4*pi)/Omega;                            % Directivity = Antenna gain with 100% antenna efficiency   directivity         - gain for antennas - relative value of how directive the thing is 4pi is full sphere stradians, laser propagated in a sphere, directivity of 1 in linear in db is 0, same as the sphere so its just one, not directive at all, 
Gain_dB      = pow2db(Directivity);                     % Optical Gain                                              db gain for antenna eff * directivity, treat laser like antenna for directivitty to dB, already accounted for efficiency in 95 convert directivety in decibels
Radiance     = ((DC_amp*I_AVG)/2)/Omega;                % TX Radiant Intensity assuming half power across beam      <<useless>> could use in a paper reference 

sig_opt_gain = sig_opt_dB + Gain_dB;                    % Signal after Optical Gain                                 total gain

M_squared         = (pi * (theta/2) * W0)/lambda;       % Beam quality                                              chatpt, may become useful later
received_radius   = dist * tan(theta);                  % distance, tan, theta, beam diverges by how much
Irradiance        = (mean(sig_opt)/2) / ...
                    (pi * received_radius^2);           % Received Irradiance (power per unit area)                 power density when received, snr ratio

%% Transmission Channel
Path_Loss     = fspl(dist,lambda);                      % free space path loss over distance dist is dB                 fspl is negative gain
sig_opt_atten = sig_opt_gain - Path_Loss;               % attenuate the optical signal

% Display Optical Power signal                          % uncomment to display
% figure;
% subplot(2,1,1);                                       % might not need
% plot(t,sig_opt_dB);
% xlim([0 10*T]);
% ylim([min(sig_opt_dB)-0.5 max(sig_opt_dB)+0.5])
% subplot(2,1,2);
% plot(t,sig_opt_atten);
% xlim([0 10*T]);
% ylim([min(sig_opt_atten)-0.5 max(sig_opt_atten)+0.5])

%% Receiver Optical Gain
A_r            = (Rx_diameter/2)^2 * pi;            % area of receiving lens
Gr             = (4 * pi * A_r) / lambda^2;         % gain of receiving lens - directivity
Gr_dB          = pow2db(Gr);                        % conversition to db
sig_opt_rx_dB  = sig_opt_atten + Gr_dB;             % added recieiver gain 
sig_opt_rx     = db2pow(sig_opt_rx_dB);             % db to power signal

%% Optical filter for Backround light
% 1550nm CWL, OD 4 Ultra Narrow Filter (OD = Optical Density) from Edmund
% Optics
Transmission = 0.95;   % Transmission percentage at center wavelength                                   % band pass filter
Filter_BW    = 1569E-9 - 1533E-9; % Bottom of upper block range minus the top of the lower block range  % bandwidth from lens, avoid getting light noise 1533nm - 1569nm filter
sig_opt_filt = sig_opt_rx * Transmission;                                                               % send signal thougth lens with multiplication of center wavelength

%% Optical Power to Current (electrical signal)
% Photodetector chosen is the UPD-70-UVIR-P from AlphaLas
eta_qe   = 0.80;                                            % Quantum efficieny
i_rx     = ((eta_qe * q * lambda)/(h * c)) * sig_opt_filt;  % Electrical current generated by PD 
i_rx_AC  = dcBlock(i_rx')';                                 % dc block receiveds

% Plot Received AC signal without DC
% figure;
% plot(t,i_rx_AC);
% xlim([0 10*T]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% SWITCHING TO dBm!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_rx     = max(i_rx_AC) .* (max(i_rx_AC) .* 50);    % Convert into electrical power for 50 Ohm input of RF amplifier & photodiode creates current look into photodiode, creates current, how to process? circuit diagram of how you could use it
P_rx_dB  = pow2db(P_rx) + 30;                       % This is in dBm the +30 turns it from dBW to dBm  from dbw to dbm, adding 10db is 10 linear ( 1 W = 0dBw -> +30dB) 10^3 -> 30dB

% Plot the power in dBm
% figure;
% plot(t,P_rx_dB);
% xlim([0 10*T]);

% Amplify the signal to process
P_rx_amp_dB = P_rx_dB + 80.5;                       % Amplifying to 1dBm based upon the specs of ADC32RF52 frim Texas Instruments getting -79.5 dBm signal , 1 - 80.5 = -79.5
P_rx_amp = db2pow(P_rx_amp_dB -30);                 % signal tx , lose signal mush, have this much gain % db back to linear, subtract dbm to dbW
% Plot amplified received signal
% figure;
% plot(t,P_rx_amp)
% xlim([0 10*T])
% ylim([min(P_rx_amp)-(abs(min(P_rx_amp)*0.1)) max(P_rx_amp)+(abs(max(P_rx_amp)*0.1))])


%% Calculate the Noise Power
BW_elec     = 15E6;  % BW of BPF-B503+                                          % bandwidth of band pass filter
Noise_floor = 10 * log10(k_bolt * BW_elec * Temp) + 30 + 80.5;                  % noise power Thermal Noise floor in dBm   thermal noise floor 80.5 gain added to signal from imaginary set of amplifiers, if gain is given so signal, noise floor grows with gain so add 80.5

% Shot noise
shot_N      = 2 * q * mean(i_rx) * BW_elec;                                     % noise is the hardest part Poisson distribution
shot_N_dB   = pow2db(shot_N) + 30 + 80.5;                                       % noise power dBm and amplified

I_dark      = 0.8E-9;                                                           % Dark current of the photodiode % photodiode is an active component, a dark room will still create a tiny amount of current
%Dark_N      = pow2db(I_dark * abs((I_dark / 50))) + 30 + 80.5;                 % Dark noise based upon photodetector datasheet
Dark_N      = 2 * q * I_dark * BW_elec;
Dark_N_dB   = pow2db(Dark_N) + 30 + 80.5; % dBm and amplified

NF_amp_dB   = 0.4;      % Linear minimum noise figure of each LNA = no specific low noise amplifier % find a real LNA working at frequency, ensure noise figure matches up
NF_amp      = db2pow(NF_amp_dB);
Gass_dB     = 27.5;     % associated gain in db of each LNA
Gass        = db2pow(Gass_dB); % into linear
Amp_N       = pow2db( NF_amp + ((NF_amp-1) / Gass) + ((NF_amp - 1)/ (Gass^2))); % noise Figure from three stages of amplifiers every part in chain contributes to noise, noise of preceding is divided by gain of other
Amp_NP      = Noise_floor + Amp_N;                                              % added to noise floor, where is the noise at

BG_N        = Noise_floor + 3;   % backround noise placeholder                  % add 3db , doubles power, as a placeholder, not sure how to calculate, find out

Total_NF    = db2pow(shot_N_dB - 30) + db2pow(Dark_N_dB - 30) ...               % add total noise floor, shot, dark, back
            + db2pow(Amp_NP - 30) + db2pow(BG_N - 30);                          % convert to db
Total_NP_dB = pow2db(Total_NF) + 30;                                            % Total Noise power in dBm                   % into dBm

%% Calculate the SNR

P_sig    = db2pow(P_rx_amp_dB - 30 - 11); % Currently decreasing SNR by 10dB    % power of signal to linear (dBm to dBW -> -30) (-11) is to adjust for SNR, that 11 is artificial 
P_noise  = db2pow(Total_NP_dB - 30);                                            % total noise power conferted from linar to db
SNR      = P_sig / P_noise;
SNR_dB   = pow2db(SNR);

%% Modify the current signal
I_new_max  = sqrt(P_sig/50);                                % current, 
I_new_sig   = (i_rx_AC ./ abs(i_rx_AC)) .* I_new_max;       % signal after dc block, ac signal/ abs of itself, will be +1 or -1 , power of signal / 50  for square wave and current
% Signal_Plotter(t,I_new_sig,T);

% subplot(2,2,2)
figure;
plot(t*1E9,I_new_sig*1E3,'LineWidth',3);
% xlim([0 10*T]);
ylim([min(I_new_sig*1E3)-(abs(min(I_new_sig*1E3)*0.1)) max(I_new_sig*1E3)+(abs(max(I_new_sig*1E3)*0.1))]);
% title('Modulated Signal at RX');
xlim([0 10000000000*T]);
xlabel('Time (ns)', 'FontSize', 30);
ylabel('Current (mA)', 'FontSize', 30);
set(gca, 'LineWidth', 1);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 3.5 2.8]);
ax = gca; % Current axes
ax.FontSize = 30;


%% Add Noise to Signal

% AWGN object
Gaus  = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR',SNR_dB);

% Rx_Noisy = Gaus(P_rx_amp);
% Rx_Noisy = Gaus(I_new);
Rx_Noisy = awgn(I_new_sig,SNR_dB,P_rx_amp_dB-30);

% Signal_Plotter(t,Rx_Noisy,T);

%subplot(2,2,3)
figure;
plot(t*1E9,Rx_Noisy*1E3);
xlim([0 10000000000*T]);
ylim([min(Rx_Noisy*1E3)-(abs(min(Rx_Noisy*1E3)*0.1)) max(Rx_Noisy*1E3)+(abs(max(Rx_Noisy*1E3)*0.1))]);
xlabel('Time (ns)', 'FontSize', 30);
ylabel('Current (mA)', 'FontSize', 30);
set(gca, 'LineWidth', 1);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 3.5 2.8]);
ax = gca; % Current axes
ax.FontSize = 30; % Choose a font size that is appropriate
% title('Signal with Noise');

%% Demodulate the signal into Binary
T_bit = length(t)/length(doub_enc_data);                % Time steps per bit
Rx_Binary_enc = zeros(1,length(doub_enc_data));
for i = 1 : length(doub_enc_data)
    if mean(Rx_Noisy(1, 1+(T_bit*(i-1)):(T_bit*i))) > 0
        Rx_Binary_enc(i) = 1;
    else
        Rx_Binary_enc(i) = 0;
    end
end


Rx_Binary_enc = Rx_Binary_enc';
%             % Checking for errors
%             rec_bits(1) = rec_bits(1)+2;
%             rec_bits(2) = rec_bits(2)+2;
%             rec_bits(3) = rec_bits(3)+2;
Num_of_code_Errors = 0;  % Initialize tracker for number of errors
for i = 1:length(enc_data)
    if enc_data(i) ~= Rx_Binary_enc(i)
        Num_of_code_Errors = Num_of_code_Errors + 1;
    end
end

%% Decode the ECC
dec_bits = dec(Rx_Binary_enc);
con_bits = con_dec(dec_bits);
            
%             % Add in errors to test counter
%             con_bits(100) = ~con_bits(100);
%             con_bits(200) = ~con_bits(200);
%             con_bits(300) = ~con_bits(300);

%% Bit error checker
Num_of_Errors = 0;                              % Initialize tracker for number of errors
ticker = 0;
con_bits = logical(con_bits);
errors = error(prbs,con_bits);
con_bits = con_bits(35:bits,1);
for i = 1:bits-34                               % viterbi decoder loss is 34
    if prbs(i) ~= con_bits(i)
        Num_of_Errors = Num_of_Errors + 1;
    end
    ticker = ticker + 1;
end

BER = Num_of_Errors / length(prbs);


% this is a multimode laser
% single mode - less powerful than multimode, impact of communication?
% multimode -
% if you don't want to bother, keep same laser
% free space comm use an application for multimode
% research single vs multimode
% This is the HHL-1550-6-95 Laser from SemiNex ( short wave infrared, for good SNR performance in atmosphere ) 


% On datasheet, graph 12 Amps gives power and voltage, DC voltage will set to something, ac message signal varies that
% more power, but don't damage
% thats why dc - 1.8 and ac is 0.4, operating at the upper end