 import mne  
 import numpy as np  
 import joblib  
 from scipy import stats  
 from scipy.signal import welch  
 import matplotlib.pyplot as plt  
 import seaborn as sns  
 from tqdm import tqdm  
   
 # Load the EDF file and the trained model  
 print("Loading EDF file and model...")  
 raw = mne.io.read_raw_edf('chb02_19.edf', preload=True, verbose=False)  
 data = raw.get_data()  
 sfreq = raw.info['sfreq']  
 model = joblib.load('seizure_predictor_model.joblib')  
   
 def extract_features_for_prediction(data, sampling_freq=256, window_duration=60):  
     window_size = int(window_duration * sampling_freq)  
     features = {}  
     for channel in range(data.shape[0]):  
         channel_data = data[channel, :window_size]  
         # Time-domain features  
         features['mean_abs_ch' + str(channel)] = np.mean(np.abs(channel_data))  
         features['std_ch' + str(channel)] = np.std(channel_data)  
         features['kurtosis_ch' + str(channel)] = stats.kurtosis(channel_data)  
         features['skew_ch' + str(channel)] = stats.skew(channel_data)  
         # Frequency-domain features  
         freqs, psd = welch(channel_data, fs=sampling_freq, nperseg=min(256, len(channel_data)))  
         features['peak_freq_ch' + str(channel)] = freqs[np.argmax(psd)]  
         features['peak_power_ch' + str(channel)] = np.max(psd)  
         # Band powers  
         delta_mask = (freqs >= 0.5) & (freqs <= 4)  
         theta_mask = (freqs >= 4) & (freqs <= 8)  
         alpha_mask = (freqs >= 8) & (freqs <= 13)  
         beta_mask = (freqs >= 13) & (freqs <= 30)  
         features['delta_power_ch' + str(channel)] = np.sum(psd[delta_mask])  
         features['theta_power_ch' + str(channel)] = np.sum(psd[theta_mask])  
         features['alpha_power_ch' + str(channel)] = np.sum(psd[alpha_mask])  
         features['beta_power_ch' + str(channel)] = np.sum(psd[beta_mask])  
     return features  
   
 # Set sliding window parameters  
 window_duration = 60  # seconds per prediction window  
 step_size = 30        # seconds; 50% overlap  
 window_samples = int(window_duration * sfreq)  
 step_samples = int(step_size * sfreq)  
 n_windows = (data.shape[1] - window_samples) // step_samples + 1  
   
 risk_scores = []  
 timestamps = []  # in minutes  
   
 print("Calculating risk scores over time...")  
 for i in tqdm(range(n_windows)):  
     start_idx = i * step_samples  
     end_idx = start_idx + window_samples  
     if end_idx > data.shape[1]:  
         break   
     window = data[:, start_idx:end_idx]  
     features = extract_features_for_prediction(window, sfreq)  
     risk_prob = model.predict_proba([list(features.values())])[0][1]  
     risk_scores.append(risk_prob * 100)  
     timestamps.append(i * step_size / 60)  # convert to minutes  
   
 # Create a dashboard visualization for the risk assessment  
 plt.style.use('default')  
 fig = plt.figure(figsize=(15, 12))  
   
 # Risk timeline with threshold markers and risk bands  
 plt.subplot(2, 1, 1)  
 plt.plot(timestamps, risk_scores, color='#2563EB', linewidth=2, marker='o', markersize=4)  
 plt.fill_between(timestamps, risk_scores, color='#2563EB', alpha=0.3)  
   
 # Adding risk level bands  
 plt.axhspan(0, 20, facecolor='green', alpha=0.2, label='Low Risk (<20%)')  
 plt.axhspan(20, 50, facecolor='yellow', alpha=0.2, label='Medium Risk (20-50%)')  
 plt.axhspan(50, 100, facecolor='red', alpha=0.2, label='High Risk (>50%)')  
   
 # Adding threshold lines  
 plt.axhline(20, color='black', linestyle='--', linewidth=1)  
 plt.axhline(50, color='black', linestyle='--', linewidth=1)  
   
 plt.title('Seizure Risk Assessment Over Time', pad=15, fontsize=20, fontweight='semibold', color='#171717')  
 plt.xlabel('Time (minutes)', labelpad=10, fontsize=16, color='#171717')  
 plt.ylabel('Risk Score (%)', labelpad=10, fontsize=16, color='#171717')  
 plt.legend()  
 plt.grid(True, axis='y', color='#F3F4F6')  
 plt.gca().set_axisbelow(True)  
 plt.gca().spines['top'].set_visible(False)  
 plt.gca().spines['right'].set_visible(False)  
   
 # Risk distribution histogram  
 plt.subplot(2, 1, 2)  
 sns.histplot(risk_scores, bins=30, color='#2563EB', alpha=0.6)  
 plt.axvline(np.mean(risk_scores), color='red', linestyle='--', label=f'Mean Risk: {np.mean(risk_scores):.1f}%')  
 plt.title('Distribution of Risk Scores', pad=15, fontsize=20, fontweight='semibold', color='#171717')  
 plt.xlabel('Risk Score (%)', labelpad=10, fontsize=16, color='#171717')  
 plt.ylabel('Frequency', labelpad=10, fontsize=16, color='#171717')  
 plt.legend()  
 plt.grid(True, axis='y', color='#F3F4F6')  
 plt.gca().set_axisbelow(True)  
 plt.gca().spines['top'].set_visible(False)  
 plt.gca().spines['right'].set_visible(False)  
   
 plt.tight_layout()  
 plt.show()  
   
 # Summary of risk assessment statistics  
 print("\nRisk Assessment Summary:")  
 print(f"Average Risk: {np.mean(risk_scores):.1f}%")  
 print(f"Maximum Risk: {np.max(risk_scores):.1f}%")  
 print(f"Minimum Risk: {np.min(risk_scores):.1f}%")  
 print(f"Standard Deviation: {np.std(risk_scores):.1f}%")  
   
 # Calculate and print percentage of time in each risk category  
 risk_array = np.array(risk_scores)  
 low_risk = np.mean(risk_array < 20) * 100  
 medium_risk = np.mean((risk_array >= 20) & (risk_array < 50)) * 100  
 high_risk = np.mean(risk_array >= 50) * 100  
   
 print("\nTime spent in risk categories:")  
 print(f"Low Risk (<20%): {low_risk:.1f}% of the time")  
 print(f"Medium Risk (20-50%): {medium_risk:.1f}% of the time")  
 print(f"High Risk (>50%): {high_risk:.1f}% of the time")  