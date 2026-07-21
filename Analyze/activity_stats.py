import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import time

def export_activity_vs_plasticity_to_excel(go_activity, gr_activity, plasticity_arrays, 
                                           output_filename="analysis_results.xlsx",
                                           plasticity_type='IE', array_names=None):
    """
    Optimized function to calculate the relationship between GO/GR cell activity 
    and one or more plasticity arrays for massive cell populations (1M+).
    """
    start_time = time.time()
    
    # ==========================================
    # EXCLUDE CELL 0 FROM ANALYSIS
    # Using Ellipsis (...) targets the last dimension (cells) 
    # to safely drop index 0 regardless of if the array is 2D or 3D
    # ==========================================
    go_activity = go_activity[..., 1:]
    gr_activity = gr_activity[..., 1:]
    
    # Ensure plasticity_arrays is an iterable list
    if isinstance(plasticity_arrays, np.ndarray):
        plasticity_arrays = [plasticity_arrays]
        
    # Apply the same exclusion to all plasticity arrays
    plasticity_arrays = [p[..., 1:] for p in plasticity_arrays]
    # ==========================================
        
    if array_names is None:
        array_names = [f"Array_{i+1}" for i in range(len(plasticity_arrays))]
        
    num_trials = plasticity_arrays[0].shape[0]
    
    print("\n--- Processing Activity Data ---")
    
    # 1. Process Activity Data (Highly vectorized)
    # GO: Sum across bins (axis 1)
    t0 = time.time()
    go_spikes_per_trial = np.sum(go_activity, axis=1)
    go_spikes_per_second = go_spikes_per_trial / 5 # 5 seconds per trial
    mean_go = np.mean(go_spikes_per_second, axis=1) 
    var_go = np.var(go_spikes_per_second, axis=1)
    range_go = np.ptp(go_spikes_per_second, axis=1)
    print(f"GO activity processed in {time.time() - t0:.2f} seconds")
    
    # GR: Already formatted as (numTrial, numGR)
    t0 = time.time()
    gr_activity_per_second = gr_activity / 5 # 5 seconds per trial
    mean_gr = np.mean(gr_activity_per_second, axis=1)
    var_gr = np.var(gr_activity_per_second, axis=1)
    range_gr = np.ptp(gr_activity_per_second, axis=1)
    print(f"GR activity processed in {time.time() - t0:.2f} seconds")
    
    # 2. Initialize Dictionary for DataFrame
    trial_numbers = np.arange(1, num_trials + 1)
    data = {
        'Trial_Number': trial_numbers,
        'Mean_GO_Activity': mean_go,
        'Variance_GO_Activity': var_go,
        'Range_GO_Activity': range_go,
        'Mean_GR_Activity': mean_gr,
        'Variance_GR_Activity': var_gr,
        'Range_GR_Activity': range_gr
    }
    
    metric_name = "Threshold" if plasticity_type == 'IE' else "Weight"
    correlations = {}
    
    print(f"\n--- {plasticity_type} Analysis Results ---")
    
    # 3. Process Plasticity Arrays
    for i, p_array in enumerate(plasticity_arrays):
        name = array_names[i]
        t0 = time.time()
        
        # Vectorized metrics
        mean_p = np.mean(p_array, axis=1)
        var_p = np.var(p_array, axis=1)
        range_p = np.ptp(p_array, axis=1)
        
        # Pearson correlations (Very fast on 1D arrays of size 1000)
        corr_go, p_val_go = pearsonr(mean_go, mean_p)
        corr_gr, p_val_gr = pearsonr(mean_gr, mean_p)
        
        correlations[name] = {
            'GO_corr': corr_go, 'GO_pval': p_val_go,
            'GR_corr': corr_gr, 'GR_pval': p_val_gr
        }
        
        print(f"[{name}] Processed in {time.time() - t0:.2f} seconds")
        print(f"  -> Corr w/ GO: {corr_go:.4f} (p={p_val_go:.4e})")
        print(f"  -> Corr w/ GR: {corr_gr:.4f} (p={p_val_gr:.4e})")
        
        # Append to DataFrame dictionary
        data[f'Mean_{plasticity_type}_{metric_name}_{name}'] = mean_p
        data[f'Variance_{plasticity_type}_{metric_name}_{name}'] = var_p
        data[f'Range_{plasticity_type}_{metric_name}_{name}'] = range_p
    
    # 4. Package and Export
    print("\nPackaging into Pandas DataFrame and saving to Excel...")
    df = pd.DataFrame(data)
    
    try:
        df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"Data successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        
    print(f"\nTotal function run time: {time.time() - start_time:.2f} seconds")
    return df, correlations

# # ==========================================
# # FILE LOADING
# # ==========================================
# go_act = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial_GOrasters.npy')
# gr_act = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial_GRrasters.npy')

# ## IE
# go_thresh = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial_GoThr.npy')
# gr_thresh = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_bothplast_1000_trial_GrThr.npy')

# print("Finished loading data")

# df_results, corrs = export_activity_vs_plasticity_to_excel(
#     go_activity=go_act, 
#     gr_activity=gr_act, 
#     plasticity_arrays=[go_thresh, gr_thresh],
#     output_filename="both_IE_Results.xlsx",
#     plasticity_type='IE'
# )

# ==========================================
# FILE LOADING
# ==========================================
go_act = np.load('/home/data/einez/homeostat_SS/75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial/75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial_GOrasters.npy')
gr_act = np.load('/home/data/einez/homeostat_SS/75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial/75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial_GRrasters.npy')

## IE
# go_thresh = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GOplast_1000_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GOplast_1000_trial_GoThr.npy')
# gr_thresh = np.load('/home/data/einez/homeostat_IE/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GRplast_1000_trial/MFGoGr_IE_shuffleMF10percent_noCS_yesGoGo_yesgrGo_GRplast_1000_trial_GrThr.npy')

## SS
# mfgoW = np.load('/home/data/einez/homeostat_SS/MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial/MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial_mfgoW.npy')
# grgoW = np.load('/home/data/einez/homeostat_SS/MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial/MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial_grgoW.npy')
gogoW = np.load('/home/data/einez/homeostat_SS/75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial/75recip_MFGoGr_SS_shuffleMF10percent_noCS_gogoplast_1000_trial_gogoW.npy')
print("Finished loading data")

df_results, corrs = export_activity_vs_plasticity_to_excel(
    go_activity=go_act, 
    gr_activity=gr_act, 
    plasticity_arrays=[gogoW],
    output_filename="75recip_gogoplast_SS_Results.xlsx",
    plasticity_type='SS'
)