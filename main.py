import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import hcbcf
import helpers
import itemCF
import proposed

dir_path = os.path.join("input_output_data", "")

def load_documents(file_path):
    documents = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                key, text = line.split('\t', 1)
                documents[key] = text.strip()
    return documents

def get_all_predictions(df, partial_df, lambda_param=.5, nn=4):
    item_similarity = itemCF.calculate_item_similarity(partial_df)
    actual_values = []
    predicted_values_ICF = []
    predicted_values_HCBCF = []
    predicted_values_HTS = []
    predicted_values_ProposedMethod = []
    predicted_values_WoS = []
    predicted_values_WoF = []
    predicted_values_WoSWoF = []

    sim_threshold = 0
    for user_idx, row in df.iterrows():
        user = row['UserName']
        print(user)
        for item in df.columns[1:]:
            if not pd.isna(row[item]) and pd.isna(partial_df.at[user_idx, item]):
                if item=='b.1.2':
                    pass
                actual_value = row[item]
                actual_values.append(actual_value)

                # Predict value using ItemCF
                predicted_value_ICF = itemCF.predict_ratingItemCF(partial_df, user, item, item_similarity, nn)
                predicted_values_ICF.append(predicted_value_ICF)
                
                # Predict value using HCBCF
                predicted_value_HCBCF = hcbcf.getHCBCFHybridPredict(partial_df, user, item, lambda_param, nn)
                predicted_values_HCBCF.append(predicted_value_HCBCF)

                # Predict value using HTS
                HTS = proposed.ProposedMethod(partial_df, documents, sim_threshold, 4, 1)
                predicted_value_HTS = HTS.getProposedPredict(user, item, lambda_param)
                predicted_values_HTS.append(predicted_value_HTS)

                # Predict value using method2
                proposed_method = proposed.ProposedMethod(partial_df, documents, sim_threshold, 4, 2)
                predicted_value_ProposedMethod = proposed_method.getProposedPredict(user, item, lambda_param)
                predicted_values_ProposedMethod.append(predicted_value_ProposedMethod)

                # Predict value using method2 WoSemSim
                proposed_method = proposed.ProposedMethod(partial_df, documents, sim_threshold, 4, 2, woSemSim=True, woFlexW=False)
                predicted_value_WoS = proposed_method.getProposedPredict(user, item, lambda_param)
                predicted_values_WoS.append(predicted_value_WoS)

                # Predict value using method2 WoFlexW
                proposed_method = proposed.ProposedMethod(partial_df, documents, sim_threshold, 4, 2, woSemSim=False, woFlexW=True)
                predicted_value_WoF = proposed_method.getProposedPredict(user, item, lambda_param)
                predicted_values_WoF.append(predicted_value_WoF)

                # Predict value using method2 WoSemSim and WoFlexW
                proposed_method = proposed.ProposedMethod(partial_df, documents, sim_threshold, 4, 2, woSemSim=True, woFlexW=True)
                predicted_value_WoSWoF = proposed_method.getProposedPredict(user, item, lambda_param)
                predicted_values_WoSWoF.append(predicted_value_WoSWoF)

    return {
        'actual_values': actual_values,
        'ICF': predicted_values_ICF,
        'HCBCF': predicted_values_HCBCF,
        'HTS': predicted_values_HTS,
        'ProposedMethod': predicted_values_ProposedMethod,
        'WoS': predicted_values_WoS,
        'WoF': predicted_values_WoF,
        'WoSWoF': predicted_values_WoSWoF
        }

def generate_and_save(df, trial, dir_path, filename_suffix, value, func):
    file_name = f"tr{trial}.{filename_suffix}{value}"
    par_path = os.path.join(dir_path, f"{file_name}.csv")
    complete_path = os.path.join(dir_path, f"{file_name}c.csv")
    if not os.path.exists(par_path):
        if func == helpers.introduce_sparsity:
            partial_df = func(df, value)
            partial_df.to_csv(par_path, index=False)
        else:
            complete_df, partial_df = func(df, value)
            complete_df.to_csv(complete_path, index=False)
            partial_df.to_csv(par_path, index=False)

def generate_partial_ratings(df, n_trials=3):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    sparsity_range = {.9, .95, .97, .98, .985, .99, .995}
    ics_range = range(2, 9, 1)
    ucs_range = range(2, 9, 1)
    for trial in range(1, n_trials + 1):
        for sparsity_level in sparsity_range:
            generate_and_save(df, trial, dir_path, 'sp', sparsity_level, helpers.introduce_sparsity)
        for n_ratings_per_item in ics_range:
            generate_and_save(df, trial, dir_path, 'ics.', n_ratings_per_item, helpers.introduce_item_cold_start_sparsity)
        for n_ratings_per_user in ucs_range:
            generate_and_save(df, trial, dir_path, 'ucs.', n_ratings_per_user, helpers.introduce_user_cold_start_sparsity)

#Start
start_time = time.time()
if not os.path.exists('Processed-Req-list.txt'):
    print('Processing started.')
    with open('Req-list.txt', 'r') as file:
        lines = file.readlines()
    processed_lines = []
    for line in lines:
        identifier, text = line.strip().split('\t')
        processed_text = helpers.preprocess(text)
        processed_lines.append(f"{identifier}\t{''.join(processed_text)}")
    with open('Processed-Req-list.txt', 'w') as file:
        for line in processed_lines:
            file.write(line + '\n')
    print('Done processing file!')

documents = load_documents('Processed-Req-list.txt')
rating = pd.read_csv('normalized_data.csv') 

n_trials = 3
generate_partial_ratings(rating, n_trials)
print('Partial ratings are generated!')

sparsity_range = [0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995]
ics_range = range(2, 9)
ucs_range = range(2, 9)

for trial in range(1, n_trials+1):
    trial_prefix = f"tr{trial}"
    
    # Loop over sparsity files
    results = []
    for sp in sparsity_range:
        # rating_df = pd.read_csv(os.path.join(dir_path, f"{trial_prefix}.sp{sp}c.csv"))
        rating_df = rating
        partial_df = pd.read_csv(os.path.join(dir_path, f"{trial_prefix}.sp{sp}.csv"))
        res_dict = get_all_predictions(rating_df, partial_df)
        actual = res_dict['actual_values']
        methods = [k for k in res_dict.keys() if k != 'actual_values']
        metrics_for_sp = {'Sparsity': sp}
        
        for method in methods:
            pred = res_dict[method]
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            
            metrics_for_sp[f'MAE_{method}'] = mae
            metrics_for_sp[f'RMSE_{method}'] = rmse
            metrics_for_sp[f'R2_{method}'] = r2
        
        results.append(metrics_for_sp)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='Sparsity').set_index('Sparsity')
        output_path = os.path.join(dir_path, f"results_{trial_prefix}_sp.xlsx")
        df_results.to_excel(output_path)
    
    #  Loop over ICS files
    results = []
    for ics in ics_range:
        rating_df = pd.read_csv(os.path.join(dir_path, f"{trial_prefix}.ics.{ics}c.csv"))
        partial_df = pd.read_csv(os.path.join(dir_path, f"{trial_prefix}.ics.{ics}.csv"))
        res_dict = get_all_predictions(rating_df, partial_df)
        actual = res_dict['actual_values']
        methods = [k for k in res_dict.keys() if k != 'actual_values']
        metrics_for_sp = {'RatingsPerItem': ics}
        
        for method in methods:
            pred = res_dict[method]
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            
            metrics_for_sp[f'MAE_{method}'] = mae
            metrics_for_sp[f'RMSE_{method}'] = rmse
            metrics_for_sp[f'R2_{method}'] = r2
        
        results.append(metrics_for_sp)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='RatingsPerItem').set_index('RatingsPerItem')
        output_path = os.path.join(dir_path, f"results_{trial_prefix}_ics.xlsx")
        df_results.to_excel(output_path)

    #  Loop over UCS files
    results = []
    for ucs in ucs_range:
        rating_df = pd.read_csv(os.path.join(dir_path, f"{trial_prefix}.ucs.{ucs}c.csv"))
        partial_df = pd.read_csv(os.path.join(dir_path, f"{trial_prefix}.ucs.{ucs}.csv"))
        res_dict = get_all_predictions(rating_df, partial_df)
        actual = res_dict['actual_values']
        methods = [k for k in res_dict.keys() if k != 'actual_values']
        metrics_for_sp = {'RatingsPerUser': ucs}
        
        for method in methods:
            pred = res_dict[method]
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            r2 = r2_score(actual, pred)
            
            metrics_for_sp[f'MAE_{method}'] = mae
            metrics_for_sp[f'RMSE_{method}'] = rmse
            metrics_for_sp[f'R2_{method}'] = r2
        
        results.append(metrics_for_sp)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='RatingsPerUser').set_index('RatingsPerUser')
        output_path = os.path.join(dir_path, f"results_{trial_prefix}_ucs.xlsx")
        df_results.to_excel(output_path)

print(f"Done! Duration: {time.time() - start_time} seconds")