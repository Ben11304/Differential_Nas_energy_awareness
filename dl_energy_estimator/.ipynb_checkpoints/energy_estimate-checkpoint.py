
import pickle
import numpy as np
import pandas as pd
import sys
import pickle
# sys.path.append('../dl_energy_estimator')
from dl_energy_estimator.utils.data_utils import preprocess_and_normalize_energy_data, parse_codecarbon_output
from dl_energy_estimator.utils.experiments_utils import split_data_set, fit_model, compute_log_transformed_features, apply_data_transforms, test_model
# # Tạo một sample để dự đoán
# sample = pd.DataFrame([{
#     "batch_size": 123,
#     "image_size": 127,
#     "kernel_size": 7,
#     "input_size":4,
#     "output_size":2,
#     "in_channels": 80,
#     "out_channels": 29,
#     "stride": 1,
#     "padding": 2,
#     "attributed": "conv2d",
#     "sub_attributed": "tanh",
#     "macs":2.18534484e+11
# }])

def calculate_energy(sample):
    op_name=sample["attributed"][0]
    if op_name=="activation":
        param_cols = ['batch_size','input_size']
        sample_norm=sample[param_cols]
        data_linear_with_log, param_cols_with_log = sample_norm, param_cols
        sub=sample["sub_attributed"][0]
        with open(f"./dl_energy_estimator/energy_model/{op_name}/{sub}/linear_test_conv_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        # Load transformers
        with open(f"./dl_energy_estimator/energy_model/{op_name}/{sub}/x_transformer.pkl", "rb") as f:
            x_transformer = pickle.load(f)
        
        with open(f"./dl_energy_estimator/energy_model/{op_name}/{sub}/y_transformer.pkl", "rb") as f:
            y_transformer = pickle.load(f)
        
        X_new_transformed = x_transformer.transform(data_linear_with_log)
    
        # Dự đoán và inverse scale (nếu muốn)
        y_pred_scaled = loaded_model.predict(X_new_transformed)
        y_pred = y_transformer.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # print(f"📌 Dự đoán năng lượng tiêu thụ: {max(y_pred[0][0],0)}")
    
    else:
        # if op_name=="linear":
        #     param_cols = ['batch_size','input_size','output_size']
        # if op_name=="maxpool2d":
        #     param_cols=['batch_size', 'image_size', 'kernel_size', 'in_channels', 'stride', 'padding']
        # if op_name=="conv2d":
        #     param_cols = ['batch_size','image_size','kernel_size','in_channels','out_channels','stride','padding']
            
        # sample_norm=sample[param_cols]
        # data_linear_with_log, param_cols_with_log = compute_log_transformed_features(sample_norm, param_cols)
        # data_linear_with_log['macs']=sample["macs"]
        
        with open(f"./dl_energy_estimator/mac_only/{op_name}/linear_test_conv_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
        # Load transformers
        # with open(f"./dl_energy_estimator/mac_only/{op_name}/x_transformer.pkl", "rb") as f:
        #     x_transformer = pickle.load(f)
        
        with open(f"./dl_energy_estimator/mac_only/{op_name}/y_transformer.pkl", "rb") as f:
            y_transformer = pickle.load(f)
        
        # X_new_transformed = x_transformer.transform(data_linear_with_log)
        
        # Dự đoán và inverse scale (nếu muốn)
        y_pred_scaled = loaded_model.predict(sample['macs'].values.reshape(-1, 1))
        y_pred = y_transformer.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        
        # print(f"📌 Dự đoán năng lượng tiêu thụ: {max(y_pred[0][0],0)}")
    return abs(y_pred[0][0])



