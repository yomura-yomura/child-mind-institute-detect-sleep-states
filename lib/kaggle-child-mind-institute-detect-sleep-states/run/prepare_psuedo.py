from pathlib import Path

import numpy as np
from cmi_dss_lib.utils.load_predicted import load_predicted
from joblib import Parallel, delayed
from tqdm import tqdm


def preprocess_for_psuedo_labels(pred:np.ndarray,th_sleep_prob:float,watch_interval_hour:float,result_path:str,s_id:str) -> None:
    th_hour_step = watch_interval_hour*12*60
    max_len_step = pred.shape[0]
    # cutting by sleep_prob
    pred_copy = pred.copy()
    for step in range(max_len_step):
        # onset 
        max_step = min(step + th_hour_step, max_len_step - 1)
        if step < max_step:
            sleep_score = np.median(pred_copy[:,0][step:max_step])
            if sleep_score < th_sleep_prob:
                pred_copy[:,1][step] = 0
        else: 
            pred_copy[:,1][step] = 0
        # wakeup 
        min_step = max(step - th_hour_step, 0)
        if min_step < step:
            sleep_score = np.median(pred_copy[:,0][min_step:step])
            if sleep_score < th_sleep_prob:
                pred_copy[:,1][step] = 0
        else:
            pred_copy[:,1][step] = 0
    
    np.savez(f"{result_path}/{s_id}.npz",pred_copy)
    return 

if __name__ == "__main__":
    watch_interval_hour = 4
    th_sleep_prob = 0.8

    path_data = ""

    dict_data = load_predicted(path_data)


    result_path = Path(__file__).parent.parent / "psuedo_label" /  "temp"

    result_path.mkdir(exist_ok=True,parents = True)
    Parallel(n_jobs=-1)(delayed(preprocess_for_psuedo_labels)(pred, th_sleep_prob, watch_interval_hour,str(result_path),s_id) for s_id,pred in tqdm(dict_data.items(),desc ="user"))

