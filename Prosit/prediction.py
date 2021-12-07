import os
import pandas as pd

import model as model_lib
import iioo
import constants
import sanitize


def predict(tensor, model, model_config, verbose=False):
    import keras
    # check for mandatory keys
    x = iioo.get_array(tensor, model_config["x"])

    model.compile(optimizer="adam", loss="mse")
    prediction = model.predict(
        x, verbose=verbose, batch_size=constants.PRED_BATCH_SIZE
    )
    if model_config["prediction_type"] == "intensity":
        tensor["intensities_pred"] = prediction
        tensor = sanitize.prediction(tensor)
    elif model_config["prediction_type"] == "iRT":
        import numpy as np
        tensor["iRT"] = prediction * np.sqrt(float(model_config["iRT_rescaling_var"])) + float(model_config["iRT_rescaling_mean"])
    else:
        raise ValueError("model_config misses parameter")

    return tensor


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # turn off tf logging
    data_path = constants.DATA_PATH
    model_dir = constants.MODEL_DIR

    data_path = r'G:\00_DDA_DL\prosit\data\test.h5'
    model_dir = r'G:\00_DDA_DL\prosit\model'
    weights_path = model_lib.get_best_weights_path(model_dir)
    weights_name = weights_path.split("/")[-1][:-5]
    data_name = r'G:\00_DDA_DL\prosit\data\test.hdf5'
    model, model_config = model_lib.load(model_dir, trained=True)

    tensor = iioo.from_hdf5(data_path)
    # tensor here is a dictionary
    tensor = predict(tensor, model, model_config, verbose=True)

    path = os.path.join(constants.OUT_DIR, "prediction.hdf5")
    # tensor here is a dictionary
    iioo.to_hdf5(tensor, path)
