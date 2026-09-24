import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

from parameters import *
from utils import _loadfile, _logMelFilterbank
from get_data import downloadData, getDataDict, getDataframe
from edge_kws import EdgeKWS

TFLITE_FILE = "marvin_kws_int8.tflite"
NPZ_FILE = "marvin_kws_svm.npz"
NUM_CALIBRATION = 500


def _fbanks(files):
    """
    Computes filterbanks for a list of wav files
    :param files: File names
    :return: Filterbanks (n, 99, 40)
    """

    return np.stack([_logMelFilterbank(_loadfile(file_name)) for file_name in files])


def export_tflite(feature_extractor, calibration_fbanks):
    """
    Converts the feature extractor to a full-integer quantized TFLite model (float input/output)
    :param feature_extractor: Keras feature extractor
    :param calibration_fbanks: Filterbanks used to calibrate activation ranges
    :return: None
    """

    saved_model_dir = MODEL_DIR + "feature_extractor_savedmodel"
    feature_extractor.export(saved_model_dir)

    def representative_dataset():
        for fbank in calibration_fbanks:
            yield [fbank[np.newaxis].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    with open(MODEL_DIR + TFLITE_FILE, "wb") as file:
        file.write(converter.convert())

    tf.io.gfile.rmtree(saved_model_dir)
    print("TFLite model size: {:.1f} KB".format(os.path.getsize(MODEL_DIR + TFLITE_FILE) / 1024))


def export_svm(pca, marvin_svm):
    """
    Stores the PCA and one-class SVM parameters as plain numpy arrays
    :param pca: Fitted PCA
    :param marvin_svm: Fitted OneClassSVM
    :return: None
    """

    np.savez(
        MODEL_DIR + NPZ_FILE,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        support_vectors=marvin_svm.support_vectors_.astype(np.float32),
        dual_coef=marvin_svm.dual_coef_[0].astype(np.float32),
        intercept=np.float32(marvin_svm.intercept_[0]),
        gamma=np.float32(marvin_svm.get_params()["gamma"]),
    )


def print_metrics(name, y_true, y_pred):
    print(
        "{:<22} acc={:.4f} precision={:.4f} recall={:.4f} F1={:.4f} MCC={:.4f}".format(
            name,
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            matthews_corrcoef(y_true, y_pred),
        )
    )


def main():
    np.random.seed(0)

    downloadData(data_path="/input/speech_commands/")
    dataDict = getDataDict(data_path="/input/speech_commands/")

    trainDF = getDataframe(dataDict["train"])
    evalDF = pd.concat(
        [getDataframe(dataDict["dev"], include_unknown=True), getDataframe(dataDict["test"], include_unknown=True)],
        ignore_index=True,
    )
    y_true = np.where(evalDF["category"] == "marvin", 1, -1)

    model = load_model(MODEL_DIR + MODEL_FILE)
    feature_extractor = Model(inputs=model.inputs, outputs=model.get_layer("features256").output)

    with open(MODEL_DIR + PCA_FILE, "rb") as file:
        pca = pickle.load(file)
    with open(MODEL_DIR + SVM_FILE, "rb") as file:
        marvin_svm = pickle.load(file)

    # Quantize with activation ranges calibrated on training clips
    calibration_files = trainDF.sample(n=NUM_CALIBRATION, random_state=0)["files"].tolist()
    export_tflite(feature_extractor, _fbanks(calibration_files))
    export_svm(pca, marvin_svm)

    # Compare float and int8 pipelines on identical filterbanks
    print("Computing filterbanks for {} evaluation files".format(evalDF.shape[0]))
    eval_fbanks = _fbanks(evalDF["files"].tolist())

    float_embeddings = feature_extractor.predict(eval_fbanks, batch_size=BATCH_SIZE, verbose=0)
    float_pred = marvin_svm.predict(pca.transform(float_embeddings))

    edge = EdgeKWS(MODEL_DIR.rstrip("/"))
    numpy_svm_pred = np.where(edge.decision(float_embeddings) >= 0, 1, -1)
    print("numpy SVM agrees with sklearn on {:.4%} of clips".format(np.mean(numpy_svm_pred == float_pred)))

    start = time.perf_counter()
    int8_embeddings = np.stack([edge.embed(fbank) for fbank in eval_fbanks])
    elapsed = time.perf_counter() - start
    int8_pred = np.where(edge.decision(int8_embeddings) >= 0, 1, -1)

    print("Keras model size: {:.1f} KB".format(os.path.getsize(MODEL_DIR + MODEL_FILE) / 1024))
    print("int8 inference: {:.2f} ms per 1 s window on this machine".format(1000 * elapsed / len(eval_fbanks)))
    print_metrics("float32 (Keras+sklearn)", y_true, float_pred)
    print_metrics("int8 (TFLite+numpy)", y_true, int8_pred)
    print("int8 agrees with float32 on {:.4%} of clips".format(np.mean(int8_pred == float_pred)))


if __name__ == "__main__":
    main()
