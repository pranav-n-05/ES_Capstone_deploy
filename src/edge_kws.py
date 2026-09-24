"""
Lightweight Marvin detector for edge devices (e.g. Raspberry Pi).
Needs only numpy, python_speech_features and a TFLite runtime - no TensorFlow or scikit-learn.
"""

import numpy as np
from python_speech_features import logfbank

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

AUDIO_SR = 16000
AUDIO_LENGTH = 16000
PARSE_PARAMS = (0.025, 0.01, 40)


def log_mel_filterbank(wave):
    """
    Computes the log Mel filterbanks for a 1 s window, same settings as training
    :param wave: Audio as an array
    :return: Filterbanks (99, 40)
    """

    wave = np.asarray(wave, dtype=np.float32)

    # Pad with noise if audio is short
    if len(wave) < AUDIO_LENGTH:
        wave = np.append(wave, np.random.normal(0, 5, AUDIO_LENGTH - len(wave))).astype(np.float32)

    fbank = logfbank(
        wave[:AUDIO_LENGTH],
        samplerate=AUDIO_SR,
        winlen=PARSE_PARAMS[0],
        winstep=PARSE_PARAMS[1],
        highfreq=AUDIO_SR / 2,
        nfilt=PARSE_PARAMS[2],
    )

    return np.array(fbank, dtype=np.float32)


class EdgeKWS:
    """
    int8 TFLite feature extractor + numpy PCA + numpy one-class SVM
    """

    def __init__(self, model_dir, num_threads=4):
        self.interpreter = Interpreter(model_path=model_dir + "/marvin_kws_int8.tflite", num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        params = np.load(model_dir + "/marvin_kws_svm.npz")
        self.pca_mean = params["pca_mean"]
        self.pca_components = params["pca_components"]
        self.support_vectors = params["support_vectors"]
        self.dual_coef = params["dual_coef"]
        self.intercept = float(params["intercept"])
        self.gamma = float(params["gamma"])

    def embed(self, fbank):
        """
        Obtain the 256-d embedding for one window
        :param fbank: Filterbanks (99, 40)
        :return: Embedding (256,)
        """

        self.interpreter.set_tensor(self.input_index, fbank[np.newaxis].astype(np.float32))
        self.interpreter.invoke()

        return self.interpreter.get_tensor(self.output_index)[0]

    def decision(self, embeddings):
        """
        One-class SVM decision function (RBF kernel) on PCA-reduced embeddings
        :param embeddings: Embeddings (n, 256)
        :return: Decision values (n,)
        """

        x = (np.atleast_2d(embeddings) - self.pca_mean) @ self.pca_components.T
        sq_dist = ((x[:, np.newaxis, :] - self.support_vectors[np.newaxis]) ** 2).sum(axis=-1)

        return np.exp(-self.gamma * sq_dist) @ self.dual_coef + self.intercept

    def predict(self, fbank):
        """
        Detect hotword presence in current window
        :param fbank: Filterbanks (99, 40)
        :return: 1 for Marvin, -1 otherwise
        """

        return 1 if self.decision(self.embed(fbank))[0] >= 0 else -1
