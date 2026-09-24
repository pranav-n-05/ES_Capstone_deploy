import os
from parameters import MODEL_DIR, PCA_FILE, SVM_FILE
from model_train import model_train, marvin_kws_model
from model_test import marvin_model_test


def main():
    trained = os.path.isfile(MODEL_DIR + SVM_FILE) and os.path.isfile(MODEL_DIR + PCA_FILE)

    if not trained:
        print("Training model")
        model_train()
        marvin_kws_model()
    else:
        print("Testing model")
        marvin_model_test()


if __name__ == "__main__":
    main()
