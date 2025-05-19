import pickle


def test_features(features_path,shape=(768,)):
    # Cargar el diccionario de features
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    print(features_path)
    first_key = next(iter(features.keys()))
    # print(f"First key: {first_key}")
    # mostrar tensor primera key
    # print(f"First tensor: {features[first_key]}")

    # Mostrar algunas claves y las dimensiones de sus arrays
    for key, feat in features.items():
        # if "F013620" in key and features_path.endswith("vit.pkl"):
        #     print(f"Key: {key}")
        if(feat.shape!=shape):
            raise ValueError('Wrong format')
    # Mostrar la clave del primer feature
    


def main():
    test_features("/features/video/timesformer_train.pkl")
    test_features("/features/video/videomae_train.pkl")
    test_features("/features/video/vivit_train.pkl")
    test_features("/features/objects/GT/swin.pkl",shape=(1024,))
    test_features("/features/objects/GT/vit.pkl",shape=(768,))
    test_features("/features/objects/text/GT/DistilBert.pkl")
    test_features("/features/objects/text/GT/Bert.pkl")
    test_features("/features/objects/text/GT/Roberta.pkl")



    test_features("/features/video/timesformer_val.pkl")
    test_features("/features/video/videomae_val.pkl")
    test_features("/features/video/vivit_val.pkl")






    test_features("/features/video/timesformer_test.pkl")
    test_features("/features/video/videomae_test.pkl")
    test_features("/features/video/vivit_test.pkl")


    print("OK!")


if __name__ == "__main__":
    main()



