def main():
    import os
    import hopsworks
    import pandas as pd
    import random

    root_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root_path, "credentials.conf"), "r") as f:
        api_key = f.read()
        api_key = api_key.split("=")[1].strip()

    hopsworks_project = hopsworks.login(api_key_value=api_key)
    # hopsworks_project = hopsworks.login()
    fs = hopsworks_project.get_feature_store()

    wine_features = fs.get_feature_group(name="wine_features", version=1)

    print(wine_features.features)

    quality = random.randint(1, 4)


if __name__ == "__main__":
    main()
