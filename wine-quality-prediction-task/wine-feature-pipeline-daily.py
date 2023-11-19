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

    random_quality = random.randint(1, 4)
    wine_feature_per_quality = fs.get_feature_group(
        name=f"wine_features_quality_{random_quality}", version=1
    )

    wine_features_statistics = wine_feature_per_quality.get_statistics()

    new_wine_features = {}
    for stat in wine_features_statistics.content["columns"]:
        new_wine_features[stat["column"]] = [
            max(0, random.gauss(stat["mean"], stat["stdDev"]))
        ]
        if stat["column"] == "wine_type":
            new_wine_features[stat["column"]][0] = round(
                new_wine_features[stat["column"]][0]
            )
    new_wine_features["quality"] = [random_quality]
    new_wine_features = pd.DataFrame(new_wine_features)

    main_features = fs.get_feature_group(name="wine_features", version=1)
    main_features.insert(new_wine_features)


if __name__ == "__main__":
    main()
