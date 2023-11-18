def main():
    import hopsworks
    import pandas as pd
    import random

    hopsworks_project = hopsworks.login()
    fs = hopsworks.project_featurestore()
