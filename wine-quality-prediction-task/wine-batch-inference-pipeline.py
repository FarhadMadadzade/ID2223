import os
import pandas as pd
import hopsworks
import joblib
import datetime
from PIL import Image
from datetime import datetime
import dataframe_image as dfi
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import seaborn as sns
import requests
import random

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=5)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")

feature_view = fs.get_feature_view(name="wine_features", version=1)
batch_data = feature_view.get_batch_data()
batch_data = batch_data.drop('wine_type', axis=1)

y_pred = model.predict(batch_data)
#print(y_pred)
offset = random.randint(1, y_pred.size) 
print("Offset: " + str(offset))
wine = y_pred[y_pred.size-offset]
print("Wine quality predicted: " + str(wine))        
wine_url = "https://github.com/FarhadMadadzade/ID2223/blob/c18fdfafc72bcf7455c7e7fde54a3c76ef0f3bba/wine-quality-prediction-task/images/" + str(int(wine)) + ".png?raw=true"
img = Image.open(requests.get(wine_url, stream=True).raw)            
img.save("./latest_wine.png")
dataset_api = project.get_dataset_api()    
dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)

wine_fg = fs.get_feature_group(name="wine_features", version=1)
df = wine_fg.read() 
#print(df)
label = df.iloc[-offset]["quality"]
print("Wine quality actual: " + str(label))
label_url = "https://github.com/FarhadMadadzade/ID2223/blob/c18fdfafc72bcf7455c7e7fde54a3c76ef0f3bba/wine-quality-prediction-task/images/" + str(int(label)) + ".png?raw=true"
img = Image.open(requests.get(label_url, stream=True).raw)            
img.save("./actual_wine.png")
dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)

monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                            version=1,
                                            primary_key=["datetime"],
                                            description="Wine quality Prediction/Outcome Monitoring"
                                            )

now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
data = {
    'prediction': [int(wine)],
    'label': [int(label)],
    'datetime': [now],
    }
monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

history_df = monitor_fg.read()
# Add our prediction to the history, as the history_df won't have it - 
# the insertion was done asynchronously, so it will take ~1 min to land on App
history_df = pd.concat([history_df, monitor_df])


df_recent = history_df.tail(4)
dfi.export(df_recent, './df_wine_recent.png', table_conversion = 'matplotlib')
dataset_api.upload("./df_wine_recent.png", "Resources/images", overwrite=True)

predictions = history_df[['prediction']]
labels = history_df[['label']]

# Only create the confusion matrix when our wine_predictions feature group has examples of all 3 wine qualities
print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
if predictions.value_counts().count() == 3:
    results = confusion_matrix(labels, predictions)

    df_cm = pd.DataFrame(results, ['True 1', 'True 2', 'True 3'], 
                    ['Pred 1', 'Pred 2', 'Pred 3']) 

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./confusion_wine_matrix.png")
    dataset_api.upload("./confusion_wine_matrix.png", "Resources/images", overwrite=True)
else:
    print("You need 3 different quality predictions to create the confusion matrix.")
    print("Run the batch inference pipeline more times until you get 3 different quality wines") 