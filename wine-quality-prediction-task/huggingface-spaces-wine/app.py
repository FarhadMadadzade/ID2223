import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=5)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(alcohol, volatile_acidity, chlorides, sulphates, free_sulfur_dioxide):
    print("Calling function")
    df = pd.DataFrame(
        [[alcohol, volatile_acidity, chlorides, sulphates, free_sulfur_dioxide]],
        columns=[
            "alcohol",
            "volatile_acidity",
            "chlorides",
            "sulphates",
            "free_sulfur_dioxide",
        ],
    )
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    #     print("Res: {0}").format(res)
    print(res)
    image = Image.open("./images/" + str(int(res[0])) + "_glass.png")
    return image


demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with alcohol/volatile acidity/chlorides/sulphates/wine type and see the prediction of the wine quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=10.0, label="alcohol"),
        gr.inputs.Number(default=0.2, label="volatile acidity"),
        gr.inputs.Number(default=0.69, label="sulphates"),
        gr.inputs.Number(default=0.03, label="chlorides"),
        gr.inputs.Number(default=71, label="free sulfur dioxide"),
    ],
    outputs=gr.Image(type="pil", label="Wine Quality (1-3)"),
)

demo.launch(debug=True)
