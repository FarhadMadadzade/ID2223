# ID2223 Lab 1

Lab 1 consists of two parts. The first part is to get familiar with the different platforms and how to deploy the UI for the model and the model monitoring for the iris dataset.

Part 2 of the lab is to do everything from scratch for an other dataset, in this case the Wine Quality dataset. The work done to get to a final product includes the following:
1. Data preprocessing. This step consists of getting familiar with the data, and doing some feature engingeering to get a dataset that can yield good results. In this case we changed the quality feature rating from a 1-10 to a 1-3 rating system because many of the qualities in the original dataset were poorly represented. We also identified which features actually had an importance to the model that we created. Finally we tested different model architectures by training them on the data (consisting of both red and white wines with the features identified as important) to decide which one worked best for the dataset.
2. Feature pipeline. Based on the results from the preprocessing we create a pipeline which reads the datafiles, takes out the relevant features, among them the ones identified in the preprocessing, and inserting them to Hopsworks. We create a total of 4 different feature groups. The first is where all the wines are included. The other 3 is one for each quality containing features for wines with that specific quality.
3. Daily feature pipeline. This pipeline randomly picks a quality between 1-3. It then fetches the feature group for that specific quality (one of the 3 ones mentioned in the previous point) and generates a new wine. For each feature it takes a random value based on a gaussian distribution with the mean and standard deviation for each feature in the feature group. It then inserts this new wine with its features into the feature group containing all the wines.
4. training pipeline. The training pipeline where we generate the model to be used
5. inference pipeline. where we run inference on the model
6. Hugginface Interactive UI. The UI for the user where one can enter different values for the features and the model prediction is presented.
7. Hugginface Dashboard UI for monitoring. The UI for monitoring the last added wine to the feature store.

Hugginface Interactive UI URL: https://huggingface.co/spaces/ID2223-labs/Wine

Hugginface Dashboard UI URL: https://huggingface.co/spaces/ID2223-labs/Wine_Monitoring
