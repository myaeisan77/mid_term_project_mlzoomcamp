# mid_term_project_mlzoomcamp

The problem is to predict the quality of wine datasets.  

**project-mlzoomcamp.iynb file**  

Step 1 : Downloaded wine dataset from kaggle (https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)  
Step 2 : Cleaned the datset and did EDA analysis  
Step 3 : Spilt the dataset into train, valid and test  
Step 4 : Trained Logistic Regression model  
Step 5 : Used Feature Elimination Technique to reduce less importance feature and found that all features are equally important  
Step 6 : K-fold Cross Validation to find the best optimization value for the logistic Regression model  
Step 7: Trained the model again with the best optimization value and increased the accuray of the model  
Step 8 : Trained RadomForest Regression model and searched the best estimator and max_depth value to reduce the RMSE  

Finally it was found that Logistic Regression Model performed better.  

**train.py**  

Save the model.bin and dv.bin files from the .iynb notebook and trained again in train.py file  

**Dockerfile**  
Create dockerfile to setup environment for running predict.py file  




