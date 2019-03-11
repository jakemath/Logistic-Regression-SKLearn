Problem Statement: 

    You are provided with data that is related to direct marketing campaigns (phone calls) of a Portuguese banking
    institution. You have to perform  classification on this data using `Logistic Regression` to predict if the client 
    will subscribe a term deposit (variable y). 

    The original dataset is available at [UCI Bank Marketing Data](https://archive.ics.uci.edu/ml/datasets/bank+marketing#). 
    For more detailed description about the dataset, please look at the site to know the attributes. 

This program performs exploratory data analysis on the dataset using Python Pandas, including filtering and combining certain categories to simplify the dataset. 

After simplifying and exploring the dataset, the attributes containing string data are encoded using Scikit-Learn LabelEncoder. After converting all the data to numerical values, the data is split into 70% train/30% test, and a Scikit-Learn logistic regression model is fitted to the data.

The performance of the model is conveyed by computing % accuracy to the test data and via Scikit-Learn Confusion Matrix and Classification Report.
