**Submitted by:**

Name: Chew Ern (Zhou En)

Email: <chewern@gmail.com>

__________________________


# Section A: Folders and Files

Folder structure is as required, with no change or addition. By executing **`run.sh`** in base folder, the selected files in **`src`** folder will be activated, which runs both regressions and classifications machine learning algorithms. In the base folder, there are also 3 other BASH executable files – **`classifiers.sh`** , **`regressions.sh`** and **`bins.sh`** – the function of these will be explained later in this section.

1.	All 14 python source code files are stored in **`src`** folder, but only 2 of which can be execute using **`run.sh`**, namely:
	1. **Regressions_Assemble.py** –This program runs the regressions methods and output the results on screen as well as appending into file **`data/Regressions_Assemble.txt`**.
	2. **Classifiers_Assemble.py** –This program runs all the classification methods and output the results on screen as well as appending into file **`data/Classifiers_Assemble.txt`**. This program will also generate confusion matrix plots and classification error plots into data folder, saved as .png files, the file names are displayed on screen when the program running.

2.	The remaining 12 python files are used for fine tuning the models. There are 3 main groups, **Tune_Regressor_** , **Tune_Classifier_** and **Tune_Bins**.
	1. **Tune_Regressor_** – All 5 files in **`src`** folder with name beginning the **`Tune_Regressor_`** belongs to this group. The full names are self-explanatory on which model can be tuned in the file, for example **`Tune_Regressor_Ridge.py`** is for tuning of Ridge regression. These 5 files can be run as standalone, or as a package using **`regressions.sh`** BASH executable script found in base folder.
	2.	 **Tune_Classifier_** – All 4 files in **`src`** folder with name beginning the **`Tune_Classifier_`** belongs to this group. Similarly, the full names are self-explanatory on which model can be tuned in the file. These 4 files can be run as standalone, or as a package using **`classifiers.sh`** BASH executable script found in base folder. 
	3.	**Tune_Bins_** - The remaining 3 files starting with **`Tune_Bins_`** in the file names are used to figure out the multi-class configuration (i.e. the edges of the price bins), which will be further explained in _Section F_. These 3 files can be run as standalone, or as a package using **`bins.sh`** BASH executable script found in base folder.


# Section B: Instructions

To execute the pipelines, enter **`./run.sh`** in the base folder and the rest will happen “under the hood” without any need for further manual commands. The programs’ output will be displayed on screen, and also written into files stored inside data folder (the file names are described earlier under _Section A_).

For hyper-parameters tuning, you will need to use the files with name beginning the **`Tune_Regressor_`** as well as **`Tune_Classifier_`** inside **`src`** folder. These are standalone programs, but you can also run then as a batch, as described under _Section A_.

For hyper-parameters tuning, you can edit the codes to adjust the range of hyper-parameters within the params variable within **`main()`** function. The programs’ output will be displayed on screen, and also written into files stored inside data folder. The file names will be displayed on screen when the program completes the run.


# Section C: Pipeline Flow:

After executing **`run.sh`**, the following will happen automatically and sequentially:

**Regressions_Assemble.py starts running**

1.	**Connects to `data/home_sales.db`** – download the raw data into the program. 
2.	**Data Cleaning** – The raw data will be filtered (data cleaning steps as outlined in **`EDA.ipynb`**) for null entries and unnecessary attributes by calling the custom function **`clean_data(df)`**
3.	**Feature Engineering** – The cleaned data will undergo feature engineering as outlined in the **`EDA.ipynb`** file.
4.	**Pre-Processing** – The target variable undergoes normalization using log function to correct for skewness. The numerical features will also be scaled to improve performance of regression, and the categorical features will undergo encoder so that regression can be performed.
5.	**Models Assembly** – 5 regression models are assembled, with the tuned hyper-parameters. A list of tuple of the model’s name cum model are formed under name **`regressors`**.
6.	**Train_test_split** – the data is split into training and testing sub-sets, test size is set at 25%.
7.	**Fit and Predict** – Each model is fitted and predictions are made. The predictions are scored and results are printed on screen as well as saved into file (as described earlier). This step is repeated for all regressors in a python for loop.
8.	**Program exits and return to `run.sh`**, which executes the next python program.

**Classifiers_Assemble.py starts running**

9.	**Connects to `data/home_sales.db`** – download the raw data into the program. 
10.	**Data Cleaning** – The raw data will be filtered (data cleaning steps as outlined in **`EDA.ipynb`**) for null entries and unnecessary attributes by calling the custom function **`clean_data(df)`**
11.	**Feature Engineering** – The cleaned data will undergo feature engineering as outlined in the **`EDA.ipynb`** file.
12.	**Pre-Processing** – The target variable undergoes normalization using log function to correct for skewness. The numerical features will also be scaled to improve performance of regression, and the categorical features will undergo encoder so that regression can be performed.
13.	**Models Assembly** – 4 classification models are assembled, with the tuned hyper-parameters. A list of tuple of the model’s name cum model are formed under name **`clfs`**
14.	**Train_test_split** – the data is split into training and testing sub-sets, test size is set at 25%.
15.	**Fit and Predict** – Each model is fitted and predictions are made. The predictions are scored and results are printed on screen as well as saved into files (as described earlier). This step is repeated for all classifiers in a python for loop. **`run.sh`** exits, returns to command prompt


# Section D: Overview of EDA

The following are key takeaways from EDA:

1.	**Longitude and Latitude** – I decided to discard this 2 attributes because the zip code attribute serves the same function. However, I think that I was too hasty to make this decision because in the USA, the zip code covers a much bigger area as compared to Singapore’s postal code (which I only just realized). So what I should have tried is to come up with grid square categories that categorize every longitude-latitude pair, and then use this as category a feature. I will try after the submission.
2.	**Dates** - Although machine learn is generally not useful for extrapolation and dates are typically discarded, I decided to retain the information on month in hope that there is some seasonality effect on prices, given that there is obvious seasonality effect on transaction volume.
3.	**Missing Data** – The steps for cleaning up the missing data are detailed in EDA.
4.	**Data Types** – A number of attributes uses floating data type, in which I changed into integer type to reduce computational load.
5.	**Skew** – Some of the numerical attributes have obvious skew, which can be corrected by applying log function (or **`log1p`** to account for values less than 1).
6.	**Correlations** – The correlation heatmap does not show up any major concern. However, there seem to be fairly low correlations between the target attribute (i.e. price) and the other attributes.


# Section E: Model Selection

I have used 5 types of regression and 4 classification methods. The scores and comments of all methods are shown below.

On the surface, the classification results on the whole look terrible compared to regression methods. However, looks can be deceiving because the best regression method produces an RMSE of 168,000, which means that the average error expected is $168,000. Comparatively, classes (or price bins) used in classification methods has range as small as $50,000, so we should not discard classification so quickly. The key issue here may be that the features used seemed to be insufficient to perform the price prediction task, and this topic will be covered in _Section G_.

**Regression:** 5 regression methods are chosen, results are tabulated below, followed by comments:

|**Score Type**|**OLS**|**OLS (unskewed)**|**Ridge**|**Lasso**|**KNeighbors**|**Decision Tree**|
|--|--|--|--|--|--|--|
|**RMSE**|212,000|168,000|169,000|169,000|238,000|229,000|
|**R2**|0.684|0.801|0.799|0.799|0.603|0.631|


1.	**Ordinary Least Square (OLS)** – The most common linear regression is tried first without any adjustment to “unskew” the price attribute. RMSE achieved is 212,000 and R2 is 0.684. This RMSE implies an average error of $212,000, which is not great for making any predictions when the median price of the dataset is $450,000.
2.	**OLS (unskewed)** – The price attribute is unskewed by applying a log function. RMSE improved noticeably to 168,000 and R2 is 0.801. Even though R2 is much closer to 1.00, the RMSE is still too large for comfort. I further applied log functions to the other skewed features (detailed in EDA), but there is no more improvements to both R2 and RMSE. As such I only apply log function on the price attribute for the other regressions.
3.	**Ridge** – used to see if regularization would help improve the predictions. The best RMSE and R2 scores come in at 169,000 and 0799, just below that of OLS. This is not surprising given that correlation check on in EDA did not reveal any collinearity, so we should not expect Ridge to be much better than OLS.
4.	**Lasso** – Not surprisingly, this method also did not perform any better than OLS, with the best RMSE and R2 scores very close to that of Ridge. This is because at alpha parameter set of almost zero, Lasso regression is essentially OLS.
5.	**KNeighbors** – This uses data clusters to guide the regression, which is different technique of the traditional OLS regressions, so I thought it may provide better results. Turns out not to be the case. The best n_neighbors is found at 9, with RMSE at 238,000 and R2 at 0.603. I suspect that this could be due to insufficiency of attributes, which will be discussed in _Section G_.
6.	**Decision Tree** – This method uses a hierarchy of if-else questions to fit the model, which somewhat mimics a part of human decision making process (the other part being emotional process, which is much harder for machines to emulate). I thought this different regression approach will offer better scores, especially I think that intuitively, Decision Tree “behaves” like logical humans, but again the best Decision Tree can do is 229,000 for RMSE and 0.632 for R2.

**Classification:** 5 classification methods are chosen for testing using 11 price bins (classes), the results are tabulated below, followed by comments on each method.

|**Score Type**|**MLP**|**SVM**|**Gradient Boosting**|**KNeighbors**|
|--|--|--|--|--|
|**Accuracy**|0.407|0.400|0.384|0.321|
|**Precision**|0.397|0.400|0.381|0.306|
|**Recall**|0.407|0.400|0.384|0.321|
|**F1 score**|0.395|0.399|0.370|0.304|

1.	**MLP** – This is a simple neural network classification method, which performs the best amongst the 4 classification methods used in this report. Initially I thought that given the large number of data, this method would perform better with more hidden layers, but this is not the case. In fact a lower number of 30 hidden layers performed slightly better than if there are 100 or more hidden layers, indicating that the best model should be less complex
2.	**SVM** – This method attempts to find non-linear relationship in the dataset, and it performs on par with MLP. Interestingly, SVM performance worsen significantly as C and gamma increase, i.e. the problem set fairly linear with very limited complex surfaces.
3.	**Gradient Boosting** – Multiple decision trees are used to find the best predictions. As I increase the number of estimators (trees), the score approaches that of MLP, but at a brutally slow pace of generating the predictions as compared to MLP.
4.	**KNeighbors** – This is similar to KNeighbors in regression, and the comparative scores are just as subpar, which is not unexpected.


# Section F: Regression or Classification?

Putting aside prediction accuracies, I would think that both methods have their utilities. Regressions are great to provide a discrete price prediction that allows for property transactions to take place. Home owners and buyers need a definite and clear transaction price point, and mortgage bankers will need to decide on the loan quantum.

However, the predictions using classification methods provides a price range that can be useful for property agents, home owners and potential buyers to maintain psychological flexibility.  For example, the property agents may prefer to tell their clients indicative price ranges for their properties, so as to not lock themselves into a rigid price point.

However, this comes to the point of price bin edges, which is discussed in greater detail in _section 9_ of **`EDA.ipynb`** file. We need to balance user-friendliness of the price bins (classes) while also taking care of increasing prediction errors as the price bins are narrowed. With an eye on ensuring there is limited data imbalance across the classes, as well as achieving better accuracy-precision scores, I have decided to use 11 classes (price bins).


# Section G: Other Considerations

1.	**Segmentation** – segment the properties into at least 2 or 3 groups according to key attributes such as size or prices, i.e. high-value, mid-range and low-end. This may help machine learning algorithms to achieve better predictions because each housing-type segment has their own unqiue characteristics, and if lumped together, will result in unnecessary complications and prediction errors (such as large RMSE). 
2.	**Unsupervised Learning for Clustering** – One possibility is to use unsupervised learning to help form the major clusters, and then we use supervised learning to do more refined predictions within each major cluster on its own.
3.	**More Attributes needed** – Although there are a total of 17 independent attributes in the database, we can only count 14 as truly useful. From personal experience, properties’ valuation depends on several other major factors, such as proximity to amenities like shops, parks, good schools, good jobs etc. Other factors such as whether the house has sufficient parking garages, condition of house garden, total built-in area, condition of kitchen are also key considerations. Perhaps with additional important attributes included in the dataset, the machine learning algorithms can perform significantly better.
4.	**Longitude and Latitude attributes** – Thinking back, I have been too hasty to discard both attributes. Probably I can form grid squares to categorize the longitude-latitude pairs. This may provide better predictions as compare to USA zipcode feature may cover too wide an area.
5.	**Deployment** – In today’s mobile age, a web site that has responsive design is important element of deployment of the predictive model. A mobile APP can also be considered if funding is sufficient as APPs requires more maintenance. The best model can be selected and packaged as part of a mobile APP, and the latest monthly transactions are use on backend to update the model, after which the updated model is pushed out as APP update. 
6.	**Continuous Data Updates** – Given that machine learning cannot reliably extrapolate, this model will need to be updated continuous so that it can remain relevant. A system will need to be designed to ensure that the raw data in each update can be reliably cleaned and checked for outlier and errors. This brings up the next point on what to do with older data.
7.	**Aging Data** – As new data continue to enter the database, should we discard the old data, or assign lower weightages to aging data? If the rate of discard of aging data is too rapid, then we may be losing valuable information because properties are not high volume items, sometimes it help to take reference to prices of similar properties in the neighborhood that are transacted up to 5 or 10 years ago. On the other hand, if we don’t discard old data, it may degrade the predictive power of the models.
