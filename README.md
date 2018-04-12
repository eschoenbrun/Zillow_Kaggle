Problem Statement: 

Zillow, a real estate database company, offers a proprietary valuation algorithm that estimates the market price of a property for its users.  In efforts to improve this ‘Zestimate’ algorithm, Zillow initiated a Kaggle competition by releasing 2.9 Million properties in three California counties with 58 property characteristics as decision variables.  The participants’ goal was to forecast each property’s value and measure the model’s accuracy by comparing those values to real sale prices of properties sold in October through December in 2016/2017.  The only constraints were competition deadlines and exclusion of external data.

Results: 

We found that by using Lasso Regression, we obtained an error margin of 0.0806, placing our model in the 58th percentile of submissions and only 0.0019 from the leading submission.  The following was our process for data analysis, modeling, and alternate trials.

Data Cleaning and Wrangling: 

An assessment of the data discovered that several variables were extremely sparse and therefore removed. Likewise, there were variables whose majority was labeled as zero values incorrectly, and also removed.  With the other variables that had missing or zero data, we found it pragmatic to impute these missing observations owing to the correlation with the target variable. We would have liked to impute these values with methods like KNN or Random Forest, but with limited computing resources, we chose to use median values for numeric variables and most frequent values for categorical variables.  Additionally, with research into the data dictionary we found that there were variables that were nearly identical or could be computed from other variables, and in those instances, we removed the more sparse variable.  Lastly, for our machine learning techniques to run, we encoded the categorical variables to create dummy variables for all variable. This left us with 156 variables and 2.9 million rows.

Data Modeling: 

Although ensemble machine learning techniques such as random forest are typically robust algorithms that yield good accuracy and generalizing well over unseen data, they require significant computing power. Therefore, we assessed these algorithms’ capabilities by sampling a subset of the most important variables and used the resulting models to predict the remaining population.  This technique returned a 0.073 MAE for our Random Forest model.

Lasso regression, an improvement on linear regression, drops variables that offer little predictive value, reducing the computational expense substantially, and allows us to use the entire data set.  Additionally, because it only retains the most significant variables and assigns coefficients to them, it is easier to explain the model to stakeholders.  The resulting model produce the most accurate prediction of property values on our validation data, giving us a 0.069 MAE.  We focused our efforts on tuning the model’s parameters for optimal efficiency and submitted two different lasso regression models. The initial test data gave us a MAE of 0.063 and placing us in the 78th percentile. As more of the test data has been released monthly as more houses are sold, we have improved to the 58th percentile, even though our model has increased to a MAE of 0.0806.

Other Techniques: 

After our submission, we followed up with several other methods, such as Polynomial Regression and tried to reduce dimensionality with Principle Component Analysis, however neither of these improved our score.
