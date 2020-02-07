# 

data cleaning
eda 
random forest
metric - recall score equation -> 1 - FNR ; goal (minimize) false negative rate
why - bc we don't want to incorrectly predict someone would stay but in realilty they left

goal - see what feature is giving us the lowest recall score (allowing the most FN)


# Predicting Employee Turnover

# Table Of Contents
1. [Overview/Background](#overview/-background)
2. [Project Questions/Goal](#project-questions/goals)
3. [Data](#the-data)
4. [Exporatory Data Analysis (EDA)](#exploratory-data-analysis-(eda))
5. [Analysis](#analysis)
6. [Conclusion](#conclusion)
7. [References](#references)



# Overview/ Background

[The Society for Human Resource Management](https://www.peoplekeep.com/blog/bid/312123/employee-retention-the-real-cost-of-losing-an-employee)
(SHRM) did a study on the cost of employee
turnover and based on their predictions every time a business replaces
a salaried employee, it costs 6 to 9 months' salary on average.

Example: 

A manager making $40,000 a year, that's $20,000 to $30,000 in recruiting and training expenses.

This factors in negatively for a business both in time and money. Thus, I wanted to explore
what indicators contribute towards employee turnover.


# Project Questions/Goals

My goals for this project was to answer the following question:

Predict Employee Turnover

What indicators contribute towards employee turnover?

What can company X do to minimize employee turnover.

# The Data

The data for this project was pulled from kaggle from [Hr Analytic](https://www.kaggle.com/lnvardanyan/hr-analytics)

The dataset was 14999 rows, 10 columns

![turnover-dataset-p1](turnover_df_slice_1.png)
Dataset the first 4 columns

![turnover-datset-p2](turnover_df_slice_2.png)
Dataset the first 6 columns

![salary-encode](salary_encoded.png)

![department-encode](department_encoded.png)


# Exploratory Data Analysis (EDA)

![satisfaction-level](satisfaction_level_percentage.png)

![last-eval](last_evaluation_percentage.png)

![number-projects](Employer_Turnover_by_number_project.png)

![avg-monthly-hours](average_monthly_hours.png)

![time-spend-company-years](Employer_Turnover_by_time_spend_company_years.png)

![work-accident](Employer_Turnover_by_Work_accident.png)

![promotion-last-5](Employer_Turnover_by_promotion_last_5years.png)

![department](Employer_Turnover_by_Department.png)

![salary](Employer_Turnover_by_Salary_rank.png)

# Model

For my model tried a variety of models, logisticRegression, pca, decision tree and random forest.
I found that the random forest model works the best 

### Metric Used
At first I was using accuracy as metric but then found recall to be the best option.
Because our main objective is to minimize false negative (type II error) - 
Prediciting an employee would stay (and not do anything about it) but in reality
they left, thus loss time and money.

### First Run
Recall-Score = 0.96

### Suspicion of Data Leakage

After running my first Recall Score, I became skeptic there may be data leakage. Long Story Short,
I tried stripping out different Feature Imporances.

![feat-imp](perc_by_feat_imp.png)

Next, I ran the recall scores.

Lastly, I checked for duplicates in my data.
I found 20% to be duplicated, dropping my dataset to [11991 rows x 9 columns]

![recall-scores-plt](Recall_b_a_data_leakage.png)


# Conclusion

In conclusion, my random forest classifier model was performing very
well which lead suspicion there was data leakage due to duplicates in my dataset.
After fixing the error, I was still proud of my models recall score performance of 90%.

As a reminder, I used all 9 features on my model. The top 2 indicators based on feature importance were:

* Satisfaction level (percentage) 34%
* Number of Projects (percentage) 19.5%

### In the Future

In the future, I would like to create a profit curve

Create Partial Dependence Plots

Check if number of projects and avergage monthly hours are colinear


# References
https://www.peoplekeep.com/blog/bid/312123/employee-retention-the-real-cost-of-losing-an-employee 

