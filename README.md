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

### metric used
At first I was using accuracy as metric but then found recall to be the best option.
Because our main objective is to minimize false negative (type II error) - 
Prediciting an employee would stay (and not do anything about it) but in reality
they left, thus loss time and money.

![roc-curve]()


### Data Leakage
After Questioning my data - I found data leakage... 
I performed feature importances
Ran Recall scores
Lastly, I performed duplicates 
I found 20% to be duplicated, dropping my dataset to [11991 rows x 9 columns]



# Conclusion

In conclusion, there is no significant difference between urban homes and suburban homes for a house hack. We also found that
the most common types of property for both living areas are 3 bedroom single family homes. The average purchase price for each
home are above $550,000.

The average initial investment for both communities are on average above $50,000.

The average cashflow for a suburban home is $-14/month, Which means you would still pay $14 to for your living situation.
The average cashlow for an urban home is $250/month.

The average NWROI for both communties is at slightly above 100% (impact the deal has on your networth).

#### In the Future
In the future, I would like to answer if single family homes cash flow the most compare to the rest of property types.

I'd like to bootstrap and find the median of for all analysis next time to eliminate any skewdness.

I would also like to explore more in depth what the minumum purchase,rent, and initial investment would be to break even on house hack.

Lastly, explore which relationship has the stongest correlation with monthly cash flow.

# References
https://www.peoplekeep.com/blog/bid/312123/employee-retention-the-real-cost-of-losing-an-employee 

