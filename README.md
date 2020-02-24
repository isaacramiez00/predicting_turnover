# Predicting Employee Turnover

# Table Of Contents
1. [Overview/Background](#overview/-background)
2. [Project Questions/Goal](#project-questions/goals)
3. [Data](#the-data)
4. [Exporatory Data Analysis (EDA)](#exploratory-data-analysis-(eda))
5. [model](#model)
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

The data for this project was pulled from kaggle from [Hr Analytics](https://www.kaggle.com/lnvardanyan/hr-analytics)

The dataset was 14999 rows, 10 columns

![turnover-dataset-p1](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/turnover_df_slice_1.png)

Dataset the first 4 columns


![turnover-datset-p2](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/turnover_df_slice_2.png)

Dataset the first 6 columns


![salary-encode](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/salary_encoded.png)


![department-encode](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/department_encoded.png)


# Exploratory Data Analysis (EDA)

![satisfaction-level](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/satisfaction_level_percentage.png)

What we ca do: Try to keep satisfaction level above 0.50

![last-eval](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/last_evaluation_percentage.png)


What we can do: Keep Projects to a minimum, spread amongs employee

![avg-monthly-hours](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/average_montly_hours.png)

My Interpretation: Work performance rate
What we can do: Check up after and see how the company can help increase work performance

![number-projects](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Employer_Turnover_by_number_project_side_barcharts.png)


What we an do: Try and keep the average monthly hour between 150-250 hours (40 - 60 per week)

![time-spend-company-years](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Employer_Turnover_by_time_spend_company_years_side_barcharts.png)

What we can do: Encourage new type of roles within company if employee is past 5 year benchmark, (possibly boredism)

![work-accident](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Employer_Turnover_by_Work_accident_side_barcharts.png)


![promotion-last-5](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Employer_Turnover_by_promotion_last_5years_side_barcharts.png)

What we can do: If we work on increasing work performance (maybe by motivating promotions) and doing so in 5 years

![department](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Employer_Turnover_by_department_side_barcharts.png)

What we can do: Look into the HR department, see how business operations are doing. This
should give valuable insight to see how other departments are operating as well because of the vary similar turnover levels.

![salary](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Employer_Turnover_by_salary_side_barcharts.png)

What we can do: Increase Employee salary. Who doesn't want more money?

# Model

My data was a classifier model.

For my model tried a variety of models, logisticRegression, pca, decision tree and random forest classifier.
I found that the Random Forest Classifier model works the best.

### Metric Used

At first I was using accuracy as metric but then found recall to be the best option.
Because our main objective is to minimize false negative (type II error) - 
Prediciting an employee would stay (and not do anything about it) but in reality
they left, thus loss time and money.

Goal: To have a the highest Recall Score value

### First Run
Recall-Score = 0.96 


### Suspicion of Data Leakage

After running my first Recall Score, I became skeptic there may be data leakage. Long Story Short,
I tried stripping out different Feature Imporances.

![feat-imp](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/perc_by_feat_imp.png)

Next, I ran the recall scores.

Lastly, I checked for duplicates in my data.
I found 20% to be duplicated, dropping my dataset to [11991 rows x 9 columns]

I reran the recall scores per feature importance and found none were a significant impact.

![recall-feat-p1](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/recall_feat_p1.png)


![recall-feat-p2](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/recall_feat_p2.png)


![recall-scores-plt](https://github.com/isaacramiez00/predicting_turnover/blob/master/imgs/Recall_b_a_data_leakage.png)


# Conclusion

In conclusion, my Random Forest Classifier model was performing very
well which lead suspicion there was data leakage due to duplicates in my dataset.
After fixing the error, I was still proud of my models recall score performance of 90%.

As a reminder, I used all 9 features on my model. The top 2 indicators based on feature importance were:

* Satisfaction level (percentage) 34%
* Number of Projects (percentage) 19.5%

As far as answering what can Company X do to minimize employee turnover, based looking at our exploratory data
the big two are assure satisfaction levels are good lower and number of projects. Other things we can do are
promote within 5 years, check into how business operations are doing in the HR department, increase salary, keep monthly hours
between 150-250, see how employers can help increase work performance.

### In the Future

In the future, I would like to create a profit curve to display the costs of employee turnover.

Create Partial Dependence Plots to illustrate the relationships between features and the predictions of a employee turnover.

Have a test script written out for Frank.


# References
Merhar, Christina. “Employee Retention - The Real Cost of Losing an Employee: 2019.”
PeopleKeep, www.peoplekeep.com/blog/bid/312123/Employee-Retention-The-Real-Cost-of-Losing-an-Employee.
