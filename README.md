# Introduce
Imagine we are the member of a Retail/Telcos company and we wants to fuel its growth by Data-driven approach. This Series has 6 part research about 6 core problem of business:
- Part 1: [Customer Segmentation](https://github.com/ToanToan110/CustomerSegmentation)
- Part 2: [Churn Prediction](https://github.com/ToanToan110/ChurnPrediction)
- Part 3: [Customer's Life Time Value](https://github.com/ToanToan110/CustomerLifeTimeValue)
- Part 4: [Sales Prediction](https://github.com/ToanToan110/SalesPrediction)
- Part 5: [Market ResponseModel](https://github.com/ToanToan110/MarketResponseModel)
- Part 6: [A/B Testing](https://github.com/ToanToan110/A-B-Testing)

# About this Project
By understanding the current business situation based on the human level (such as Lifetime value, Churn rate, Segment,...) businesses can expand the business picture through capturing sales, viewing Consider how each customer-specific strategy affects sales.

From there, make financial, investment, marketing plans and calculate uplift value.

***Main techniques used:***

Time series Forecasting

# Pre-requisites
- Dataset: This notebook use [Sales Dataset](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) of a competition on Kaggle

References:

- [Prediction Sales](https://towardsdatascience.com/predicting-sales-611cb5a252de)

# Explodatory Analysis 
The overall trend of sales by date of the dataset:

![download](https://github.com/ToanToan110/SalesPrediction/assets/64849001/5a2bd335-b12e-4443-84f7-e7601c1e64b8)

- We can see that the trend looks like inscrease by year and the time series has the seasonality characteristic.
# Feature Engineering
With the seasonal time series, we usually use the lag as a features for prediction task.
  
