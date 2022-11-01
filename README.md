# House Rocket - Real Estate Business Insights

This repository contains codes for the porfolio analysis of a real estate company. <br>
All information below is fictional.

## Objectives
* Perform exploratory data analysis on properties available on dataset.
* Determine which properties should be purchased according to business criteria.
* Develop an online [dashboard](https://house-rocket-analytics-rnd.herokuapp.com/) that can be acessed by the CEO from a mobile or computer.
<br>

## 1. Business Problem
House Rocket business model consists of purchasing and reselling properties through a digital platform. The data scientist is in charge to develop a online dashboard so that the company's CEO can have an overview of properties available to became part of House Rocket portfolio in King County (USA).<br>

The [dashboard](https://house-rocket-analytics-rnd.herokuapp.com/) must contain:

   * A table view with attributes filters. 
   * A map view with properties available.
   * A visual distribution of properties by key attributes.
   * The sugested investment relation.
   * Expected investment outcome.<br><br>

## 2. Business Results
There are 21,436 available properties. Based on business criteria, 310 selected properties should be bought by House Rocket resulting on a US$60M profit.<br>
Necessary Investment: US$131,440,127.00<br>
Estimated Revenue: US$192,282,831.65<br>
Estimated Profit: US$60,842,704.65<br>
Estimated Margin: 31.46%

## 3. Business Assumptions
* The data available is only from May 2014 to May 2015.
* Seasons of the year:<br>
   * Spring starts on March 21st<br>
   * Summer starts on June 21st<br>
   * Fall starts on September 23rd<br>
   * Winter starts on December 21st<br>
* Business criteria to determine whether a property should be bought are:
    * Propertie price is bellow regional median.  
    * Propertie condition is above regional median.  
    * Propertie living area is above regional median.  
    * Propertie living area is above regional median.

<details><summary>The variables on original dataset goes as follows:</summary><br>

Variable | Definition
------------ | -------------
|id | Unique ID for each property available|
|date | Date that the property was available|
|price | Sale price of each property |
|bedrooms | Number of bedrooms|
|bathrooms | Number of bathrooms, where .5 accounts for a room with a toilet but no shower, and .75 or ¾ bath is a bathroom that contains one sink, one toilet and either a shower or a bath.|
|sqft_living | Square footage of the apartments interior living space|
|sqft_lot | Square footage of the land space|
|floors | Number of floors|
|waterfront | A dummy variable for whether the apartment was overlooking the waterfront or not|
|view | An index from 0 to 4 of how good the view of the property was|
|condition | An index from 1 to 5 on the condition of the apartment|
|grade | An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.|
|sqft_above | The square footage of the interior housing space that is above ground level|
|sqft_basement | The square footage of the interior housing space that is below ground level|
|yr_built | The year the property was initially built|
|yr_renovated | The year of the property’s last renovation|
|zipcode | What zipcode area the property is in|
|lat | Lattitude|
|long | Longitude|
|sqft_living15 | The square footage of interior housing living space for the nearest 15 neighbors|
|sqft_lot15 | The square footage of the land lots of the nearest 15 neighbors|
</details>
<details><summary>Variables created during the project development goes as follow:</summary><br>

## 4. Solution Strategy
1. Understanding the business model
2. Understanding the business problem
3. Collecting the data
4. Data Description
5. Data Filtering
6. Feature Engineering
8. Exploratory Data Analysis
9. Insights Conclusion
10. Dashboard deploy on [Heroku](https://house-rocket-analytics-rnd.herokuapp.com/)
<br>

## 5. Top 3 Data Insights
1. The number of properties built with basements decreased after the 80s.
2. Almost 60% of the properties became available during summer/spring.
<br>

## 6. Conclusion
The objective of this project was to create a online dashboard for the House Rocket's CEO. Deploying the dashboard on Heroku platforms provided the CEO acess from anywhere, facilitating data visualization and business decisions.
<br><br>

## 7. Next Steps
* Determine which season of the year would be the best to execute a sale.
* Implement a Machine Learning algorithim to define selling proces and increase revenue.
<br>

---
## References:
* Dataset House Sales in King County (USA) from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)
* Variables meaning on [Kaggle discussion](https://www.kaggle.com/harlfoxem/housesalesprediction/discussion/207885)
* Python from Zero to DS lessons on [Youtube](https://www.youtube.com/watch?v=1xXK_z9M6yk&list=PLZlkyCIi8bMprZgBsFopRQMG_Kj1IA1WG&ab_channel=SejaUmDataScientist)
* Blog [Seja um Data Scientist](https://sejaumdatascientist.com/os-5-projetos-de-data-science-que-fara-o-recrutador-olhar-para-voce/)
