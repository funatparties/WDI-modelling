## World Development Indicator (WDI) Modelling
### Overview
This project is about collecting and modelling data from The World Bank's database of [World Development Indicators](https://data.worldbank.org/indicator).
The [API](https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information) provided by The World Bank is utilised to collect a variety of statistics
about the countries of the world (such as GDP and population) which are then stored in files locally. Machine learning techniques are then applied to this data to analyse and investigate trends.
### Data Collection
The `data_collection.py` file contains the functions used to access the API and download data (bulk data is available by manual download however this project is 
intended to demonstrate API usage). Each indicator (the measures collected in the database) has a code such as 
[`NY.GDP.MKTP.CD`](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) which is used by the API. A list of codes is kept in the file to specify which data is 
to be downloaded.

Along with the specified indicators, some basic classifications used by The World Bank for each country are downloaded:
* Region
* Income Level
* Lending Type

#### Region
Countries are divided into seven regions:
* Latin America & Caribbean
* South Asia
* Sub-Saharan Africa
* Europe & Central Asia
* Middle East & North Africa
* East Asia & Pacific
* North America
#### Income Level
Countries are divided into five income levels:
* High income
* Upper middle income
* Lower middle income
* Low income
* Not Classified
#### Lending Type
The World Bank provides two primary lending funds for developing nations: The International Bank for Reconstruction and Development (IBRD) 
and The International Development Association (IDA). Based on this, countries are classified into one of four lending types:
* IBRD
* IDA
* IDA Blend
* Not Classified
### Analysis
The functions in `analysis.py` allow the visualisation and application of machine learning techniques on the data. A framework is provided for training and optimising
algorithms suchs as support vector classifiers.

An example is training a support vector classifier to predict Lending Type based on GDP per capita and percentage urban population. The support vector machine used
a radial basis function kernel and a grid search was performed over values of the regularisation parameter `C` and the kernel length scale parameter `gamma`. The
heatmap in the upper left shows mean validation scores under 5-fold cross-validation for each pair of values. Validation curves are plotted with respect to each 
parameter at its optimal value to show the progression of overfitting vs underfitting in each case. Furthermore, since only two variables are being used, a visualisation
of decision boundaries for the best model is included in the upper right along with the training data. <p>
<img src="https://github.com/funatparties/WDI-modelling/blob/master/images/2D%20SVC.png" width="1000">
###### A model summary produced for an RBF support vector classifier. Peak accuracy was 82.5%.
