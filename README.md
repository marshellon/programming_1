
#The correlation between gross national product (GNP) per capita and CO2 emissions per country.

The gross nationale product(GNP) per capital. Source: https://www.kaggle.com/datasets/ammon1/demographic?resource=download

CO2 emmission per capital from 2002-2018. Source: https://www.kaggle.com/datasets/thedevastator/global-fossil-co2-emissions-by-country-2002-2022

About the data:
the emission dataset gives a detailed look at global CO2 emissions at the country level, allowing for a greater awareness of the amount that each country adds to the global combined human impact on climate.
It includes data on total emissions as well as emissions from coal, oil, gas, cement manufacturing and flaring, and other sources.
The data also shows which countries are leading in pollution levels and identifies potential areas where reduction efforts should be focused.
This dataset is critical for anyone interested in learning about their own environmental footprint or conducting research on global development trends.

The GNP dataset include the Gross National Product (GNP) of multiple countries from 1960 to 2018.
The data set is derived from the World Bank and includes columns for population, GDP, inflation, fertility rate, migration, and production, as all of these factors contribute to the calculation of GDP. 


The analysis demonstrates the link between the gross national product and CO2 emissions of various countries around the world.
It accomplishes this by collecting data from both of the previously described datasets and plotting the two variables (GNP and emission) using a Spearman rank correlation test. 

# running the analysis.

The notebook can be opened with the file called  GNP.ipynb
read requirements to see what it needs.\

## requirements

the code requirse the modules:
pandas (for loading and handeling the data)\
numpy  (calculations)\
matplotlib (plotting)\
Summary_functions_and_code_clean_v12 (included needed for statistics)\
seaborn (spearmans test)\
yaml (reading config file)\
The config Yml file can be edited to the correct data path\
The two datasets are called demographic for the GNP data and GCB2022v27_MtCO2_flat for the emission data.\
world-CO2-emission-vs-GDP is the picture loaded in the notebook


The analises uses VS-code in combination with Jupiter notebook 
#Licence
The code is licensed undder the MIT license
