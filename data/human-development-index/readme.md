# Human Development Index - Data package

This data package contains the data that powers the chart ["Human Development Index"](https://ourworldindata.org/grapher/human-development-index?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website.

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For most countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The remaining columns are the data columns, each of which is a time series. If the CSV data is downloaded using the "full data" option, then each column corresponds to one time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data columns are transformed depending on the chart type and thus the association with the time series might not be as straightforward.


## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

### How we process data at Our World In Data
All data and visualizations on Our World in Data rely on data sourced from one or several original data providers. Preparing this original data involves several processing steps. Depending on the data, this can include standardizing country names and world region definitions, converting units, calculating derived indicators such as per capita measures, as well as adding or adapting metadata such as the name or the description given to an indicator.
[Read about our data pipeline](https://docs.owid.io/projects/etl/)

## Detailed information about each time series


## Human Development Index
The Human Development Index (HDI) is a summary measure of key dimensions of human development: a long and healthy life, a good education, and a decent standard of living. Higher values indicate higher human development.
Last updated: May 7, 2025  
Next update: May 2026  
Date range: 1990–2023  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
UNDP, Human Development Report (2025) – with minor processing by Our World in Data

#### Full citation
UNDP, Human Development Report (2025) – with minor processing by Our World in Data. “Human Development Index – UNDP” [dataset]. UNDP, Human Development Report, “Human Development Report” [original data].
Source: UNDP, Human Development Report (2025) – with minor processing by Our World In Data

### What you should know about this data
* The Human Development Index (HDI) provides a broad, intuitive measure for comparing overall human progress across countries and over time. A higher HDI implies longer, healthier lives, better education and higher command over resources; however, it does not capture inequality, sustainability or subjective wellbeing
* It is a composite summary of a country's average achievement in three basic dimensions of human development – health (life expectancy), knowledge (schooling) and material wellbeing (GNI per capita) – combined through a geometric mean into a single 0‑to‑1 score.
* Each dimension of the HDI is captured by a specific index:
  - **Healthy life**: Captured by the _Life Expectancy Index_, which is based on life expectancy at birth.
  - **Good education (knowledge)**: Captured by _Education Index_, which is based on the expected and mean years of schooling.
  - **Decent standard of living**: Captured by _Gross National Income (GNI) Index_, which is based on the GNI per capita (PPP$).
* The index is estimated by normalizing and aggregating the above indicators. First, the indicators are brought onto the same scale, ranging from 0 to 1. This is done by setting minimum and maximum values for each indicator. The minimum and maximum values for each indicator are defined as follows:
  - Life expectancy at birth ranges between 20 and 85 years
  - Expected years of schooling between 0 and 18 years; Mean years of schooling, between 0 and 15 years
  - GNI per capita between 100 and 75,000 international-$ at 2021 prices.
* The HDI is then estimated as the geometric mean of these indices. The education index is the arithmetic mean (average) of the mean years of schooling and expected years of schooling.

### How is this data described by its producer - UNDP, Human Development Report (2025)?
The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development: a long and healthy life, being knowledgeable and having a decent standard of living. The HDI is the geometric mean of normalized indices for each of the three dimensions.

The health dimension is assessed by life expectancy at birth, the education dimension is measured by mean of years of schooling for adults aged 25 years and more and expected years of schooling for children of school entering age. The standard of living dimension is measured by gross national income per capita. The HDI uses the logarithm of income, to reflect the diminishing importance of income with increasing GNI. The scores for the three HDI dimension indices are then aggregated into a composite index using geometric mean. Refer to Technical notes for more details.

The HDI can be used to question national policy choices, asking how two countries with the same level of GNI per capita can end up with different human development outcomes. These contrasts can stimulate debate about government policy priorities.

The HDI simplifies and captures only part of what human development entails. It does not reflect on inequalities, poverty, human security, empowerment, etc. The HDRO provides other composite indices as broader proxy on some of the key issues of human development, inequality, gender disparity and poverty.

A fuller picture of a country's level of human development requires analysis of other indicators and information presented in the HDR statistical annex.

### Source

#### UNDP, Human Development Report – Human Development Report
Retrieved on: 2025-05-07  
Retrieved from: https://hdr.undp.org/  


## World region according to OWID
Regions defined by Our World in Data, which are used in OWID charts and maps.
Last updated: January 1, 2023  
Date range: 2023–2023  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
Our World in Data – processed by Our World in Data

#### Full citation
Our World in Data – processed by Our World in Data. “World region according to OWID” [dataset]. Our World in Data, “Regions” [original data].
Source: Our World in Data

### Source

#### Our World in Data – Regions


    