# FinalMachineLearning

This project utilizes Linear regression, SVR, Random forest, ARIMA, SARIMA and Holt Winters exponential smoothing to forecast hourly weather data.
We take in one month of hourly data as train data, run each model and test on a seperate validation set, then compare root mean squared error.

Below is the link to a google slidshow going through what this project is. 
https://docs.google.com/presentation/d/1K0o1jeUx8h5qf2DxzTMDn3KgiUY_LKQwvq-sRd3KZPI/edit?slide=id.p#slide=id.p 

Here is a link to the dataset we used
https://www.ncei.noaa.gov/access/search/data-search/normals-hourly-2006-2020?dataTypes=HLY-HTDH-NORMAL&dataTypes=HLY-DEWP-NORMAL&dataTypes=HLY-HIDX-NORMAL&dataTypes=HLY-PRES-NORMAL&dataTypes=HLY-PRES-10PCTL&dataTypes=HLY-PRES-90PCTL&dataTypes=HLY-TEMP-NORMAL&dataTypes=HLY-TEMP-10PCTL&dataTypes=HLY-TEMP-90PCTL&dataTypes=HLY-WCHL-NORMAL&dataTypes=HLY-WIND-AVGSPD&dataTypes=HLY-WIND-VCTDIR&dataTypes=HLY-WIND-VCTSPD&dataTypes=HLY-WIND-1STDIR&dataTypes=HLY-WIND-1STPCT&pageSize=100&pageNum=1&startDate=2025-04-09T00:00:00&endDate=2025-04-16T00:00:59&bbox=47.232,-95.645,41.820,-87.426 
