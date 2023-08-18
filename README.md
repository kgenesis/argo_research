# Studying the Equatorial Pacific with BGC Argo Float and Satellite Data

The aim of this project was to understand the effect of tropical instability waves on nutrient concentrations in the equatorial pacific as they relate to the biological pump and carbon cycle. The equatorial pacific is an under researched area and many questions still remain on what drives productivity in this region. My interest in this area of research pertains to: deviations from the Redfield ratio, limiting nutrients, drivers on upwelling, and factors that influence seasonal nutrient cycles. Because this region is still poorly understood, this research has only investigated how nutrient concentrations differ from inside a TIW vs outside with initial results showing an increase in nitrate without correlating chlorophyll or oxygen concentrations. This might indicate phytoplankton growth without nitrate, possibly suggesting that iron is a limiting nutrient. 

Methods included developing a TIW index using functions that took argo float profiles and 
a) filtered out months 3-5 to only look at summer-winter seasons which is when TIW are known to occur using pandas date_time 
b) selected times where there was a sea level anomaly 2sigma greater than the mean between 3-8ºN around 10º E and W of where the argo float was at that time, suggesting that there was a TIW passing in close proximity to the float at the time it surfaced 
c) selected times when there was a sea surface temperature anomaly 2sigma less than the mean 10º N, S, E, W of where the float was at that time, suggesting the float was in an equatorial cold core at the time it surfaces. These areas are known for there high mixing rates
d) selected times when there was a delSST anomaly (delSSTx^2 + delSSTy^2 was at its maxima) indicating the drastic change in temperature around a leading edge front, which is an area known for it's downwelling. This function was not finished, as I have not yet figured out how to identify an argo float within the LEF calculated area
e) still need to define other structures within the TIW
f) floats that were not in any of these structures but surfaced at times when there was a TIW (as indicated by the season and SLA anomoly) are taken to be in a transition zone
g) nutrient concentrations were calculated for floats surfacing in each of these structures and compared to the rest of the data, at first only looking at float 73 around 140W between 0-2ºN, but hopefully expanding to floats that occur all across a TIW and would surface in the different types of structures at, above, and below the equator. 

Next Steps for this part of the project:
a) research and understand the different types of structures that occur within the TIW and find ways to identify them and colocate argo floats within them
b) compare nutrient data and identify causes for patterns in upwelling/mixing
c) would like to gain a better understanding of the 4D TIW profile by looking at how these structures develop and change over time and how nutrients are mixed/cycled through them (would look at timescales for structure occurance vs nutrient occurange at given lat/lons)

## Installation

download satellite data from ERDAPP NOAA database
download float data from GO_BGC interactive map or use code provided in bgc-workshop google colab
functions used for colocation found in Utils, created and analyzed in BGC_float_SST_colocation and colocation_all_floats
methods for plots found in Trop_Pacific_Ref or sensor_plots
QC check in QC_flags
environment in argo_env

## Usage

functions described as above for colocation
plots for seasonal cycles, argo colocation in floats, and nutrient profile depths have been used and code can be found in notebooks. Examples found in folder. 

## Contributing
Kalena Genesis, Yassir Eddebar, Amy Waterhouse, Gunnar Voet... potentially David Archer and Malte Jensen

## License 
