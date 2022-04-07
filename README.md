# Peak detection 
This script was used to develop a specific functionality for our kinematics data analysis.
Here is a brief description of the experiment and of the desired output, to provide context.
I left two test dataset in the test_data folder.
## Kinematics experiment
We apply reflective markers on the joints of a rat and let him run on a runway. At first, we get
the 3D coordinates of each marker, which need to be parsed in single steps. To do this we have to
look at the signal coming from the toe marker and detect the moment the toe
lifted and the moment it landed back on the runway. This is the input data of the
script. 

The aim is to identify the correct intervals on the time series that correspond to the step.
This goes approximately from the small inflection point on the ascending part of the peak,
to the lowest point in the descending part.