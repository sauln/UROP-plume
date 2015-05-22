## UROP-plume

Repository of the dynamic plume simulation used in 

```
M. Fahad, N. Saul, B. Bingham, Y. Guo. "Robotic Simulation of Dynamic Plume Tracking by Unmanned Surface Vessels," IEEE International Conference on Robotics and Automation,  To Appear, May 2015.
```


To generate plume data, run genSim.py with a keyword argument of where you want the data saved.  Note, do not write the entire filename, just the root.  ie with keyword data-set1-, the data will be saved as data-set1-1.py, data-set1-2.py, and so on until the generation is complete.

```
python genSim.py data/data-set1-
```

To run the simulation, use the runSim.py with keyword argument of the data to be used, ie 
```
python runSim.py data/data-set1-
```

The simulation will automatically find the first set of data and load subsequent files when it need to.

Please look in genPlots.py for different examples on how to access the data sets without running the full simulation.
