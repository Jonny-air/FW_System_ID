# FW_System_ID
python scripts to automate the fitting of a model to real flight data

Steps:
...



Generating the lookup Table:
Check whether all the cases have converged. If not that means that the requested steady state cannot be obtained within the tolerances. Revisit the ID parameters, often adjusting CD2 to a point where it gives realistic results makes sense. 

Checking the lookup table:
The Min/Max Sink and Min/Max Sink + points should be on a horizontal line. The flight path and pitch angles should be within tolerances. The airspeed constraints should make sense and the cruise airspeed should be somewhat centered between the minimum and the maximum airspeed to get a good middle point for the interpolations. The Min Sink point should correspond to a negative pitch, otherwise the ID is possibly wrong. 
