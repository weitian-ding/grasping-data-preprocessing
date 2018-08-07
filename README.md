# grasping-data-preprocessing

## terminologies
* gripper: 
* polaris: 
* polaris tool1:
* polaris tool2:

## clean and index dataset
[index_dataset.py](bin/index_dataset.py) merges gripper motor recordings and polaris recordings in each grasp, and produces and index for each grasp in the dataset.
### parsing gripper recordings
Gripper recordings for each grasp are written in plain text and saved on disk in a single text file. Below is an example of gripper recordings for one grasp. 
```
1  2018-07-25 21:06:58.207791,15468, 14303, 17585, 15988
2  Time Difference between Labview PC and the Laptop running Gripper(+ive means Desktop is ahead): 49253330
3  2018-07-25 21:07:01.087666,0, 0, 0, 15985
4  2018-07-25 21:07:01.279673,15724, 14052, 17837, 15984
5  2018-07-25 21:07:01.423645,0, 0, 0, 15984
6  Start time: 2018-07-25 21:06:58.207791
7  End time: 2018-07-25 21:07:01.383107
8  Task_time: 3.175316
9  Joystick Displacement mapped to fixed finger positions in gripper
10 S:button 1 pressed-gripping success
11 T:12-precision disk bluelid 1-obs**Fingers at calibration = [14631, 15141, 16749, 16378]
```
The following information is extracted from gripper recordings,
* gripper motor parameters: there is one timestamp and 4 motor parameters (motor[1-4]) associated with each gripper record (e.g. line 1). motor parameters in each gripper file forms a nx4 matrix, `motor_data`, where n is the number of lines which consists of one gripper motor record. If `motor_data[i][j]` is 0, (e.g. line 3), then the value of `motor_data[i][j]` is obtained via linear interpolation between two non-zero values in j-th column of `motor_data`. 
* time difference: the time difference in microseconds between gripper recording and polaris recording (line 2) is extracted to synchronize gripper motor parameters and polaris parameters.
* grip type: `T:[id]` (line 11) denotes the integer-ided grip type
* description: line 10 and line 11 in the above gripper file froms the description of a grasp
* status: a grasp is successful if `success` is indicated (line 10), failure otherwise
### parsing polaris recordings
Polaris recording for each grasp is also written in plain text and saved on disk in a single text file. Below is an example of polaris recordings for one grasp.
```
1  Tool 1 :C:\Program Files (x86)\Northern Digital Inc\8700449.rom
2  Tool 2 :C:\Program Files (x86)\Northern Digital Inc\8700339.rom
3  YCB Object No.:352318
4  Frame No: TimeStamp--Tool 1, 2: Tx,Ty, Tz, Q0, Qx, Qy, Qz, *** Tx,Ty, Tz, Q0, Qx, Qy, Qz
5  Measurement Completed for 25/07/2018 at 9:07 PM.
6
7  Frame 1, 2018-07-25-21-07-47.668569, Both Rigid bodies out of volume?
...
16 Frame 28, 2018-07-25-21-07-48.128595, Both Rigid bodies out of volume?
17 Frame 31, 2018-07-25-21-07-48.177598, Tool  1, -343.663635, -241.315613, -1725.945190, 0.327069, 0.365894, -0.029537, 18 0.870791, Tool 2, 0.000000, 0.000000, 0.000000,  0.000000, 0.000000, 0.000000, 0.000000
18 Frame 34, 2018-07-25-21-07-48.262603, Tool  1, -341.475250, -238.048141, -1726.609863, 0.329228, 0.361876, -0.031009, 0.871605, Tool 2, 0.000000, 0.000000, 0.000000,  0.000000, 0.000000, 0.000000, 0.000000
19 Frame 40, 2018-07-25-21-07-48.342607, Tool  1, -337.957947, -230.865555, -1726.995972, 0.333874, 0.359146, -0.033812, 0.870862, Tool 2, 0.000000, 0.000000, 0.000000,  0.000000, 0.000000, 0.000000, 0.000000
20 Frame 46, 2018-07-25-21-07-48.426612, Tool  1, -335.737122, -223.222626, -1726.810181, 0.338297, 0.361927, -0.032742, 0.868040, Tool 2, 0.000000, 0.000000, 0.000000,  0.000000, 0.000000, 0.000000, 0.000000
21 Frame 49, 2018-07-25-21-07-48.510617, Tool  1, -334.386139, -218.979599, -1727.212769, 0.339724, 0.362396, -0.033178, 0.867270, Tool 2, 0.000000, 0.000000, 0.000000,  0.000000, 0.000000, 0.000000, 0.000000
...
```
The following information is extracted from polaris recordings,
* polaris parameters:

## creating training dataframe
