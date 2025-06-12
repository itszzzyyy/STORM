# STORM

## Dataset

The dataset is available at https://hub.marinecadastre.gov/pages/vesseltraffic

## Data preprocess

All related files are in the `datapreprocess` folder.

```
cd datapreprocess
python preprocess.py
python main.py  --config arg.ini
```

You can obtain the `1_raw_q_demo.npy and 1_raw_db_demo.npy` file to store the query trajectory and database trajectory

## Training

```
cd model
python main.py
```

## Evaluation

```
python main.py
```

You can obtain the `mean_rank.npy` file to store the test results.

## Demo

```
python demo.py
```

You can access the main demo page like this:

![image-20250612143814655](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250612143814655.png)

**Click the "Activate Similar Trajectory Search" button** to enable trajectory similarity retrieval functionality. This feature allows you to search for trajectories similar to **a user-specified query trajectory**. **When finished, click the "Deactivate Similar Trajectory Search" button** to terminate the subprocess.![image-20250612144120959](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250612144120959.png)

![image-20250612144133661](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250612144133661.png)

You can **enable and disable** the *Automatic Similar Trajectory Cluster Detection* function **using the same operation**.

![image-20250612145142108](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20250612145142108.png)
