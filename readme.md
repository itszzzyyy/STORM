# STORM

## Dataset

The dataset is available at https://hub.marinecadastre.gov/pages/vesseltraffic

## Data preprocess

All related files are in the `datapreprocess` folder.

```
cd datapreprocess
python preprocess.py
python main.py  --config arg-ais.ini
```

You can obtain the `xxx_raw.npy` file to store the query trajectory and database trajectory

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
