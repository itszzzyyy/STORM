The dataset is available at https://hub.marinecadastre.gov/pages/vesseltraffic

Dataprocess：

1.python preprocess.py to initially process the dataset

2.python main.py to get the training dataset and validation dataset for model training

3.modify the arg-ais.ini and python main.py to get test data

Model training:

Modify the `arg.ini` file and run the `main.py` file for training

Modify the `arg.ini` file for testing.

You can obtain the `xxx_raw.npy` file to store the query trajectory and database trajectory, and the `meanrank.npy` file to store the test results.

Demo:

Run the `demo.py` file to execute the demo.



