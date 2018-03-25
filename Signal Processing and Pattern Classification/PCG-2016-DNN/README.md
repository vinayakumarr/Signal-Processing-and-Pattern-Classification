Download the physionet dataset 

```
>> wget http://physionet.org/physiobank/database/challenge/2016/training.zip
>> unzip training.zip
```

Build a feature vector from the raw data and train the CNN

python preprocess/train_model.py <path_to_physionet_data>
e.g.,
python preprocess/train_model.py training/ f


