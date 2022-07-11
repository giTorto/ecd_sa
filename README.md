# CRF
To run crf train and test
```commandline
python modeling/crf.py --train joint_model_data/train_set.json --test joint_model_data/partition_C.json joint_model_data/partition_D.json
```

To run crf train and val
```commandline
python modeling/crf.py --train joint_model_data/train_set.json --test joint_model_data/partition_A.json joint_model_data/partition_B.json
```

# BILSTM

To test Bilstm data loader
```commandline
python data_loading/torch_loader.py -d joint_model_data/train_set.json
```