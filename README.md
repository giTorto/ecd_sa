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

To run the model
```commandline
python modeling/bilstm_crf.py --train joint_model_data/train_set.json --val joint_model_data/partition_A.json joint_model_data/partition_B.json
```


To run the model with test files
```commandline
python modeling/bilstm_crf.py --train joint_model_data/train_set.json --val joint_model_data/partition_A.json joint_model_data/partition_B.json --test joint_model_data/partition_C.json joint_model_data/partition_D.json 
```


To connect to tensorboard
```commandline
tensorboard --logdir=lightning_logs/
ssh -L 6006:localhost:6006 giuliano@192.168.163.222
```