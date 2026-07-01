# IECIA

*IECIA: Identiffer information entropy and cross-example interpolationbased adversarial attack against code models*

## Requirement

```
torch == 1.8.0
dgl == 0.7.2
transformers == 3.3
```

This is the recommended environment. Other versions may be compatible.

## Preparing the Dataset

**Use the pre-processed datasets**

1. Use the already pre-processed datasets -- [OJ], [OJClone], and [Authorship] in this website.
 
2. Put the contents of OJ, OJClone and Authorship in the corresponding directories of `data`, `data_clone`, and `data_author` respectively.

3. The pre-processed datasets for GRU, LSTM, LSCNN, and CodeBERT are all included in the directories now.

**Pre-process the datasets**

1. Download the raw datasets, *i.e.*, `oj.tar.gz` 

2. Put `oj.tar.gz` in the directory of `data`.

3. Run the following commands to build the OJ dataset for the DL models. The dataset format of OJClone and Authorship is almost identical to OJ, and the code can be reused.

```sh
> cd preprocesslstm
> python3 main.py 
```

4. Copy `oj.pkl.gz`, `oj_uid.pkl.gz`, and `oj_inspos.pkl.gz` in the directory of `data` and paste them into `data_clone`.

5. Run the following commands to build the different dataset for the DL models.

```sh
> cd preprecess_lstm
> python3 main.py
> cd ../preprocess_author
> python3 main.py
> cd ../preprocess_clone
> python3 main.py
> cd ..
```

6. Training Models.

## Training the DL Model

All trained models used in this study are available at [Google Drive](https://drive.google.com/drive/folders/1Rt2Y980YJsev3fwfij_gi0w60boS4tCe?usp=sharing). Alternatively, you can train the models by following the steps below.

The source code directories are named according to the dataset and the model. `code`, `code_clone` and `code_author` refers to OJ, OJClone and Authorship, respectively.

The source code files to train each model (*i.e.*, GRU, LSTM, LSCNN and CodeBERT) on each dataset (*i.e.*, OJ, OJClone, and Authorship) are included in each corresponding directory. For instance, `code-OJ-codebert` refers to CodeBERT for OJ. 

*E.g.*, run the following commands to train a LSTM model on OJ.

```sh
> cd code-OJ-lstm
> python3 train.py -gpu 0 -model LSTM -lr 1e-3 -save_dir MODEL/SAVE/PATH --data ../data/oj.pkl.gz
> cd ..
```

Run the following commands to train a GRU model on OJCLONE.

```sh
> cd code-clone-gru
> python3 train.py -gpu 0 -model GRU -lr 1e-3 -save_dir MODEL/SAVE/PATH --data ../data_clone/oj.pkl.gz
> cd ..
```


## Attack

Run `python attacker.py` in each directory to attack the DL models.

*E.g.*, run the following commands to attack the CodeBERT model on OJ.

```sh
> cd code-OJ-codebert
> python3 attacker.py --model_dir ../CODEBERT/MODEL/PATH
> cd ..
```

One may use the attacking algorithm by employing different `Attacker`'s in the code.

## Adversarial Training

Take LSTM on OJClone for example.

1. Run the following commands to create the adversarial example training set.

```sh
> cd code-clone-lstm
> python attacker4training.py
> cd ..
```

2. Run the following commands to adversarially train the model.

```sh
> cd code-clone-lstm
> python train.py --adv_train_path ../ADVERSARIAL/EXAMPLE/SET --OTHER_ARGUMENTS
> ..
```

3. Go back to step 1 to iteratively update the adversarial example set upon the current training set.

Acknowledgement

This repository is built upon the publicly available code of CodeBERT, LSTM, GRU, LSCNN, CODA, and CARROT. We sincerely thank the authors for their valuable contributions.

