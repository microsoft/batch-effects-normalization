# Batch Effects Normalization (BEN)

This respository contains the code accompanying the paper "[Incorporating knowledge of plates in batch normalization improves generalization of deep learning for microscopy images](https://www.biorxiv.org/content/10.1101/2022.10.14.512286v1)" by [Alex Lin](https://sites.google.com/view/alexanderlin) and [Alex Lu](https://www.alexluresearch.com/).  Please find licensing details in the file `license.txt`.

### Overview
- Our code is built on the [PyTorch](https://pytorch.org/) library.
- The majority of the code is packaged into modules in the `biomass` directory.  Here, you will find specific implementations of models, data loaders, data augmentation transforms, etc.  To simply run the code, you do not need to directly edit anything in this directory.
- The Python scripts in the root directory (e.g. `train_erm.py`, `train_simclr.py`, etc.) are the outer functions that run the experiments for the paper.  To run a particular experiment, execute `python [script_name].py` (e.g. `python train_erm.py`) in the command line.  
- Each script has an associated YAML file in the `configs` directory, so you do not need to edit the scripts themselves to vary experimental settings.  Within each YAML config, there is a dictionary of hyperparameter configurations for each experiment.  You can vary these to re-run our experiments with different settings.  Our codebase uses [Hydra](https://hydra.cc/) for experiment management; for more information on how Python scripts and YAML files interact, please consult the Hydra documentation.
- When you run a script, our codebase will create three things:
    - In the `outputs` directory, it will create a directory named by the timestamp in which you ran the script.  The YAML config file associated with the run will be copied and dumped in this directory so you can refer back to what hyperparameters you used at that time.
    - In the `runs` directory, it will dump a [TensorBoard](https://www.tensorflow.org/tensorboard) charting the progress of model training.  Within the tensorboard, we record metrics such as accuracy, loss, gradient dynamics, etc.
    - In the `checkpoints` directory, it will dump a checkpoint of the PyTorch model associated with the experiment.
- The `requirements.txt` file contains all of the Python packages and associated versions that we used.  You can run `pip install -r requirements.txt` in the command line to automatically install the correct versions for each package.
- The `misc` directory contains miscellaneous files needed for running certain experiments.

### Supervised Learning Experiments
- To reproduce our supervised learning experiments, use the script `train_erm.py`.  (Note that ERM stands for empirical risk minimization, i.e. another term for supervised learning.)
- The current associated config file `configs/train_erm.yaml` is set up to run supervised learning with BEN, our batch effects correction method.
- To run vanilla supervised learning (i.e. without BEN), make the following edits to `configs/train_erm.yaml`: delete `train_groupby`, delete `max_groups`, set `eval_plate_sampler: False`, set `eval_batch_size: 75` (or whatever batch size you prefer), and set `use_train_at_eval: False`.
- The first time you run `train_erm.py`, note that our script should automatically download the RxRx1-Wilds dataset for you from the [Wilds package](https://github.com/p-lambda/wilds).       

### Self-Supervised Learning Experiments
- To reproduce our self-supervised learning experiments, use the script `train_simclr.py` for training the base model and `train_classifier.py` for fitting the linear classifier on the learned representations.
- Thus, in the `train_classifier.yaml` file, there is an argument `model_path: xxxxx` that needs to point to a saved checkpoint obtained from running `train_simclr.py`.  Make sure to manually set the correct path for `train_classifier.yaml` after running `train_simclr.py`.
- In both `train_simclr.yaml` and `train_classifier.yaml`, there is an argument `img_dir` that needs to point to a directory of cropped cells for RxRx1-Wilds.  A zipped file of this directory can be downloaded at this link (note it is about ~4 GB in size): https://zenodo.org/record/7272553#.Y2KkNuzMJTZ  
- The current config file `train_simclr.yaml` is setup to train the vanilla SimCLR algorithm (without BEN).  To run SimCLR + BEN, simply change the argument `sampler: random` to `sampler: plate`.  Then, to apply BEN while training the classifier, go to `train_classifier.yaml` and change the arguments `sampler: random` -> `sampler: plate` and `model_train_mode: False` -> `model_train_mode: True`. 
- To use MinCLR (i.e. multiple instance constrastive learning, a new method that we developed) instead of SimCLR, simply go to `train_simclr.yaml` and change `mode: random_single` to `mode: random` (this will define positive anchors as random cells from the same image instead of random augmentations of the same single cell).  To run MinCLR + BEN, follow the aforementioned instructions for SimCLR + BEN. 
- To increase the number of positive anchors during training, go to `train_simclr.yaml` and change `num_img: 2` to any other value (e.g. `num_img: 5`).  Make sure that this value also matches the argument `num_views: 2` (e.g. `num_views: 5`).  
- To train representations from a cell-level supervised classifier (instead of a self-supervised learner), use the script `train_supervised_cell.py`.  The current YAML file `train_supervised_cell.yaml` is designed for standard supervised learning (without BEN).  To use BEN, simply change `sampler: random` to `sampler: plate`.

### Transfer Learning Experiments
- To reproduce our transfer learning experiments, use the notebook `pybbbc.ipynb`.  
- To obtain the dataset for BBBC021, use the following package: https://github.com/giacomodeodato/pybbbc and follow the instructions for "Data download" and "Dataset creation".  Note that this can take several hours.  Afterwards, the data should be dumped by default into a directory called `~/.cache/`.
- You also need to obtain two files that list the nuclei centers for cells in this dataset.  You can download these files (called `supplement_Object.txt` and `supplement_Image.txt`) from the supplementary files of this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3884769/ (under the title "Data S2").
- Since this is transfer learning, you should also have a link to the checkpoint of a pre-trained model on the RxRx1-Wilds (cell level) dataset, obtained by running a self-supervised learning experiment (for example).  This checkpoint will need to be loaded into the state dict of the PyTorch model (see notebook).
- Towards the end of the notebook, we calculate both NSC and NSCB accuracy (see the paper for more details on these metrics).    