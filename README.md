# Details on Directory Structure

- The executable scripts live in the root directory.  You will see names like "train_erm.py", "train_simclr.py" etc. -- these refer to the different experiments you can run. 
- Each executable Python file (e.g. "train_erm.py") has a corresponding config YAML file in the configs directory (e.g. "configs/train_erm.yaml").  Using the config, you can edit the hyperparameters for each script.  For example, if you want to change the batch size during training, you can just change "train_batch_size: 75" in the current YAML file to whatever you want.  (Note: 75 is what I used for all supervised learning experiments.  For self-supervised learning experiments, I am already using the full plate during training.  You may also notice "eval_batch_size: ~" in the file.  This is because for evaluation, no batch size needs to be entered due to the earlier argument "eval_plate_sampler: True" forcing the whole plate to be used during evaluation.)    
- "biomass" directory: This is where the main parts of the code lives.  If you would like to look into the implementation details behind the models, the datasets, the dataloaders, etc., you can find them here.
- "outputs" directory: I'm using an experiment manager called Hydra.  Whenever you run a script, it will dump the current state of the corresponding YAML into a time-stamped folder in this directory.  This way, you can go back and look at what hyperparameters you used for that run in case you change the YAML later. 
- "runs" directory: For each run, a TensorBoard file will be created in this directory based on the timestamp (usually the same as the timestamp used in the "outputs" directory, though it may be ~1 second off for certain runs).  You can use this TensorBoard file to monitor results as the model trains (I record metrics such as loss, accuracy, etc.)
- "checkpoints" directory: If you have "save_model: True" in a YAML file, a corresponding checkpoint of the PyTorch model will be dumped in this directory.  The checkpoint is again named by the timestamp of the run.  
