# Training
Once we have a dataset that has been preprocessed, we can begin training the voice.

## Arguments
- **metadata_path**: The path to your dataset metadata file (labels)
- **dataset_directory**: The path to your audio directory
- **output_directory**: The path to save checkpoints
- alphabet_path (optional): The path to the alphabet file (if using a non-English dataset)
- checkpoint_path (optional): The path to a specific checkpoint to start training from
- transfer_learning_path (optional): The path to an existing model to transfer learn from
- epochs: Number of epochs to rub training for
- batch_size (optional): Batch size/ memory usage. Calculated automatically if not given

## How to run

1. Download NVIDIA's [Tacotron 2](https://github.com/NVIDIA/tacotron2#training-using-a-pre-trained-model) model
   
2. Move into `tacotron2_statedict.pt` into `training` folder

3. Run `python train.py -m dataset/metadata.csv -d dataset/wavs -o dataset/checkpoints -t tacotron2_statedict.pt`