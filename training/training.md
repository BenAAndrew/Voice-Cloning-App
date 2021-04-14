# Training
Once we have a dataset that has been preprocessed, we can begin training the voice.

## Arguments
- **metadata_path**: The path to your dataset metadata file (labels)
- **dataset_directory**: The path to your audio directory
- **output_directory**: The path to save checkpoints
- find_checkpoint (optional): Whether to automatically continue training from the latest checkpoint (if found)
- checkpoint_path (optional): The path to a specific checkpoint to start training from
- transfer_learning_path (optional): The path to an existing model to transfer learn from
- overwrite_checkpoints (optional): Whether to delete old checkpoints (default is true)
- epochs: Number of epochs to rub training for
- batch_size (optional): Batch size/ memory usage. Calculated automatically if not given

## How to run
`python train.py -m dataset/metadata.csv -d dataset/wavs -o dataset/checkpoints `
