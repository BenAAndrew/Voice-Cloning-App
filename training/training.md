# Training
Once we have a dataset that has been preprocessed, we can begin training the voice.

## Arguments
- **metadata_path**: The path to your dataset metadata file (labels)
- **dataset_directory**: The path to your audio directory
- **output_directory**: The path to save checkpoints
- find_checkpoint (optional): Whether to automatically continue training from the latest checkpoint (if found)
- checkpoint_path: The path to a specific checkpoint to start training from
- epochs: Number of epochs to rub training for

## How to run
`python train.py -m dataset/metadata.csv -d dataset/wavs -o dataset/checkpoints `
