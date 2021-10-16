function addSuggestion(value, low, medium, high){
    if(value < low){
        return " (Poor quality) 🙁";
    }else if(value < medium){
        return " (OK quality) 🙂";
    }else if(value < high){
        return " (Good quality) 😃";
    }else{
        return " (Excellent quality) 😄";
    }
}

let SECONDS_PER_ITERATION_PER_SECOND = 0.5;
let CHECKPOINT_SIZE_GB = 0.333;

// Label info
function labelInfo(){
    epochs = document.getElementById("epochs").value;
    total_clips = document.getElementById("total_clips").value;
    duration = document.getElementById("duration").value;
    batch_size = document.getElementById("batch_size").value;
    average_duration = parseInt(duration) / parseInt(total_clips);

    iters_per_epoch = Math.ceil(parseInt(total_clips) / parseInt(batch_size));
    iters = iters_per_epoch * parseInt(epochs);

    // Time estimate
    seconds = Math.ceil(iters * average_duration * SECONDS_PER_ITERATION_PER_SECOND);

    days = Math.floor(((seconds / 60) / 60) / 24);
    seconds -= ((days * 24) * 60) *60;

    hours = Math.floor((seconds / 60) / 60);
    seconds -= (hours * 60) * 60;

    minutes = Math.floor(seconds / 60);
    seconds -= minutes * 60;

    estimate = days+" days, "+hours+":"+minutes+":"+seconds;
    document.getElementById("time_estimate").innerHTML = estimate;

    // Disk usage
    backup_checkpoints = iters / parseInt(document.getElementById("backup_checkpoint_frequency").value);
    console.log(backup_checkpoints);
    disk_usage_gb = (backup_checkpoints + 1) * CHECKPOINT_SIZE_GB;
    document.getElementById("disk_usage").innerHTML = disk_usage_gb.toFixed(2) + "GB";
}

// Epochs
low_epoch_threshold = 2500;
medium_epoch_threshold = 5000;
high_epoch_threshold = 7500;

low_epoch_threshold_with_tl = 500;
medium_epoch_threshold_with_tl = 1500;
high_epoch_threshold_with_tl = 2500;

function showEpochsLabel(){
    newVal = document.getElementById("epochs").value;
    text = newVal.toString();
    if(document.getElementById("pretrained_model").files.length == 1){
        text += addSuggestion(newVal, low_epoch_threshold_with_tl, medium_epoch_threshold_with_tl, high_epoch_threshold_with_tl);
    } else {
        text += addSuggestion(newVal, low_epoch_threshold, medium_epoch_threshold, high_epoch_threshold);
    }
    document.getElementById("epochs_label").innerHTML = text;
    labelInfo();
}
document.getElementById("pretrained_model").addEventListener("change", showEpochsLabel, false);
showEpochsLabel();

// Checkpoints
function showCheckpoints(dataset){
    select = document.getElementById("checkpoint");
    // Delete options
    for (i = select.options.length-1; i >= 0; i--) {
        select.options[i] = null;
    }
    if(dataset in checkpoints){
        matching_checkpoints = checkpoints[dataset];
        // Add checkpoint options
        for (i = 0; i < matching_checkpoints.length; i++) {
            option = document.createElement("option");
            option.value = matching_checkpoints[i];
            option.text = matching_checkpoints[i];
            select.appendChild(option);
        }
        document.getElementById("checkpoint_field").style.display = "block";
        document.getElementById("pretrained_model_field").style.display = "none";
    } else {
        document.getElementById("checkpoint_field").style.display = "none";
        document.getElementById("pretrained_model_field").style.display = "block";
    }
}

// Dataset
low_dataset_threshold = 60 * 60;
medium_dataset_threshold = 180 * 60;
high_dataset_threshold = 300 * 60;

function showDatasetInfo(){
    datasetpath = document.getElementById("dataset").value;
    
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            j = JSON.parse(this.response);
            if(j.error){
                document.getElementById("duration").value = 0;
                document.getElementById("total_clips").value = 0;
                document.getElementById("dataset_label").innerHTML = "WARNING: "+j.error;
            } else {
                duration = j.total_duration;
                document.getElementById("duration").value = duration;
                total_clips = j.total_clips;
                document.getElementById("total_clips").value = total_clips;
                hours = Math.floor(duration / 60 / 60);
                minutes = Math.floor(duration / 60) - (hours * 60);
                text = hours+" hours, "+minutes+" minutes";
                text += addSuggestion(duration, low_dataset_threshold, medium_dataset_threshold, high_dataset_threshold);
                document.getElementById("dataset_label").innerHTML = text;
            }
            labelInfo();
            showCheckpoints(datasetpath);
        }
    };
    xmlhttp.open("GET", "/dataset-duration?dataset="+datasetpath, true);
    xmlhttp.send();
}
showDatasetInfo();

// Batch size
function showBatchSize(){
    newVal = document.getElementById("batch_size").value;
    document.getElementById("batch_size_label").value = newVal;
    labelInfo();
}
showBatchSize();

function editBatchSize(){
    document.getElementById("batch_size").value = document.getElementById("batch_size_label").value;
    showBatchSize();
}

// Checkpoint frequency
function showCheckpointFrequencyLabel(){
    document.getElementById("checkpoint_frequency_label").innerHTML =  document.getElementById("checkpoint_frequency").value;
}
showCheckpointFrequencyLabel();

function showCheckpointBackupFrequencyLabel(){
    document.getElementById("backup_checkpoint_frequency_label").innerHTML =  document.getElementById("backup_checkpoint_frequency").value;
    labelInfo();
}
showCheckpointBackupFrequencyLabel();

// Validation size
function showValidationSize(){
    document.getElementById("validation_size_label").innerHTML = document.getElementById("validation_size").value * 100 + "%";
}
showValidationSize();
