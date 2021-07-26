function selectModel(){
    model = document.getElementById("model").value;
    select = document.getElementById("checkpoint");
    // Delete options
    for (i = select.options.length-1; i >= 0; i--) {
        select.options[i] = null;
    }
    matching_checkpoints = checkpoints[model];
    // Add checkpoint options
    for (i = 0; i < matching_checkpoints.length; i++) {
        option = document.createElement("option");
        option.value = matching_checkpoints[i];
        option.text = matching_checkpoints[i];
        select.appendChild(option);
    }
}
selectModel();

function selectVocoder(){
    vocoder_type = document.getElementById("vocoder_type").value;
    select = document.getElementById("vocoder");
    // Delete options
    for (i = select.options.length-1; i >= 0; i--) {
        select.options[i] = null;
    }

    // Add options
    if(vocoder_type == "waveglow"){
        for (i = 0; i < waveglow_models.length; i++) {
            option = document.createElement("option");
            option.value = waveglow_models[i];
            option.text = waveglow_models[i];
            select.appendChild(option);
        }
    } else {
        for (i = 0; i < hifigan_models.length; i++) {
            option = document.createElement("option");
            option.value = hifigan_models[i];
            option.text = hifigan_models[i];
            select.appendChild(option);
        }
    }
}
selectVocoder();
