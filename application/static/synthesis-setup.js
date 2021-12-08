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
    
    // Add custom hifigan model to vocoder list
    vocoder_list = document.getElementById("vocoder");
    if(hifigan_custom_models[model]){
        custom_model = hifigan_custom_models[model][0];
        option = document.createElement("option");
        option.value = "custom-"+custom_model;
        option.text = "Custom: "+custom_model;
        vocoder_list.insertBefore(option, vocoder_list.firstChild);
        vocoder_list.value = "custom-"+custom_model;
    } else {
        if(vocoder_list.firstChild.value.startsWith("custom-")){
            vocoder_list.options[0] = null;
        }
    }
}
selectModel();
