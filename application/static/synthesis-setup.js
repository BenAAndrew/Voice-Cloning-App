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
