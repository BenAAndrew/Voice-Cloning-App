function showMaxDecoderStepsLabel(){
    document.getElementById("max_decoder_steps_label").innerHTML = document.getElementById("max_decoder_steps").value;
}
showMaxDecoderStepsLabel();

function showSilenceLabel(){
    document.getElementById("silence_label").innerHTML = document.getElementById("silence").value + " seconds";
}
showSilenceLabel();
