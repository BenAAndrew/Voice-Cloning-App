function selectVocoderType(){
    vocoder = document.getElementById("vocoder").value;
    if(vocoder == "waveglow"){
        document.getElementById("hifigan-fields").style.display = "none";
        document.getElementById("waveglow-fields").style.display = "block";
    } else {
        document.getElementById("hifigan-fields").style.display = "block";
        document.getElementById("waveglow-fields").style.display = "none";
    }
}
selectVocoderType();
