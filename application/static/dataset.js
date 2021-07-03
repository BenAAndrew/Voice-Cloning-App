function showConfidenceLabel(){
    newVal = document.getElementById("confidence").value;
    text = newVal.toString();
    document.getElementById("confidence_label").innerHTML = newVal.toString();
}
showConfidenceLabel();

function selectLanguage(){
    language = document.getElementById("language").value;
    confidence = document.getElementById("confidence");
    if(language == "English"){
        confidence.value = 0.85;
    } else {
        confidence.value = 0.7;
    }
    showConfidenceLabel();
}
