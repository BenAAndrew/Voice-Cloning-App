function showConfidenceLabel(){
    newVal = document.getElementById("confidence").value;
    text = newVal.toString();
    document.getElementById("confidence_label").innerHTML = newVal.toString();
}
showConfidenceLabel();
