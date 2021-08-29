function createTextInput(){
    var element = document.createElement("input");
    element.className = "form-control";
    element.type = "text";
    element.id = "text";
    element.name = "text";
    element.required = true;
    return element;
}

function addTextInput(){
    var section = document.getElementById("multi-text");
    section.appendChild(document.createElement("br"));
    section.appendChild(createTextInput());
}

function changeTextMethod(){
    method = document.getElementById("text_method").value;
    section = document.getElementById("text_input");
    section.innerHTML = "";

    if(method == "single" || method == "multi"){
        var element = createTextInput();
        if(method == "single"){
            section.appendChild(element);
        } else {
            var container = document.createElement("span");
            container.id = "multi-text";
            section.appendChild(container);
            container.appendChild(createTextInput());

            var link = document.createElement("a");
            link.onclick = addTextInput;
            link.href = "#";
            link.appendChild(document.createTextNode("Hi"));
            section.appendChild(link);
        }
    } else {
        var element = document.createElement("textarea");
        element.className = "form-control";
        element.id = "text";
        element.name = "text";
        element.rows = "4";
        element.required = true;
        section.appendChild(element);
    }
}
changeTextMethod();

function showMaxDecoderStepsLabel(){
    document.getElementById("max_decoder_steps_label").innerHTML = document.getElementById("max_decoder_steps").value;
}
showMaxDecoderStepsLabel();

function showSilenceLabel(){
    document.getElementById("silence_label").innerHTML = document.getElementById("silence").value + " seconds";
}
showSilenceLabel();
