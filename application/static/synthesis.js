function createTextInput(value){
    var element = document.createElement("input");
    element.className = "form-control";
    element.type = "text";
    element.id = "text";
    element.name = "text";
    element.required = true;
    if(value)
        element.value = value;
    return element;
}

function addTextInput(value){
    var section = document.getElementById("multi-text");
    section.appendChild(document.createElement("br"));
    section.appendChild(createTextInput(value));
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
            link.onclick = function() { addTextInput(); };
            link.href = "#";
            link.appendChild(document.createTextNode("Add line"));
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

    // Populate existing data
    if(text){
        if(method == "single" || method == "paragraph"){
            document.getElementById("text").value = text;
        } else {
            if(Array.isArray(text)){
                document.getElementById("text").value = text[0];
                for(var i = 1; i < text.length; i++){
                    addTextInput(text[i]);
                }
            } else {
                document.getElementById("text").value = text;
            }
        }
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
