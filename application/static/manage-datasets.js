
function selectUnlabelledClip(){
    dataset = document.getElementById("dataset").value;
    clip = document.getElementById("unlabelled_clip").value;
    source = document.getElementById("audioSource");
    source.src = 'data/datasets/'+dataset+'/unlabelled/'+clip;
    audio = document.getElementById("audio");
    audio.load();
}

function selectDataset(){
    dataset = document.getElementById("dataset").value;

    var dataset_duration = new XMLHttpRequest();
    dataset_duration.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            j = JSON.parse(this.response);
            if(j.error){
                document.getElementById("error").innerHTML = "ERROR: "+j.error;
                document.getElementById("dataset-info").style.display = "none";
            } else {
                duration = j.total_duration;
                hours = Math.floor(duration / 60 / 60);
                minutes = Math.floor(duration / 60) - (hours * 60);
                document.getElementById("size").innerHTML = hours+" hours, "+minutes+" minutes";
                document.getElementById("clips").innerHTML = j.total_clips;
                document.getElementById("shortest-clip").innerHTML = j.min_clip_duration.toFixed(2);
                document.getElementById("mean-duration").innerHTML = j.mean_clip_duration.toFixed(2);
                document.getElementById("longest-clip").innerHTML = j.max_clip_duration.toFixed(2);
                document.getElementById("words").innerHTML = j.total_words;
                document.getElementById("mean-words").innerHTML = j.mean_words_per_clip.toFixed(2);
                document.getElementById("distinct-words").innerHTML = j.total_distinct_words;

                document.getElementById("error").innerHTML = "";
                document.getElementById("dataset-info").style.display = "block";
            }
        }
    };
    dataset_duration.open("GET", "/dataset-duration?dataset="+dataset, true);
    dataset_duration.send();

    select = document.getElementById("unlabelled_clip");
    var unlabelled_clips = new XMLHttpRequest();
    unlabelled_clips.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            j = JSON.parse(this.response);
            var clips = j.unlabelled;
            if(clips.length > 0) {
                document.getElementById("label-clip-message").innerHTML = "Label clips that the dataset generator couldn't";
                document.getElementById("label-clip").style.display = "block";
                for(let i = 0; i < clips.length; i++){
                    option = document.createElement("option");
                    option.value = clips[i];
                    option.text = clips[i];
                    select.appendChild(option);
                }
                selectUnlabelledClip();
            } else {
                document.getElementById("label-clip-message").innerHTML = "No clips to label";
                document.getElementById("label-clip").style.display = "none";
            }
        }
    };
    unlabelled_clips.open("GET", "/unlabelled-clips?dataset="+dataset, true);
    unlabelled_clips.send();
}
selectDataset();
