function selectDataset(){
    dataset = document.getElementById("dataset").value;

    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
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
            labelInfo();
            showCheckpoints(dataset);
        }
    };
    xmlhttp.open("GET", "/dataset-duration?dataset="+dataset, true);
    xmlhttp.send();
}
selectDataset();
