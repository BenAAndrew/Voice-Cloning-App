// add warning
function warn(e){
    var confirmationMessage = 'If you leave before saving, your changes will be lost.';
    (e || window.event).returnValue = confirmationMessage;
    return confirmationMessage;
}
window.addEventListener("beforeunload", warn);

$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/voice');
    console.log("connected!");

    socket.on('progress', function(msg) {
        console.log(msg);
        percentage = msg.number/msg.total;
        $('#progress').val(percentage*100);
        $('#progress_label').text(msg.number + "/" + msg.total);
    });

    socket.on('done', function(msg) {
        window.removeEventListener("beforeunload", warn);
        document.getElementById('next_link').style.visibility='visible';
    });

    socket.on('status', function(msg) {
        $('#pinned').text(msg.text);
    });

    socket.on('alignment', function(msg) {
        $('#alignment-heading').text("Latest sample - Iteration " + msg.iteration)
        $('#alignment-img').attr("src", msg.image);
        $('#alignment').show();
    });

    socket.on('logs', function(msg) {
        console.log(msg);
        $('#logs').append(msg.text+"<br>");
    });

    socket.on('error', function(msg) {
        console.log(msg);
        var r = confirm(msg.text);
        if (r == true) {
            window.history.back();
        }
    });
});
