document.addEventListener('DOMContentLoaded', function() {
    var convert = document.getElementById('scrapeBook');
    var download = document.getElementById('exportBook');

    convert.addEventListener('click', function() {
        function getBook(){
            function scrapePage(){
                function hashString(str){
                    let hash = 0;
                    for (let i = 0; i < str.length; i++) {
                        hash += Math.pow(str.charCodeAt(i) * 31, str.length - i);
                        hash = hash & hash; // Convert to 32bit integer
                    }
                    return hash;
                }
    
                var hashes = {};
                var content = [];
                function addDiv(div){
                    let hash  = hashString(div.innerText);
                    if (hashes[hash] === undefined) {
                        hashes[hash] = true;
                        content.push(div.outerHTML);
                    }
                }
    
                var appFrame = document.querySelector('#KindleReaderIFrame').contentDocument;
                var contentFrames = Array.from(appFrame.querySelectorAll('iframe')).map(f => f.contentDocument);
                Array.from(contentFrames[1].querySelectorAll('body > div')).forEach(addDiv);
                appFrame.getElementById('kindleReader_pageTurnAreaRight').click();
                return content;
            }
    
            function extractContent(s) {
                var span = document.createElement('span');
                span.innerHTML = s;
                return span.textContent || span.innerText;
            }

            function getFooter(){
                var appFrame = document.querySelector('#KindleReaderIFrame').contentDocument;
                return appFrame.getElementById("kindleReader_footer_message").innerHTML.split(" ");
            }

            function getCurrentPageNumber(){
                var footer = getFooter();
                return parseInt(footer[3]);
            }

            var footer = getFooter();
            var num_pages = parseInt(footer[5]);
            var current_page = parseInt(footer[3]);
            var book = "";
            var last_page = "";
			
            function getPageText(){
                text = "\n\n";
                page = scrapePage();
                for (let i = 0; i < page.length; i++){
                    text += extractContent(page[i]) + "\n";
                }
                if(text != last_page){
                    book += text;
                    last_page = text;
                }
                current_page = getCurrentPageNumber();
                if(current_page == num_pages){ 
                    alert("Extracted "+num_pages+" pages");
                    var data = document.getElementById("data");
                    if(data){
                        data.innerHTML = book;
                    } else {
                        document.body.innerHTML += "<span id='data'>"+book+"</span>";
                    }
					return;
                }
				setTimeout(getPageText, 1000);
			}

            getPageText();
            return text;
        }

        chrome.tabs.executeScript({
            code: '(' + getBook + ')();'
        }, (result) => {
            console.log(result);
        });
    }, false);

    download.addEventListener('click', function() {
        chrome.tabs.executeScript({
            code: '(function getData(){ return document.getElementById("data").innerHTML; })();'
        }, (result) => {
            console.log(result);
            var blob = new Blob([result], {type: "text/plain"});
            var url = URL.createObjectURL(blob);
            chrome.downloads.download({
                url: url,
                filename: "book.txt"
            });
        });        
    }, false);
}, false);
