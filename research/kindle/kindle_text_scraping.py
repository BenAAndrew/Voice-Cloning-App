import argparse
import time

from selenium import webdriver
from html2text import html2text

if __name__ == "__main__":
    """Script to extract an Kindle book to text file."""
    parser = argparse.ArgumentParser(description="Extract text from Kindle audio books")
    parser.add_argument(
        "-b",
        "--book",
        help="Book URL i.e. https://read.amazon.co.uk/?asin=12345678",
        type=str,
        default="https://read.amazon.co.uk/",
    )
    parser.add_argument("-u", "--username", help="Kindle username", type=str, required=True)
    parser.add_argument("-p", "--password", help="Kindle password", type=str, required=True)
    parser.add_argument("-o", "--output", help="Output text file path", default="kindle.txt", type=str)
    parser.add_argument(
        "-d", "--delay", help="Delay at start after login to select book/ skip screens", default=15, type=int
    )
    args = parser.parse_args()

    driver = webdriver.Chrome()
    driver.get(args.book)
    time.sleep(1)
    driver.find_element_by_name("email").send_keys(args.username)
    driver.find_element_by_name("password").send_keys(args.password)
    driver.find_element_by_id("signInSubmit").click()
    time.sleep(args.delay)

    book = []

    try:
        while True:
            page = driver.execute_script(
                """
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

                var timeout = null;
                var appFrame = document.querySelector('#KindleReaderIFrame').contentDocument;
                var contentFrames = Array.from(appFrame.querySelectorAll('iframe')).map(f => f.contentDocument);
                Array.from(contentFrames[1].querySelectorAll('body > div')).forEach(addDiv);
                appFrame.getElementById('kindleReader_pageTurnAreaRight').click();
                return content;
                """
            )
            text = "\n".join([html2text(item).strip() for item in page])
            book.append(text)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

with open(args.output, "w") as f:
    for line in book:
        f.write(line)
        f.write("\n")
