import os
from bs4 import BeautifulSoup
import requests

AREAS = ["woodlands", "sle", "tpe", "kje", "bke", "cte",
         "pie", "kpe", "aye", "mce", "ecp", "stg"]


def extract_image(area, main_dir="/home/xinyuan/images/", verbose=True):
    """
    Extracts images belonging to a given area with BeautifulSoup and stores them in folders.
    :param area: string, specifies the area name.
    :param main_dir: string, the main folder that contains sub-folders of images.
    :param verbose: boolean, specifies whether to display detailed logging.
    :return: None.
    """
    if verbose:
        print("Processing image extraction for the area", area)
    parent_url = "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/"
    page_url = parent_url + area + ".html"
    # Specify the user-agent, otherwise, may encounter Page Not Found
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"}
    html_page = requests.get(page_url, headers=headers)
    # Load full contents of the html
    soup = BeautifulSoup(html_page.content, "html.parser")
    # Extract the div that contains all the images
    div_images = soup.find("div", class_="road-snapshots")
    # Find all the images in the div
    images = div_images.findAll("img")
    # Create a new folder to save images if not exist
    dir = main_dir + area + "/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for image in images:
        img_url = "https:" + image["src"]  # extract the source of each image
        img_name = image["alt"]
        try:
            img = requests.get(img_url)
        except:
            continue
        f = open(dir + img_name.replace("/", "or") + ".jpg", "wb")
        f.write(img.content)
        f.close()
    if verbose:
        print("Images saved to the folder", area)
        print()


if __name__ == "__main__":
    for area in AREAS:
        extract_image(area, verbose=False)
