# coding=utf-8
"""
--
"""
from typing import Dict, Union, Any
from requests import get
from bs4 import BeautifulSoup
from lxml import etree


def generate_wikivoyage_url(citie_name: str) -> str:
    """
    Args:
    Returns:
    """
    return f"https://en.wikivoyage.org/wiki/{citie_name}"


def format_image_url(url: str) -> str:
    if not url.startswith("http"):
        return "https:" + url


def get_to_do(elements):
    finded = False
    paragraphs = []
    for element in elements:
        if not finded:
            if element.name == "h2":
                if element.find("span") and element.find("span").get("id") == "Do":
                    finded = True
        else:
            if element.name == "p":
                paragraphs.append(element)
            if element.name == "h2":
                break
    return paragraphs


def get_go_next(elements):
    finded = False
    candidates = []
    cities = []
    for element in elements:
        if not finded:
            if element.name == "h2":
                if element.find("span") and element.find("span").get("id") == "Go_next":
                    finded = True
        else:
            if element.name == "ul":
                candidates.append(element.findAll("li"))
            if element.name == "noscript":
                break

    for element in candidates:
        for candidate in element:
            for city in candidate.findAll("a"):
                cities.append(city.get("title"))
    return cities


class WikivoyagerExtractor:

    def __init__(self):
        pass

    @staticmethod
    def _get_html(citie) -> BeautifulSoup:
        request = get(url=generate_wikivoyage_url(citie))
        if request.status_code not in [200, 404]:
            raise ValueError("Unexpected status code returned from url '%s'. Status returned: %s" %
                             (generate_wikivoyage_url(citie), request.status_code))
        else:
            if request.status_code == 404:
                return None
            else:
                return BeautifulSoup(request.text, "html.parser")

    def extract_by_name(self, citie_name: str, seach_name: str):
        html_page = self._get_html(seach_name)
        if not html_page:
            return None
        try:
            if html_page.findAll("p")[0].text.lower().find("there is currently no text in this page"):
                return None
            else:
                description = ""
                for element in html_page.findAll(attrs="mw-parser-output")[0].findChildren("p", limit=3):
                    if element.text:
                        if element.text.find("citie_name"):
                            description += element.text + "\n"

                citie_images = []
                for element in html_page.findAll(attrs="mw-parser-output")[0].findAll(attrs="thumbinner"):
                    images = element.findAll("img")
                    citie_images.append([format_image_url(img.get("src")) for img in images])

                citie_to_do = ""
                for element in get_to_do(html_page.findAll(attrs="mw-parser-output")[0].findAllNext(recursive=False)):
                    citie_to_do += element.text + "\n"

                cities_next = get_go_next(html_page.findAll(attrs="mw-parser-output")[0].findAllNext(recursive=False))

            return {"name": citie_name, "description": description, "images": citie_images, "do": citie_to_do,
                    "next": cities_next}
        except Exception as error:
            raise RuntimeError("Unexpected error occurred in extracting %s information" % citie_name) from error
