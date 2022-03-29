# coding=utf-8
"""Base driver to scrape data from Wikimedia websites."""

import re
import json
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


APPLICATION_HEADERS = {
    'User-Agent': 'destinations_similarity/0.1'
}


class WikiExtractor(object):
    """Class for extracting content from Wikimedia."""

    def __init__(self, wiki: str, lang: str):
        """Initialize driver."""
        self.wiki = wiki
        self.lang = lang
        self.rest_url = f"https://{lang}.{wiki}.org/api/rest_v1"

        # Create Session object for faster retrieval
        self.session = requests.Session()
        self.session.headers.update(APPLICATION_HEADERS)

    @classmethod
    def clean_content(cls, text: str, tags: List[str] = None) -> List[str]:
        """Remove HTML tags and citations from text.

        Args:
            text (str): The text to be cleaned.
            tags (List[str], optional): List of tags to be extracted.
                Defaults to ['p', 'li'].

        Returns:
            List[str]: A list with each piece of text extracted from the
                specified tags.
        """
        tags = tags or ['p', 'li']
        soup = BeautifulSoup(text, "html.parser")
        return [
            re.sub(r'\[.*?\]|<.*?>', '', str(x)).strip()
            for x in soup.find_all(tags)
        ]

    def extract_images(self, page: str) -> List[str]:
        """Retrieve images (as links) for a specified page.

        Args:
            page (str): The name of the page.

        Returns:
            List[str]: A list with the URLs of the images.
        """
        request = self.session.get(f"{self.rest_url}/page/media-list/{page}")
        response = json.loads(request.text)
        items = response.get('items', [])

        images_links = []

        for item in items:
            if item['type'] == 'image' and 'srcset' in item:
                images_links += [f"https:{item['srcset'][0]['src']}"]

        return images_links

    def extract_content_raw(
        self, page: str, summary: bool, sections: List[str] = None
    ) -> Dict[str, str]:
        """Retrieve the HTML-formatted sections from a page.

        Args:
            page (str): The name of the page.
            summary (bool): Boolean that specifies if the summary for the page
                must be retrieved.
            sections (List[str], optional): A list of sections to be retrieved.
                Defaults to None.

        Returns:
            Dict[str, str]: A dictionary with the sections, where each key is
                the section name.
        """
        sections = sections or []

        request = self.session.get(
            f"{self.rest_url}/page/mobile-sections/{page}")
        response = json.loads(request.text)

        sections_data = {}

        # Retrieve summary
        if summary and 'lead' in response:
            sections_data['summary'] = response['lead']['sections'][0]['text']

        # Retrieve sections and subsections (with HTML tags)
        if sections and 'remaining' in response:
            page_sections = response['remaining']['sections']

            # Get index of sections found
            idx_sections_found = [
                i for i, section in enumerate(page_sections)
                if section.get('line') in sections
            ]

            # Get level of each section, to identify subsections
            levels = [section.get('toclevel', -2) for section in page_sections]

            for start in idx_sections_found:
                try:
                    # Get index next section at same toclevel
                    end = next(
                        i + (start + 1)
                        for i, level in enumerate(levels[start + 1:])
                        if level <= levels[start]
                    )
                except StopIteration:   # End of page reached
                    end = len(page_sections)

                # Update dictionary
                sections_data[page_sections[start]['line']] = '\n'.join([
                    subsection.get('text', '')
                    for subsection in page_sections[start:end]
                ])

        return sections_data

    def extract_content(
        self, page: str, summary: bool, sections: List[str] = None,
        sections_tags: Dict[str, List[str]] = None,
        section_types: Dict[str, str] = None
    ) -> Dict[str, str]:
        """Retrieve formatted (clean) text from Wikipedia."""
        sections = sections or []
        results = self.extract_content_raw(page, summary, sections)

        # Get tags to keep and types to convert
        sections_tags = {
            section: sections_tags.get(section, ['p', 'li'])
            for section in ['summary'] + sections
        }
        section_types = {
            section: section_types.get(section, 'str')
            for section in ['summary'] + sections
        }

        # Clean the sections
        for section in results:
            if section_types[section] == 'list':
                results[section] = self.clean_content(
                    results[section], tags=sections_tags[section])
            elif section_types[section] == 'str':
                results[section] = '\n'.join(self.clean_content(
                    results[section], tags=sections_tags[section]))
            else:
                raise NotImplementedError(
                    f"No implementation for this type of output: "
                    f"{section_types[section]}"
                )

        return results
