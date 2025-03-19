import json
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class SoictProfileCrawler:
    def __init__(self, base_url="https://soict.hust.edu.vn/"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_page(self, url):
        """Fetch a webpage and return its content"""
        try:
            response = requests.get(url, headers=self.headers)
            response.encoding = "utf-8"  # Ensure proper encoding for Vietnamese
            if response.status_code == 200:
                return response.text
            else:
                print(
                    f"Failed to retrieve page {url}. Status code: {response.status_code}"
                )
                return None
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def extract_text_from_section(self, soup, section_title, next_tags=None):
        """Extract text from a section with the given title, trying multiple possible next tags"""
        # Default tags to try if none specified
        if next_tags is None:
            next_tags = ["ul", "ol"]
        # Convert to list if a single tag was provided
        elif isinstance(next_tags, str):
            next_tags = [next_tags]

        # Convert section_title to list if it's a string
        if isinstance(section_title, str):
            section_titles = [section_title]
        else:
            section_titles = section_title

        section_element = None
        found_title = None

        # Try to find the section heading
        for element in soup.find_all(["h3", "h2", "h4"]):
            for title in section_titles:
                if title.lower() in element.text.lower():
                    section_element = element.parent
                    found_title = title
                    break
            if section_element:
                break

        if not section_element:
            # Try to find it by looking at div containers
            for div in soup.find_all("div", class_="section-title-container"):
                for title in section_titles:
                    if div.find(string=re.compile(title, re.IGNORECASE)):
                        section_element = div
                        found_title = title
                        break
                if section_element:
                    break

        if section_element:
            # Try each of the possible next tags
            for next_tag in next_tags:
                content_element = section_element.find_next(next_tag)
                if content_element:
                    # Make sure we're not grabbing content from a later section
                    next_section = content_element.find_next(
                        "div", class_="section-title-container"
                    )
                    if (
                        next_section
                        and section_element.find_next(
                            "div", class_="section-title-container"
                        )
                        != next_section
                    ):
                        continue

                    if next_tag in ["ul", "ol"]:
                        return [
                            li.text.strip() for li in content_element.find_all("li")
                        ]
                    else:
                        return content_element.text.strip()

        # If we get here, we didn't find anything
        return [] if any(tag in ["ul", "ol"] for tag in next_tags) else ""

    def extract_paragraphs_until_next_section(self, soup, start_element):
        """Extract all paragraphs after start_element until the next section heading"""
        paragraphs = []
        for element in start_element.find_next_siblings():
            if (
                element.name in ["div"]
                and element.get("class")
                and "section-title-container" in element.get("class")
            ):
                break
            if element.name == "p":
                text = element.text.strip()
                if text:
                    paragraphs.append(text)
        return " ".join(paragraphs)

    def extract_email(self, soup):
        """Extract email addresses from the profile page"""
        emails = []

        # Look for email in paragraphs containing "email" or with mailto links
        for p in soup.find_all("p"):
            text = p.text.lower()
            if "email" in text or "@" in text:
                # Find email using regex
                email_matches = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
                emails.extend(email_matches)

                # Also look for mailto links
                for a in p.find_all("a", href=True):
                    if a["href"].startswith("mailto:"):
                        email = a["href"].replace("mailto:", "").strip()
                        if email not in emails and "vp" not in email:
                            emails.append(email)

        # If still not found, look for any mailto links in the page
        if not emails:
            for a in soup.find_all("a", href=True):
                if a["href"].startswith("mailto:"):
                    email = a["href"].replace("mailto:", "").strip()
                    if email not in emails:
                        emails.append(email)

        return emails

    def parse_profile(self, html_content):
        """Parse a profile page and extract relevant information"""
        if not html_content:
            return {}

        soup = BeautifulSoup(html_content, "html.parser")
        profile_data = {}

        # Name
        name_element = soup.select_one("p.lead strong")
        if name_element:
            profile_data["name"] = name_element.text.strip()

        # Title
        title_paragraphs = []
        title_p = soup.select_one("p.lead + p")
        if title_p and title_p.find("strong"):
            title_paragraphs.append(title_p.text.strip())

            # Check if there's another title element
            next_p = title_p.find_next("p")
            if next_p and next_p.find("strong") and "Email" not in next_p.text:
                title_paragraphs.append(next_p.text.strip())

        profile_data["title"] = " ".join(title_paragraphs)

        # Email
        emails = self.extract_email(soup)
        profile_data["email"] = emails

        # Education path
        education_lines = []
        # Find a paragraph that contains education-related terms
        for p in soup.select("p"):
            text = p.text.strip()
            if any(
                edu_term in text.lower()
                for edu_term in ["tiến sỹ", "thạc sỹ", "kỹ sư", "đại học", "bằng"]
            ):
                if "Email" not in text and len(text.split("\n")) > 1:
                    education_lines = [
                        line.strip() for line in text.split("\n") if line.strip()
                    ]
                    break

        profile_data["education_path"] = education_lines

        # Research fields
        profile_data["research_field"] = self.extract_text_from_section(
            soup, "Lĩnh vực nghiên cứu"
        )

        # Interested fields
        profile_data["interested_field"] = self.extract_text_from_section(
            soup, "Các nghiên cứu quan tâm"
        )

        # Introduction
        intro_section = None
        for element in soup.find_all(["h3", "h2", "h4"]):
            if "giới thiệu" in element.text.lower():
                intro_section = element.parent
                break

        intro_text = ""
        if intro_section:
            intro_text = self.extract_paragraphs_until_next_section(soup, intro_section)
        profile_data["introduction"] = intro_text

        # Notable publications - try different section titles and different tag types
        profile_data["notable_publication"] = self.extract_text_from_section(
            soup,
            ["Các công trình khoa học tiêu biểu", "Công trình khoa học tiêu biểu"],
            ["ul", "ol"],
        )

        # Awards and nominations
        profile_data["awards"] = self.extract_text_from_section(
            soup, ["Giải thưởng, khen thưởng", "Giải thưởng"]
        )

        # Teaching subjects
        profile_data["teaching_subjects"] = self.extract_text_from_section(
            soup, "Giảng dạy"
        )

        # Current projects - try different possible section titles and tag types
        profile_data["current_project"] = self.extract_text_from_section(
            soup,
            [
                "Dự án hiện tại",
                "Các dự án đang thực hiện",
                "Các dự án hiện tại",
                "Dự án đang thực hiện",
            ],
            ["ul", "ol"],
        )

        return profile_data

    def get_all_profile_urls(self, staff_page_url):
        """Get all profile URLs from the staff listing page."""
        html_content = self.get_page(staff_page_url)
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        profile_links = []

        # Find all profile links - adapt this to match the actual structure of the staff page
        for link in soup.select("a"):
            href = link.get("href")
            if href and (".html" in href or "/pgs-" in href or "/ts-" in href):
                if any(
                    term in href.lower()
                    for term in ["/pgs-", "/ts-", "/gv-", "/officer/"]
                ):
                    full_url = urljoin(self.base_url, href)
                    if full_url not in profile_links:
                        profile_links.append(full_url)

        return profile_links

    def crawl_profiles(
        self, urls=None, staff_page_url=None, output_file="profiles.json"
    ):
        """Crawl multiple profile pages and save to JSON file."""
        results = []

        # If specific URLs are provided, use them
        if urls:
            profile_urls = urls
        # Otherwise try to get them from the staff page
        elif staff_page_url:
            profile_urls = self.get_all_profile_urls(staff_page_url)
            print(f"Found {len(profile_urls)} profile URLs")
        else:
            print("No URLs provided.")
            return []

        for i, url in enumerate(profile_urls):
            print(f"[{i+1}/{len(profile_urls)}] Crawling: {url}")
            html_content = self.get_page(url)
            if html_content:
                profile_data = self.parse_profile(html_content)
                profile_data["url"] = url
                results.append(profile_data)
                # Save incrementally in case of interruption
                if (i + 1) % 5 == 0 or (i + 1) == len(profile_urls):
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                    print(f"Saved {len(results)} profiles to {output_file}")
                time.sleep(1)  # Be nice to the server

        return results
