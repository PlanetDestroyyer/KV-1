#!/usr/bin/env python3
"""
Knowledge Importer for KV-1

Automatically fetch and import knowledge from various sources:
- arXiv papers
- MathWorld formulas
- Wikipedia articles
- OEIS sequences

Usage:
    # Import all formulas from MathWorld
    python knowledge_importer.py --source mathworld --topic "calculus"

    # Import arXiv papers
    python knowledge_importer.py --source arxiv --query "reinforcement learning" --max 10

    # Import Wikipedia math articles
    python knowledge_importer.py --source wikipedia --category "Mathematics"
"""

import requests
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET


class ArxivImporter:
    """Import papers from arXiv."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 10, category: str = None) -> List[Dict]:
        """
        Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum papers to fetch
            category: arXiv category (e.g., 'cs.AI', 'math.NT')

        Returns:
            List of paper metadata
        """
        params = {
            'search_query': f'cat:{category} AND all:{query}' if category else f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        print(f"[+] Searching arXiv for: {query}")
        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            print(f"[!] Error: {response.status_code}")
            return []

        # Parse XML response
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}

        papers = []
        for entry in root.findall('atom:entry', namespace):
            paper = {
                'title': entry.find('atom:title', namespace).text.strip(),
                'summary': entry.find('atom:summary', namespace).text.strip(),
                'authors': [author.find('atom:name', namespace).text
                           for author in entry.findall('atom:author', namespace)],
                'published': entry.find('atom:published', namespace).text,
                'url': entry.find('atom:id', namespace).text,
                'pdf_url': None
            }

            # Find PDF link
            for link in entry.findall('atom:link', namespace):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
                    break

            papers.append(paper)

        print(f"[+] Found {len(papers)} papers")
        return papers

    def save_papers(self, papers: List[Dict], output_file: str):
        """Save papers to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(papers, f, indent=2)
        print(f"[+] Saved to {output_file}")


class MathWorldImporter:
    """Import formulas from Wolfram MathWorld."""

    BASE_URL = "https://mathworld.wolfram.com"

    def get_topic_list(self) -> List[str]:
        """Get list of mathematical topics."""
        # Major MathWorld topics
        topics = [
            "Algebra",
            "Calculus",
            "Geometry",
            "NumberTheory",
            "Trigonometry",
            "LinearAlgebra",
            "DifferentialEquations",
            "ComplexAnalysis",
            "Topology",
            "SetTheory",
            "Logic",
            "Combinatorics",
            "Probability",
            "Statistics"
        ]
        return topics

    def scrape_topic(self, topic: str) -> Dict:
        """
        Scrape formulas from a MathWorld topic.

        Note: This is a template - actual scraping needs BeautifulSoup
        and respect for robots.txt
        """
        url = f"{self.BASE_URL}/{topic}.html"

        print(f"[+] Fetching {topic} from MathWorld...")
        print(f"    URL: {url}")

        # TODO: Implement actual scraping with BeautifulSoup
        # For now, just return the URL for manual download

        return {
            'topic': topic,
            'url': url,
            'note': 'Visit URL to view formulas. Automated scraping requires BeautifulSoup.'
        }


class WikipediaImporter:
    """Import mathematical articles from Wikipedia."""

    API_URL = "https://en.wikipedia.org/w/api.php"

    def search_category(self, category: str, max_results: int = 50) -> List[Dict]:
        """
        Get articles from a Wikipedia category.

        Args:
            category: Category name (e.g., "Mathematics", "Theorems")
            max_results: Maximum articles to fetch

        Returns:
            List of article titles and URLs
        """
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': max_results,
            'format': 'json'
        }

        print(f"[+] Fetching Wikipedia articles from category: {category}")
        response = requests.get(self.API_URL, params=params)

        if response.status_code != 200:
            print(f"[!] Error: {response.status_code}")
            return []

        data = response.json()
        articles = []

        for member in data.get('query', {}).get('categorymembers', []):
            articles.append({
                'title': member['title'],
                'pageid': member['pageid'],
                'url': f"https://en.wikipedia.org/wiki/{member['title'].replace(' ', '_')}"
            })

        print(f"[+] Found {len(articles)} articles")
        return articles

    def get_article_content(self, title: str) -> str:
        """
        Get Wikipedia article content in plain text.

        Args:
            title: Article title

        Returns:
            Article content
        """
        params = {
            'action': 'query',
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
            'titles': title,
            'format': 'json'
        }

        response = requests.get(self.API_URL, params=params)
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        for page_id, page_data in pages.items():
            return page_data.get('extract', '')

        return ""


class OEISImporter:
    """Import sequences from OEIS (Online Encyclopedia of Integer Sequences)."""

    BASE_URL = "https://oeis.org/search"

    def search(self, query: str) -> List[Dict]:
        """
        Search OEIS for sequences.

        Args:
            query: Search query (sequence name or numbers)

        Returns:
            List of sequences
        """
        params = {
            'q': query,
            'fmt': 'json'
        }

        print(f"[+] Searching OEIS for: {query}")
        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            print(f"[!] Error: {response.status_code}")
            return []

        data = response.json()
        sequences = []

        for result in data.get('results', [])[:10]:  # Limit to 10
            sequences.append({
                'number': result.get('number'),
                'name': result.get('name'),
                'data': result.get('data'),
                'url': f"https://oeis.org/A{result.get('number', '000000'):06d}"
            })

        print(f"[+] Found {len(sequences)} sequences")
        return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Import knowledge from various sources into KV-1"
    )

    parser.add_argument(
        '--source',
        choices=['arxiv', 'mathworld', 'wikipedia', 'oeis'],
        required=True,
        help='Source to import from'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='Search query (for arXiv, Wikipedia, OEIS)'
    )

    parser.add_argument(
        '--topic',
        type=str,
        help='Topic name (for MathWorld)'
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Wikipedia category or arXiv category'
    )

    parser.add_argument(
        '--max',
        type=int,
        default=10,
        help='Maximum results to fetch'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./imported_knowledge',
        help='Output directory for imported knowledge'
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Import based on source
    if args.source == 'arxiv':
        importer = ArxivImporter()
        papers = importer.search(
            query=args.query or "machine learning",
            max_results=args.max,
            category=args.category
        )
        output_file = f"{args.output}/arxiv_papers.json"
        importer.save_papers(papers, output_file)

        print("\n" + "="*60)
        print("ARXIV PAPERS IMPORTED")
        print("="*60)
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}")
            print(f"   URL: {paper['url']}")

    elif args.source == 'mathworld':
        importer = MathWorldImporter()
        if args.topic:
            result = importer.scrape_topic(args.topic)
            print(f"\n[i] Visit: {result['url']}")
        else:
            topics = importer.get_topic_list()
            print("\n" + "="*60)
            print("AVAILABLE MATHWORLD TOPICS")
            print("="*60)
            for topic in topics:
                print(f"  - {topic}")
                print(f"    URL: {importer.BASE_URL}/{topic}.html")

    elif args.source == 'wikipedia':
        importer = WikipediaImporter()
        articles = importer.search_category(
            category=args.category or "Mathematics",
            max_results=args.max
        )

        output_file = f"{args.output}/wikipedia_articles.json"
        with open(output_file, 'w') as f:
            json.dump(articles, f, indent=2)

        print("\n" + "="*60)
        print("WIKIPEDIA ARTICLES IMPORTED")
        print("="*60)
        for i, article in enumerate(articles[:10], 1):
            print(f"{i}. {article['title']}")
            print(f"   {article['url']}")

    elif args.source == 'oeis':
        importer = OEISImporter()
        sequences = importer.search(args.query or "Fibonacci")

        output_file = f"{args.output}/oeis_sequences.json"
        with open(output_file, 'w') as f:
            json.dump(sequences, f, indent=2)

        print("\n" + "="*60)
        print("OEIS SEQUENCES IMPORTED")
        print("="*60)
        for i, seq in enumerate(sequences, 1):
            print(f"{i}. A{seq['number']:06d}: {seq['name']}")
            print(f"   Data: {seq['data']}")


if __name__ == '__main__':
    main()
