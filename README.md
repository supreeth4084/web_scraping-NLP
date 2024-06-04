# web_scraping-NLP
This Python script extracts text content from a list of URLs and performs sentiment analysis on the extracted text.


Introduction

This script extracts text content from a list of URLs provided in a CSV file (input.csv) and performs sentiment analysis on the extracted text.

Requirements

Python 3.x
pandas
bs4
requests
nltk
aiohttp
aiofiles
Installation

You can install the required libraries using pip:

Bash
pip install pandas bs4 requests nltk aiohttp aiofiles
Use code with caution.
content_copy
Downloading NLTK Punkt package

The script utilizes the NLTK Punkt package for sentence tokenization. If it's not already installed, the script downloads it automatically.

Usage

Make sure you have the required libraries installed.
Place your CSV file containing the URLs (with columns like 'URL' and optionally 'URL_ID') in the same directory as this script.
Ensure the StopWords and MasterDictionary folders containing stopwords and sentiment lexicon files are present in the same directory.
Run the script from the command line:
Bash
python sentiment_analysis.py
Use code with caution.
content_copy
Output

The script extracts text content from URLs and saves them as separate files named with their corresponding URL IDs in a folder named output_text.
The script performs sentiment analysis and generates a new CSV file named output.csv containing the original URL data along with sentiment scores, readability metrics, and other text analysis results.
The script prints the total execution time in seconds and minutes.
Notes

This script utilizes asynchronous programming for efficient webpage retrieval.
The script calculates various readability metrics including polarity, subjectivity, Fog Index, average sentence length, and more.
The script leverages sentiment lexicons (positive-words.txt and negative-words.txt) located in the MasterDictionary folder for sentiment analysis. Make sure these files contain valid sentiment words.
