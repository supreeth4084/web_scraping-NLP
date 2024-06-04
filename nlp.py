import pandas as pd
from bs4 import BeautifulSoup 
import requests
import nltk
import pandas as pd
from nltk import sent_tokenize, punkt
import os
import re
import time
import asyncio
import aiohttp
import aiofiles


start_time = time.time()


try:
    # Check if the 'punkt' package is already installed.
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # If not installed, download the 'punkt' package.
    import nltk
    nltk.download('punkt')

async def fetch_and_save(session, url, url_id, output_path):
    try:
        async with session.get(url) as response:
            html_content = await response.text()
            soup = BeautifulSoup(html_content, "html.parser")
            title = soup.title.string
    except AttributeError:
            async with session.get(url) as response:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, "html.parser")
                title = soup.title.text 
    async with aiofiles.open(f"{output_path}/{url_id}.txt", "w", encoding='utf-8') as output_file:
        await output_file.write(title) 
        await output_file.write("\n\n")
        for paragraph in soup.find_all('div', {'class': 'td-post-content'}):
            await output_file.write(paragraph.text)
            await output_file.write("\n")

async def extract_text(path1, path2):
    df = pd.read_csv(path1)
    urls = list(df['URL'])
    url_ids = list(df['URL_ID'])

    async with aiohttp.ClientSession() as session:
        tasks = []
        for url, url_id in zip(urls, url_ids):
            tasks.append(fetch_and_save(session, url, url_id, path2))
        await asyncio.gather(*tasks)



async def extract_text(path1, path2):
    df = pd.read_csv(path1)
    urls = list(df['URL'])
    url_ids = list(df['URL_ID'])

    async with aiohttp.ClientSession() as session:
        tasks = []
        for url, url_id in zip(urls, url_ids):
            tasks.append(fetch_and_save(session, url, url_id, path2))
        await asyncio.gather(*tasks)

def tokenizer(path):
  """
  Reads a file, tokenizes it, and removes punctuation characters.
  """
  with open(path, 'rb') as f:
    rawdata = f.read()

  # Decode bytes and perform tokenization in one step
  clean_data = nltk.word_tokenize(rawdata.decode('utf-8', errors='ignore'))

  # Use regular expressions for efficient punctuation removal
  filtered_tokens = [token for token in clean_data if not re.match(r'[^\w\s]', token)]
  return filtered_tokens


def sentence_len(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()

    # Split the text into sentences using nltk.sent_tokenize
        sentences = nltk.sent_tokenize(text)

    # Print the number of sentences
        return len(sentences)

def is_two_syllables(word):

    vowels = 'aeiouyAEIOUY'
    vowel_count = 0
    consecutive_consonants = 0

    for char in word:
        if char in vowels:
            vowel_count += 1
            consecutive_consonants = 0  # reset on vowel
        else:
            consecutive_consonants += 1
        # exceptions for single consonant followed by silent 'e'
        if consecutive_consonants > 2 and word[-1] == 'e':
            consecutive_consonants -= 1

# Two syllables if there are at least two vowels and no more than two consecutive consonants
    return vowel_count >= 2 and consecutive_consonants <= 2

def find_two_syllable_words(path):

    two_syllable_words = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            words = line.strip().lower().split()  # lowercase, remove whitespace, split
            for word in words:
                if is_two_syllables(word):
                    two_syllable_words.append(word)
    return len(two_syllable_words)


def count_syllables(word):
# This function counts the number of syllables in a word.

# Args:
#     word: The word to count syllables in.

# Returns:
#     The number of syllables in the word.
    count = 0
    vowels = "aeiouyAEIOUY"
    word = word.lower()

    if word:
    # Check if the word ends with "es" or "ed" and remove it
        if word.endswith("es") or word.endswith("ed"):
            word = word[:-2]

    # Iterate through the word and count vowels
        for i in range(len(word)):
            if word[i] in vowels:
                count += 1

    return count

def count_syllables_in_text(path):
# """
# This function counts the number of syllables in a text file.

# Args:
#     filepath: The path to the text file.

# Returns:
#     The total number of syllables in the text file.
# """

    total_syllables = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.split()
            for word in words:
                total_syllables += count_syllables(word)

    return total_syllables

def pronouns_count(path):
    pronouns_list = ['i','we','my','ours','I,' ,'we,', 'my,' ,'ours,' ]
    pronuouns = tokenizer(path)
    pronouns = [noun.lower() for noun in pronuouns] 
    pronouns = [noun for noun in pronuouns if noun in pronouns_list]
    return len(pronouns)

def word_len(path):
    with open(path, 'r', encoding="utf-8") as f:
        text = f.read()

        # Remove any whitespace characters
        text = text.replace(" ", "")

# Return the length of the text
    return len(text)


async def main():
    
    results=[]
    await extract_text('input.csv','output_text')

    
    numbers = ["{:04d}".format(i) for i in range(1, 101)]

        # Get the path to the StopWords folder
    stopwords_folder_path = "StopWords"

    # List all files in the StopWords folder
    stopwords_files = os.listdir(stopwords_folder_path)

    # Load the stopwords from each file
    stopwords = []

    string_pattern = '[^\\x00-\\x7f]'
    regex = re.compile(string_pattern.encode('utf-8'))


    for filename in stopwords_files:
        filepath = os.path.join(stopwords_folder_path, filename)
        with open(filepath, 'rb') as f:
            rawdata = f.read()
            clean_data = regex.sub(b'', rawdata)
            stopwords.extend(clean_data.decode('utf-8').splitlines())

    # Remove duplicate stopwords
    stopwords = set(stopwords)
    stopwords = list(stopwords)

    for i in range(len(numbers)):
        full_path = f"output_text/blackassign{numbers[i]}.txt"
        # Open the text file
        text_tokens =  tokenizer(full_path)

        text_tokens_clean = [tok for tok in text_tokens if tok not in stopwords]


        # Initialize positive and negative scores
        positive_score = 0
        negative_score = 0

        # Define positive and negative word dictionaries

        positive_words =  tokenizer('MasterDictionary/positive-words.txt')


        negative_words =  tokenizer('MasterDictionary/negative-words.txt')



        # Calculate positive and negative scores

        positive_score = sum(tok in positive_words for tok in text_tokens_clean)
        negative_score = -sum(tok in negative_words for tok in text_tokens_clean)

        # Calculate polarity, subjectivity, average sentence length, and complex words
        Word_Count = len(text_tokens_clean)
        sentence_lenght = sentence_len(full_path)
        polarity = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity = (positive_score + negative_score)/ (Word_Count + 0.000001 )
        avg_sentence_length = Word_Count / sentence_lenght
        complex_words = find_two_syllable_words(full_path)
        Percentage_of_Complex_words = complex_words / Word_Count
        Fog_Index = 0.4*(avg_sentence_length / Percentage_of_Complex_words)
        Average_Number_of_Words_Per_Sentence = Word_Count / sentence_lenght
        Syllable_Count_Per_Word = count_syllables_in_text(full_path)
        Personal_Pronouns = pronouns_count(full_path)
        Average_Word_Length = word_len(full_path) / Word_Count



        # Create a dictionary with the results
        results.append( {
            'Positive Score': positive_score,
            'Negative Score': negative_score,
            'Polarity': polarity,
            'Subjectivity': subjectivity,
            'Average Sentence Length': avg_sentence_length,
            'Complex Words': complex_words,
            'Percentage of Complex words': Percentage_of_Complex_words,
            'Fog Index': Fog_Index,
            'Average_Number_of_Words_Per_Sentence': Average_Number_of_Words_Per_Sentence,
            'complex_words': complex_words,
            'Word Count': Word_Count,
            'Syllable Count Per Word': Syllable_Count_Per_Word,
            'Personal Pronouns': Personal_Pronouns,
            'Average Word Length': Average_Word_Length
        })
    # Create DataFrame from results list outside the loop
    df_results = pd.DataFrame(results)

    df = pd.read_csv('input.csv')
    df = pd.concat([df,df_results],axis=1)
    # Save the DataFrame to a CSV file
    df.to_csv('output.csv')


if __name__ == "__main__":
    asyncio.run(main())


print("The article text is extracted and saved in a text file\nwith URL_ID as its file name in a folder named output_text \n\nOutput is in the output.csv file.\n")
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.2f} in seconds")
print(f"Total Execution time: {(end_time - start_time) / 60:.2f} in minutes")