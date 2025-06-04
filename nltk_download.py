#!/usr/bin/env python3
import nltk

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    
    print("Downloading NLTK resources...")
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"✅ {resource} downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading {resource}: {str(e)}")
    
    print("\nVerifying installation:")
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            print(f"✅ {resource} verified successfully")
        except LookupError:
            print(f"❌ {resource} is not available")

if __name__ == "__main__":
    nltk.data.path.append('/export/home/rcsguest/rcs_jcamacho/nltk_data')
    download_nltk_resources()
