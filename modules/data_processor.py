# Modificar la parte inicial del archivo modules/data_processor.py
import os
import re
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
import time
import hashlib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import openai
import umap
import warnings
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType, 
    IntegerType,
    FloatType,
    ArrayType,
    BooleanType
)

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataProcessor:
    """Processor for loading, preprocessing, and extracting features from data."""

    def __init__(self, config, logger, spark_manager):
        """
        Initializes the data processor.

        Args:
            config: Configuration manager
            logger: Logger instance
            spark_manager: Spark session manager
        """
        self.config = config
        self.logger = logger
        self.spark_manager = spark_manager
        self.input_file = config.get_input_file_path()
        self.output_file = config.get_output_file_path()
        self.text_preprocessor = TextPreprocessor(config)
        self.feature_extractor = FeatureExtractor(config, logger)
        self.logger.info("DataProcessor initialized")
    
    def load_data(self, file_path=None):
        """
        Loads data from a Stata file and handles duplicate entries.

        Args:
            file_path: Optional path to override the default input file.

        Returns:
            Loaded DataFrame (PySpark DataFrame)
        """
        filepath = file_path or self.input_file
        self.logger.info(f"Loading data from {filepath}")
        
        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            # Obtain Spark session
            spark = self.spark_manager.get_or_create_session()
            
            # Load the Stata file into a pandas DataFrame first
            pd_df = pd.read_stata(filepath, convert_categoricals=False)
            
            # Log basic dataset information
            self.logger.info(f"Loaded dataset with {pd_df.shape[0]} rows and {pd_df.shape[1]} columns")
            
            # Check for and remove exact duplicates
            initial_rows = pd_df.shape[0]
            pd_df = pd_df.drop_duplicates()
            deduped_rows = pd_df.shape[0]
            
            if initial_rows > deduped_rows:
                self.logger.info(f"Removed {initial_rows - deduped_rows} exact duplicate rows")
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = spark.createDataFrame(pd_df)
            
            # Cache the DataFrame for better performance
            spark_df = spark_df.cache()
            
            # Trigger an action to materialize the cache
            row_count = spark_df.count()
            self.logger.info(f"Successfully loaded {row_count} rows into Spark DataFrame")
            
            return spark_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def save_data(self, dataframe, file_path=None):
        """
        Saves data to a Stata file.

        Args:
            dataframe: DataFrame to save (PySpark DataFrame or pandas DataFrame)
            file_path: Optional path that overrides the configuration
        """
        filepath = file_path or self.output_file
        self.logger.info(f"Saving data to {filepath}")
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert Spark DataFrame to pandas DataFrame if needed
            if isinstance(dataframe, SparkDataFrame):
                self.logger.info("Converting Spark DataFrame to pandas DataFrame")
                pd_df = dataframe.toPandas()
            else:
                # Assume it's already a pandas DataFrame
                pd_df = dataframe
            
            # Save the pandas DataFrame to Stata format
            self.logger.info(f"Saving pandas DataFrame with {pd_df.shape[0]} rows and {pd_df.shape[1]} columns")
            pd_df.to_stata(filepath, write_index=False)
            
            self.logger.info(f"Successfully saved data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise RuntimeError(f"Failed to save data: {str(e)}")
    
    def preprocess_text_columns(self, dataframe, text_columns=None):
        """
        Preprocesses all text columns in the DataFrame.
        
        Args:
            dataframe: DataFrame with text columns
            text_columns: Optional list of column names to process (default: use config)
            
        Returns:
            DataFrame with preprocessed text columns
        """
        columns_to_process = text_columns or self.config.get_text_columns()
        self.logger.info(f"Preprocessing {len(columns_to_process)} text columns")
        
        processed_df = dataframe
        
        for column in columns_to_process:
            if column not in dataframe.columns:
                self.logger.warning(f"Column '{column}' not found in DataFrame, skipping")
                continue
            
            self.logger.info(f"Preprocessing column: {column}")
            
            # For Spark DataFrame, use UDF to apply preprocessing
            if isinstance(dataframe, SparkDataFrame):
                # Create a UDF (User Defined Function) for preprocessing
                preprocessor_udf = F.udf(self.text_preprocessor.preprocess_text, StringType())
                
                # Apply the UDF to the column
                processed_df = processed_df.withColumn(
                    f"{column}_preprocessed", 
                    preprocessor_udf(F.col(column))
                )
            else:
                # For pandas DataFrame, use apply
                processed_df[f"{column}_preprocessed"] = processed_df[column].apply(
                    self.text_preprocessor.preprocess_text
                )
            
            self.logger.info(f"Completed preprocessing for column: {column}")
        
        return processed_df
    
    def extract_features(self, dataframe, text_columns=None, feature_method=None):
        """
        Extracts features from text columns.
        
        Args:
            dataframe: DataFrame with text columns
            text_columns: Optional list of column names to process (default: use config)
            feature_method: Optional method override (default: use config)
            
        Returns:
            Tuple of (DataFrame with features, features_dict)
        """
        columns_to_process = text_columns or self.config.get_text_columns()
        method = feature_method or self.config.get_config_value('feature_extraction.method', 'hybrid')
        
        self.logger.info(f"Extracting features from {len(columns_to_process)} text columns using {method} method")
        
        features_dict = {}
        
        # Convert to pandas for feature extraction if it's a Spark DataFrame
        if isinstance(dataframe, SparkDataFrame):
            pd_df = dataframe.toPandas()
        else:
            pd_df = dataframe
        
        # Process each column
        for column in columns_to_process:
            preprocessed_column = f"{column}_preprocessed"
            
            if preprocessed_column not in pd_df.columns:
                self.logger.warning(
                    f"Preprocessed column '{preprocessed_column}' not found. "
                    f"Falling back to original column '{column}'."
                )
                preprocessed_column = column
            
            if preprocessed_column not in pd_df.columns:
                self.logger.error(f"Column '{preprocessed_column}' not found in DataFrame, skipping")
                continue
            
            # Get non-null texts
            texts = pd_df[preprocessed_column].dropna().tolist()
            
            if not texts:
                self.logger.warning(f"No valid texts found in column '{preprocessed_column}', skipping")
                continue
            
            self.logger.info(f"Extracting features from column: {preprocessed_column}")
            
            # Extract features based on method
            if method == 'tfidf':
                features = self.feature_extractor.extract_tfidf_features(texts)
                features_dict[column] = features
            elif method == 'embedding':
                features = self.feature_extractor.extract_embeddings(texts)
                features_dict[column] = features
            elif method == 'hybrid':
                # For hybrid, extract both and concatenate later when needed
                tfidf_features = self.feature_extractor.extract_tfidf_features(texts)
                embedding_features = self.feature_extractor.extract_embeddings(texts)
                features_dict[f"{column}_tfidf"] = tfidf_features
                features_dict[f"{column}_embedding"] = embedding_features
            else:
                self.logger.error(f"Unknown feature extraction method: {method}")
                raise ValueError(f"Unknown feature extraction method: {method}")
        
        self.logger.info(f"Feature extraction completed for {len(features_dict)} features")
        
        return pd_df, features_dict


class TextPreprocessor:
    """Preprocessor for text data."""

    def __init__(self, config):
        """
        Initializes the text preprocessor.

        Args:
            config: Configuration manager
        """
        self.config = config
        self.preprocessing_options = config.get_preprocessing_options()
        
        # Initialize preprocessing components
        self.lemmatizer = WordNetLemmatizer() if self.preprocessing_options.get('lemmatize', False) else None
        
        # Get stopwords
        self.stopwords = set()
        if self.preprocessing_options.get('remove_stopwords', True):
            self.stopwords = set(stopwords.words('english'))
            
        # Add custom stopwords if provided
        custom_stopwords = self.preprocessing_options.get('custom_stopwords', [])
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Minimum word length
        self.min_word_length = self.preprocessing_options.get('min_word_length', 2)
        
        # Maximum text length (in characters)
        self.max_length = self.preprocessing_options.get('max_length', 10000)
        
        # Punctuation to remove
        self.punctuation = set(string.punctuation)
    
    def preprocess_text(self, text):
        """
        Preprocesses a text string with enhanced handling.

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text as a string
        """
        # Handle None/NaN values
        if text is None or pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Truncate text if it's too long
        if self.max_length > 0 and len(text) > self.max_length:
            text = text[:self.max_length]
        
        # Convert to lowercase if enabled
        if self.preprocessing_options.get('lowercase', True):
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove file paths
        text = re.sub(r'[a-zA-Z]:\\[\\\S|*\S]?.*', ' ', text)
        
        # Tokenize text
        tokens = word_tokenize(text)

        # Process each token
        processed_tokens = []
        for token in tokens:
            # Skip short words
            if len(token) < self.min_word_length:
                continue
            
            # Remove punctuation if enabled
            if self.preprocessing_options.get('remove_punctuation', True):
                if all(char in self.punctuation for char in token):
                    continue
                token = ''.join(char for char in token if char not in self.punctuation)
                
                # Skip if token is too short after removing punctuation
                if len(token) < self.min_word_length:
                    continue
            
            # Remove stopwords if enabled
            if self.preprocessing_options.get('remove_stopwords', True) and token in self.stopwords:
                continue
            
            # Apply lemmatization if enabled
            if self.preprocessing_options.get('lemmatize', False) and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        # Reconstruct text from tokens
        preprocessed_text = ' '.join(processed_tokens)
        
        return preprocessed_text

    def preprocess_column(self, dataframe, column_name):
        """
        Preprocesses a text column in a DataFrame.

        Args:
            dataframe: DataFrame containing the column
            column_name: Name of the column to preprocess

        Returns:
            DataFrame with the preprocessed column
        """
        if column_name not in dataframe.columns:
            warnings.warn(f"Column '{column_name}' not found in DataFrame")
            return dataframe
        
        if isinstance(dataframe, SparkDataFrame):
            # For Spark DataFrame, use UDF
            preprocessor_udf = F.udf(self.preprocess_text, StringType())
            return dataframe.withColumn(
                f"{column_name}_preprocessed", 
                preprocessor_udf(F.col(column_name))
            )
        else:
            # For pandas DataFrame
            dataframe[f"{column_name}_preprocessed"] = dataframe[column_name].apply(self.preprocess_text)
            return dataframe


class FeatureExtractor:
    """Feature extractor for texts."""

    def __init__(self, config, logger):
        """
        Initializes the feature extractor.

        Args:
            config: Configuration manager
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.feature_config = config.get_feature_extraction_config()
        
        # Initialize extractors based on configuration
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        
        # Initialize cache dictionary
        self.feature_cache = {}
        
        # Create cache directory if enabled
        self.cache_enabled = self.config.get_config_value('performance.cache_embeddings', True)
        self.cache_directory = self.config.get_config_value('performance.cache_directory', 'cache')
        
        if self.cache_enabled:
            os.makedirs(self.cache_directory, exist_ok=True)
            self.logger.info(f"Feature caching enabled. Cache directory: {self.cache_directory}")
    
    def extract_tfidf_features(self, texts):
        """
        Extracts TF-IDF features from texts.

        Args:
            texts: List of preprocessed texts

        Returns:
            TF-IDF feature matrix (scipy sparse matrix or numpy array)
        """
        # Check cache first
        cache_key = self._get_cache_key(texts, 'tfidf')
        cached_features = self.load_cached_features('tfidf', cache_key)
        if cached_features is not None:
            self.logger.info(f"Loaded TF-IDF features from cache for {len(texts)} texts")
            return cached_features
        
        self.logger.info(f"Extracting TF-IDF features for {len(texts)} texts")
        
        # Get TF-IDF configuration
        tfidf_config = self.feature_config.get('tfidf', {})
        max_features = tfidf_config.get('max_features', 5000)
        ngram_range = tuple(tfidf_config.get('ngram_range', [1, 2]))
        min_df = tfidf_config.get('min_df', 5)
        
        # Initialize vectorizer if not already done
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                lowercase=False,  # Text is already preprocessed
                token_pattern=r'\S+',  # Any non-space sequence (words are already tokenized)
            )
        
        # Extract features
        try:
            features = self.tfidf_vectorizer.fit_transform(texts)
            
            # Normalize features for better clustering performance
            features = normalize(features, norm='l2', axis=1)
            
            # Cache features
            if self.cache_enabled:
                self.cache_features(features, 'tfidf', cache_key)
            
            self.logger.info(f"Extracted TF-IDF features with shape: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting TF-IDF features: {str(e)}")
            raise RuntimeError(f"Failed to extract TF-IDF features: {str(e)}")
    
    def extract_embeddings(self, texts):
        """
        Extracts embeddings using the configured model.

        Args:
            texts: List of preprocessed texts

        Returns:
            Embedding matrix (numpy array)
        """
        # Get embedding configuration
        embedding_config = self.feature_config.get('embedding', {})
        model_type = embedding_config.get('model', 'sentence-transformers')
        
        # Check cache first
        cache_key = self._get_cache_key(texts, model_type)
        cached_features = self.load_cached_features(model_type, cache_key)
        if cached_features is not None:
            self.logger.info(f"Loaded {model_type} embeddings from cache for {len(texts)} texts")
            return cached_features
        
        self.logger.info(f"Extracting {model_type} embeddings for {len(texts)} texts")
        
        # Extract embeddings based on model type
        if model_type == 'openai':
            embeddings = self.extract_openai_embeddings(texts)
        elif model_type == 'sentence-transformers':
            embeddings = self.extract_sentence_transformer_embeddings(texts)
        else:
            self.logger.error(f"Unknown embedding model type: {model_type}")
            raise ValueError(f"Unknown embedding model type: {model_type}")
        
        # Apply dimensionality reduction if configured
        dim_reduction_config = embedding_config.get('dimensionality_reduction', {})
        if dim_reduction_config and dim_reduction_config.get('method'):
            embeddings = self.reduce_dimensionality(embeddings)
        
        # Cache embeddings
        if self.cache_enabled:
            self.cache_features(embeddings, model_type, cache_key)
        
        return embeddings
    
    def extract_openai_embeddings(self, texts):
        """
        Extracts embeddings using the OpenAI API.

        Args:
            texts: List of preprocessed texts

        Returns:
            Embedding matrix (numpy array)
        """
        openai_config = self.feature_config.get('embedding', {}).get('openai', {})
        model_name = openai_config.get('model_name', 'text-embedding-ada-002')
        api_key_env = openai_config.get('api_key_env', 'OPENAI_API_KEY')
        batch_size = openai_config.get('batch_size', 20)
        max_retries = openai_config.get('max_retries', 3)
        backoff_factor = openai_config.get('backoff_factor', 2)
        
        # Get API key from environment
        api_key = os.environ.get(api_key_env)
        if not api_key:
            self.logger.error(f"OpenAI API key not found in environment variable: {api_key_env}")
            raise RuntimeError(f"OpenAI API key not found in environment variable: {api_key_env}")
        
        # Initialize OpenAI client
        openai.api_key = api_key
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            self.logger.info(f"Processing OpenAI embeddings batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Retry logic for API calls
            for retry in range(max_retries):
                try:
                    response = openai.Embedding.create(
                        model=model_name,
                        input=batch
                    )
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = backoff_factor ** retry
                        self.logger.warning(
                            f"OpenAI API call failed (attempt {retry+1}/{max_retries}), "
                            f"retrying in {wait_time} seconds: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"OpenAI API call failed after {max_retries} attempts: {str(e)}")
                        raise RuntimeError(f"Failed to extract OpenAI embeddings: {str(e)}")
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings)
        
        self.logger.info(f"Extracted OpenAI embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def extract_sentence_transformer_embeddings(self, texts):
        """
        Extracts embeddings using Sentence Transformers.

        Args:
            texts: List of preprocessed texts

        Returns:
            Embedding matrix (numpy array)
        """
        st_config = self.feature_config.get('embedding', {}).get('sentence_transformers', {})
        model_name = st_config.get('model_name', 'all-MiniLM-L6-v2')
        
        # Initialize model if not already done
        if self.sentence_transformer is None:
            try:
                self.logger.info(f"Loading Sentence Transformer model: {model_name}")
                self.sentence_transformer = SentenceTransformer(model_name)
            except Exception as e:
                self.logger.error(f"Error loading Sentence Transformer model: {str(e)}")
                raise RuntimeError(f"Failed to load Sentence Transformer model: {str(e)}")
        
        # Extract embeddings
        try:
            batch_size = self.config.get_config_value('performance.batch_size', 32)
            # More robust way to check logging level without assuming logger structure
            try:
                # First attempt: check if logger has a level attribute directly
                show_progress_bar = hasattr(self.logger, 'level') and self.logger.level <= logging.INFO
            except AttributeError:
                try:
                    # Second attempt: check if logger has a logger attribute with level
                    show_progress_bar = hasattr(self.logger, 'logger') and self.logger.logger.level <= logging.INFO
                except AttributeError:
                    # Fallback: don't show progress bar
                    show_progress_bar = False
            
            self.logger.info(f"Extracting embeddings with batch size {batch_size}")
            embeddings = self.sentence_transformer.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=True
            )
            
            self.logger.info(f"Extracted Sentence Transformer embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error extracting Sentence Transformer embeddings: {str(e)}")
            raise RuntimeError(f"Failed to extract Sentence Transformer embeddings: {str(e)}")
        
    def reduce_dimensionality(self, feature_matrix):
        """
        Reduces the dimensionality of a feature matrix using enhanced parameters.

        Args:
            feature_matrix: Feature matrix (NumPy array or SciPy sparse matrix)

        Returns:
            Reduced feature matrix (NumPy array)
        """
        dim_reduction_config = self.feature_config.get('embedding', {}).get('dimensionality_reduction', {})
        method = dim_reduction_config.get('method', 'umap')
        n_components = dim_reduction_config.get('n_components', 50)
        random_state = dim_reduction_config.get('random_state', 42)

        self.logger.info(
            f"Reducing dimensionality from {feature_matrix.shape[1]} to {n_components} "
            f"using {method}"
        )

        try:
            if method == 'umap':
                # UMAP with tuned parameters for better cluster separation
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=30,        # Balance between local and global structure
                    min_dist=0.1,          # Tighter clusters
                    metric='cosine',       # Better suited for text embeddings
                    low_memory=True,       # Enable low memory mode
                    verbose=True,          # Show progress
                    spread=0.8,            # Controls point dispersion
                    densmap=True,          # Preserves local density
                    n_epochs=500           # More training epochs for better stability
                )
                reduced_features = reducer.fit_transform(feature_matrix)

            elif method == 'pca':
                # PCA reduction using scikit-learn
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components, random_state=random_state)
                reduced_features = reducer.fit_transform(feature_matrix)

            elif method == 'tsne':
                # t-SNE reduction using scikit-learn
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=n_components,
                    random_state=random_state,
                    perplexity=30,
                    n_iter=1000,
                    learning_rate=200
                )
                reduced_features = reducer.fit_transform(feature_matrix)

            else:
                self.logger.error(f"Unknown dimensionality reduction method: {method}")
                raise ValueError(f"Unknown dimensionality reduction method: {method}")

            self.logger.info(f"Dimensionality reduced to shape: {reduced_features.shape}")
            return reduced_features

        except Exception as e:
            self.logger.error(f"Error reducing dimensionality: {str(e)}")
            raise RuntimeError(f"Failed to reduce dimensionality: {str(e)}")

    
    def _get_cache_key(self, texts, feature_type):
        """
        Generates a cache key for a set of texts and feature type.
        
        Args:
            texts: List of texts
            feature_type: Type of features
            
        Returns:
            Cache key string
        """
        # Create a hash of the texts and feature configuration
        text_hash = hashlib.md5(''.join(texts[:100]).encode()).hexdigest()
        if feature_type == 'tfidf':
            config_hash = hashlib.md5(str(self.feature_config.get('tfidf', {})).encode()).hexdigest()
        else:
            config_hash = hashlib.md5(str(self.feature_config.get('embedding', {})).encode()).hexdigest()
        
        return f"{feature_type}_{text_hash[:10]}_{config_hash[:10]}"
    
    def cache_features(self, features, feature_type, cache_key=None):
        """
        Caches extracted features.

        Args:
            features: Feature matrix
            feature_type: Type of features
            cache_key: Optional cache key (if None, generate from feature_type)
        """
        if not self.cache_enabled:
            return
        
        key = cache_key or feature_type
        
        # Save to memory cache
        self.feature_cache[key] = features
        
        # Save to disk cache
        cache_path = os.path.join(self.cache_directory, f"{key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
            self.logger.debug(f"Cached {feature_type} features to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache features to disk: {str(e)}")
    
    def load_cached_features(self, feature_type, cache_key=None):
        """
        Loads cached features if available.

        Args:
            feature_type: Type of features
            cache_key: Optional cache key (if None, use feature_type)

        Returns:
            Feature matrix or None if not available
        """
        if not self.cache_enabled:
            return None
        
        key = cache_key or feature_type
        
        # Check memory cache first
        if key in self.feature_cache:
            self.logger.debug(f"Loaded {feature_type} features from memory cache")
            return self.feature_cache[key]
        
        # Check disk cache
        cache_path = os.path.join(self.cache_directory, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                
                # Store in memory cache for future use
                self.feature_cache[key] = features
                
                self.logger.debug(f"Loaded {feature_type} features from disk cache: {cache_path}")
                return features
            except Exception as e:
                self.logger.warning(f"Failed to load features from disk cache: {str(e)}")
        
        return None