import numpy as np
from pathlib import Path
from typing import Union
import os
from dotenv import set_key
from gensim import downloader

def generate_csv_data(path_to_embeddings: Union[str, os.PathLike] = Path.cwd() / Path("resources/data"),
                    vector_size: int = 300,
                    preloaded_data: bool = True,
                    corpus_name: str = "word2vec-google-news-300",
                    single_file: bool = False,
                    words_filename: Union[str, os.PathLike] = "words.txt",
                    embeddings_filename : Union[str, os.PathLike] = "words_embeddings.npy") -> os.PathLike:

    """ Utility function for loading raw embedding data. Expects your embeddings to be in .npy format,
        so if they are not consider using convert_to_npy() if you have them in csv or txt formats
        All files are expected to have UTF-8 encoding
        (CAUTION! If using downloaded corpus, takes lot's of system resources (7+GB of RAM, around same of storage space for default corpus)
    Args:
        path_to_embeddings (Union[str, bytes, os.PathLike], optional): Path to your preloaded embeddings. Defaults to bundled with project embeddings.
        vector_size (int): Size of embedding vectors. Defaults to 300.
        preloaded_data (bool, optional): Specifies to use word corpus stored locally or download one (by default uses gensim's .load() function and google-news-300 corpus). Defaults to True.
        corpus_name (str, optional): (WARNING! Expects to recieve specific name of corpus to use (options can be viewed using downloader.info()). Defaults to word2vec-google-news-300. 
        single_file (bool, optional): Specifies to use one (see words_single_file.csv in ./resources/csv) or two data files. Defaults to False.
        words_filename (Union[str, bytes, os.PathLike], optional): Filename of file containing all used words. Defaults to words.txt.
        embeddings_filename (Union[str, bytes, os.PathLike], optional): Filename of file containing all embeddings for used words. Defaults to words_embeddings.npy.
    """

    if vector_size <= 0:
        raise ValueError("Vector size must be positive integer")
    
    set_key(Path.cwd() / Path(".env"), "VECTOR_SIZE", str(vector_size))

    path_to_csvs = Path.cwd() / Path("resources/csv")

    if preloaded_data and not single_file:
        
        # Formatting for PostgreSQL COPY-able usage
        words = np.loadtxt(Path(path_to_embeddings) / Path(words_filename), dtype=str, converters={0: lambda x: "\"" + x + "\""}, encoding='UTF-8')
        embeddings = np.load(Path(path_to_embeddings) / Path(embeddings_filename), allow_pickle=False)

        # Samee thing here
        str_embeddings = np.array(["\"" + np.array2string(_, separator=',', sign='-', floatmode='fixed', suppress_small=True, max_line_width=5000, formatter={"float": lambda x: str(x).strip()}) + "\"" for _ in embeddings], dtype=str)
        # Default words and embeddings contain duplicates, so we need to get rid of them
        csv_embeddings = np.column_stack((words, str_embeddings))
        # Because np.unique reorders array's entries in unpredicatable way, we need to:
        # 1. Get indexes of all unique entries
        _, unique_indeces = np.unique(csv_embeddings, return_index=True, axis=0)
        # 2. Restore original order
        sorted_indices = np.sort(unique_indeces)
        # 3. Store only unique entries, while retaining the original order
        unique_csv_embeddings = csv_embeddings[sorted_indices]

        # Saving our import-ready csv file
        np.savetxt(path_to_csvs / Path('words.csv'), unique_csv_embeddings, delimiter=',', fmt='%s', encoding='UTF-8')
        return path_to_csvs / Path('words.csv')
    
    elif preloaded_data and single_file:
        csv_embeddings = np.loadtxt(Path(path_to_embeddings) / Path(embeddings_filename), delimiter=',', converters={0: lambda x: "\"" + str(x).strip() + "\"", 
                                                                                                                     1: lambda x: "\"" + str(x).strip(),
                                                                                                                     vector_size: lambda x: str(x).strip() + "\""}, dtype=str, encoding='UTF-8')
        # Because np.unique reorders array's entries in unpredicatable way, we need to:
        # 1. Get indexes of all unique entries
        _, unique_indeces = np.unique(csv_embeddings, return_index=True, axis=0)
        # 2. Restore original order
        sorted_indices = np.sort(unique_indeces)
        # 3. Store only unique entries, while retaining the original order
        unique_csv_embeddings = csv_embeddings[sorted_indices]

        # Saving our import-ready csv file
        np.savetxt(path_to_csvs / Path('words.csv'), unique_csv_embeddings, delimiter=',', fmt='%s', encoding='UTF-8')
        return path_to_csvs / Path('words.csv')
    
    else:
        # Expects to have word vocab in words file
        loaded_model = downloader.load(name=corpus_name)
        words = np.loadtxt(Path(path_to_embeddings) / Path(words_filename), dtype=str, encoding='UTF-8')
        # Loads embeddings for words in vocab
        embeddings = np.array([loaded_model[word] for word in words], dtype='float32')
        # embedding array preformatting
        str_embeddings = np.array(["\"" + np.array2string(_, separator=',', sign='-', floatmode='fixed', suppress_small=True, max_line_width=5000, formatter={"float": lambda x: str(x).strip()}) + "\"" for _ in embeddings], dtype=str)
        # word array preformatting
        words = ["\"" + word + "\"" for word in words]
        # Default words and embeddings contain duplicates, so we need to get rid of them
        csv_embeddings = np.column_stack((words, str_embeddings))
        # Because np.unique reorders array's entries in unpredicatable way, we need to:
        # 1. Get indexes of all unique entries
        _, unique_indeces = np.unique(csv_embeddings, return_index=True, axis=0)
        # 2. Restore original order
        sorted_indices = np.sort(unique_indeces)
        # 3. Store only unique entries, while retaining the original order
        unique_csv_embeddings = csv_embeddings[sorted_indices]

        # Saving our import-ready csv file
        np.savetxt(path_to_csvs / Path('words.csv'), unique_csv_embeddings, delimiter=',', fmt='%s', encoding='UTF-8')
        return path_to_csvs / Path('words.csv')