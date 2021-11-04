import pandas as pd
from . import refseq

def preprocess(categories = None, base_dir = None, max_files = None):
    if categories is None:
        categories = refseq.REFSEQ_CATEGORIES
    
    if isinstance(categories, str):
        categories = [categories]
    
    dfs = []
    for name in categories:
        category = refseq.RefSeqCategory(name=name, max_files=max_files, base_dir=base_dir)
        category_df = category.write_h5()
        dfs.append(category_df)

    df = pd.concat(dfs, ignore_index=True)
    
    return df