import pandas as pd
from . import refseq
import asyncio


def preprocess(categories=None, base_dir=None, max_files=None, file_indexes=None):
    if not categories:
        categories = refseq.REFSEQ_CATEGORIES

    if isinstance(categories, str):
        categories = [categories]

    dfs = []
    for name in categories:
        print(f"Preprocessing {name}", flush=True)
        category = refseq.RefSeqCategory(name=name, max_files=max_files, base_dir=base_dir)
        category_df = category.write_h5(file_indexes=file_indexes)
        dfs.append(category_df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def download(categories=None, base_dir=None, max_files=None):
    if not categories:
        categories = refseq.REFSEQ_CATEGORIES

    if isinstance(categories, str):
        categories = [categories]

    for name in categories:
        category = refseq.RefSeqCategory(name=name, max_files=max_files, base_dir=base_dir)
        print(f"Downloading raw files for {category}")
        category.download_all()
        # asyncio.run(category.download_all())
