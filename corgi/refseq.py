from dataclasses import dataclass
from os import access
import urllib.request
import requests
import humanize
import sys
import gzip
from Bio import SeqIO
import re
import progressbar
import h5py
import pandas as pd
import asyncio
import httpx
from appdirs import user_data_dir
from bs4 import BeautifulSoup

from pathlib import Path
from . import tensor


REFSEQ_CATEGORIES = [
    "archaea",
    "bacteria",
    "fungi",
    "invertebrate",
    "mitochondrion",
    "plant",
    "plasmid",
    "plastid",
    "protozoa",
    "vertebrate_mammalian",
    "vertebrate_other",
    "viral",
]

def root_dir():
    """
    Returns the path to the root directory of this project.

    This is useful for finding the data directory.
    """
    return Path(__file__).parent.resolve()


def global_data_dir():
    """ Returns the path to the directory to hold all the data from the VBA website. """
    return root_dir() / "data"


def filesize_readable(path: (str, Path)) -> str:
    path = Path(path)
    if not path.exists():
        return f"File {path} does not exist."
    return humanize.naturalsize(path.stat().st_size)


@dataclass
class RefSeqCategory:
    name: str
    max_files: int = None
    base_dir: str = global_data_dir()

    def __getstate__(self):
        # Only returns required elements
        # Needed because h5 files cannot be pickled
        return dict(name=self.name, max_files=self.max_files, base_dir=self.base_dir)

    def base_url(self):
        return f"https://ftp.ncbi.nlm.nih.gov/refseq/release/{self.name}"

    def filename(self, index) -> str:
        """ The filename in the RefSeq database for this index. """
        if self.max_files:
            assert index < self.max_files

        return f"{self.name}.{index+1}.1.genomic.fna.gz"

    def fasta_url(self, index: int) -> str:
        """ The url for the fasta file for this index online. """
        return f"{self.base_url()}/{self.filename(index)}"

    def index_html(self):
        index_path = Path(user_data_dir())/f"{self.name}.index.html"
        if not index_path.exists():
            url = self.base_url()
            print("Downloading:", url)
            urllib.request.urlretrieve(url, index_path)
        with open(index_path, 'r') as f:
            contents = f.read()
            soup = BeautifulSoup(contents, 'html.parser')
        return soup

    async def download(self, index: int):
        local_path = self.fasta_path(index, download=False)
        
        if not local_path.exists():
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            async with httpx.AsyncClient(limits=limits) as client:
                url = self.fasta_url(index)
                print(f"downloading {url}")
                response = await client.get(url)    
                open(local_path, 'wb').write(response.content)
                print(f"done {local_path}")
        return local_path

    async def download_all_async(self):
        paths = []
        max_files = self.max_files_available()
        if self.max_files:
            max_files = min(max_files, self.max_files)
        print(f"max_files = {max_files}")
        for index in range(max_files):
            paths.append(self.download(index))
        await asyncio.gather(*paths)

    def download_all(self):
        max_files = self.max_files_available()
        if self.max_files:
            max_files = min(max_files, self.max_files)
        print(f"max_files = {max_files}")
        for index in range(max_files):
            print(f"{self.name} - {index}")
            try:
                self.fasta_path(index, download=True)
            except:
                print(f"failed to download. {self.name} file {index}")

    def fasta_path(self, index: int, download=True) -> Path:
        """
        Returns the local path for the file at this index.

        If the file does not exist already in the base_dir then it is downloaded.
        """
        local_path = Path(self.base_dir) / self.name / self.filename(index)
        local_path.parent.mkdir(exist_ok=True, parents=True)
        if download and not local_path.exists():
            url = self.fasta_url(index)
            print("Downloading:", url)
            urllib.request.urlretrieve(url, local_path)
        return local_path

    def h5_path(self):
        base_dir = Path(self.base_dir)
        base_dir.mkdir(exist_ok=True, parents=True)
        return base_dir / f"{self.name}.h5"

    def fasta_seq_count(self, index: int) -> int:
        fasta_path = self.fasta_path(index)
        seq_count = 0
        with gzip.open(fasta_path, "rt") as fasta:
            for line in fasta:
                if line.startswith(">"):
                    seq_count += 1
        return seq_count

    def get_seq(self, accession):
        if not hasattr(self, 'read_h5'):
            self.read_h5 = h5py.File(self.h5_path(), "r")
        try:
            return self.read_h5[self.dataset_key(accession)]
        except:
            print(f"Failed to read {accession} in {self.name}")
            return []

    def dataset_key(self, accession):
        return f"/{self.name}/{accession}"

    def max_files_available(self):
        max_files = 0
        soup = self.index_html()
        for link in soup.findAll("a"):
            m = re.match(self.name + r"\.(\d+)\.1\.genomic\.fna\.gz", link.get("href"))
            if m:
                max_files = max(int(m.group(1)), max_files)
        return max_files

    def write_h5(self, show_bar=True, file_indexes=None):
        result = []
        if not sys.stdout.isatty():
            show_bar = False

        if file_indexes is None or len(file_indexes) == 0:
            max_files = self.max_files_available()
            if self.max_files:
                max_files = min(max_files, self.max_files)

            file_indexes = range(max_files)

        with h5py.File(self.h5_path(), "a") as h5:
            for file_index in file_indexes:
                print(f"Preprocessing file {file_index} from {self.name}", flush=True)
                # Try to get next file
                try:
                    fasta_path = self.fasta_path(file_index)
                    seq_count = self.fasta_seq_count(file_index)
                except:
                    # If it fails, then assume it doesn't exist and exit
                    print(f"Fasta file at index {file_index} for {self.name} not found.")
                    continue
                    
                with gzip.open(fasta_path, "rt") as fasta:
                    if show_bar:
                        bar = progressbar.ProgressBar(max_value=seq_count - 1)
                    seqs = SeqIO.parse(fasta, "fasta")
                    for i, seq in enumerate(seqs):
                        dataset_key = self.dataset_key(seq.name)

                        # Check if we already have this dataset. If not then add.
                        if not dataset_key in h5:
                            dset = h5.create_dataset(
                                dataset_key,
                                data=tensor.dna_seq_to_numpy(seq),
                                dtype="u1",
                                compression="gzip",
                                compression_opts=9,
                            )

                        result.append( dict(category=self.name, accession=seq.name, file_index=file_index) )
                        if i % 20 == 0:
                            if show_bar:
                                bar.update(i)
                    if show_bar:
                        bar.update(i)
                print()
        
        df = pd.DataFrame(result)
        return df
        
    def h5_filesize(self) -> str:
        return filesize_readable(self.h5_path())

    def fasta_filesize(self, index) -> str:
        return filesize_readable(self.fasta_path(index))

    def total_fasta_filesize(self) -> str:
        size = sum(self.fasta_path(i).stat().st_size for i in range(self.max_files))
        return humanize.naturalsize(size)

    def total_fasta_filesize_server_bytes(self) -> int:
        size = 0
        for index in range(self.max_files):
            url = self.fasta_url(index)
            response = requests.head(url, allow_redirects=True)
            size += int(response.headers.get("content-length", 0))
        return size

    def total_fasta_filesize_server(self) -> str:
        return humanize.naturalsize(self.total_fasta_filesize_server_bytes())

    # def df(self):
    #     data = []
    #     with h5py.File(self.h5_path(), "r") as h5:

