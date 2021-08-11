from dataclasses import dataclass
import urllib.request
import requests
import humanize
import gzip
from Bio import SeqIO
import progressbar
import h5py

from pathlib import Path


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
class ReqSeqCategory:
    name: str
    max_files: int = None
    base_dir: str = global_data_dir()

    def filename(self, index) -> str:
        """ The filename in the RefSeq database for this index. """
        if self.max_files:
            assert index < self.max_files

        return f"{self.name}.{index+1}.1.genomic.fna.gz"

    def fasta_url(self, index: int) -> str:
        """ The url for the fasta file for this index online. """
        return f"https://ftp.ncbi.nlm.nih.gov/refseq/release/{self.name}/{self.filename(index)}"

    def fasta_path(self, index: int) -> Path:
        """
        Returns the local path for the file at this index.

        If the file does not exist already in the base_dir then it is downloaded.
        """
        local_path = Path(self.base_dir) / self.name / self.filename(index)
        local_path.parent.mkdir(exist_ok=True, parents=True)
        if not local_path.exists():
            url = self.fasta_url(index)
            print("Downloading:", url)
            urllib.request.urlretrieve(url, local_path)
        return local_path

    def h5_path(self):
        return Path(self.base_dir) / f"{self.name}.h5"

    def fasta_seq_count(self, index: int) -> int:
        fasta_path = self.fasta_path(index)
        with gzip.open(fasta_path, "rt") as fasta:
            seqs = SeqIO.parse(fasta, "fasta")
            seq_count = sum(1 for _ in seqs)
        return seq_count

    def write_h5(self):
        with h5py.File(self.h5_path(), "a") as h5:
            for file_index in range(self.max_files):
                fasta_path = self.fasta_path(file_index)
                seq_count = self.fasta_seq_count(file_index)
                with gzip.open(fasta_path, "rt") as fasta:
                    bar = progressbar.ProgressBar(max_value=seq_count - 1)
                    seqs = SeqIO.parse(fasta, "fasta")
                    for i, seq in enumerate(seqs):
                        dataset_name = f"/{self.name}/{seq.name}"
                        # print(dataset_name)

                        # Check if we already have this dataset. If not then add.
                        if not dataset_name in h5:
                            dset = h5.create_dataset(
                                dataset_name,
                                data=fasta_seq_to_numpy(seq),
                                dtype="u1",
                                compression="gzip",
                                compression_opts=9,
                            )

                        if i % 20 == 0:
                            bar.update(i)
                    bar.update(i)

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
