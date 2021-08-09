import unittest
import numpy as np

from corgi import refseq


class TestRefSeq(unittest.TestCase):
    def setUp(self):
        self.mitochondrion = refseq.ReqSeqCategory("mitochondrion", max_files=1)

    def test_filename(self):
        self.assertEqual(
            self.mitochondrion.filename(0), "mitochondrion.1.1.genomic.fna.gz"
        )

    def test_filename_beyond_max(self):
        with self.assertRaises(AssertionError) as context:
            self.mitochondrion.filename(1)

    def test_fasta_url(self):
        self.assertEqual(
            self.mitochondrion.fasta_url(0),
            "https://ftp.ncbi.nlm.nih.gov/refseq/release/mitochondrion/mitochondrion.1.1.genomic.fna.gz",
        )
