import numpy as np

from fastai.torch_core import TensorBase

vocab_to_int = {'A':1,'C':2,'G':3,'T':4,'N':0}
int_to_vocab = dict(zip(vocab_to_int.values(),vocab_to_int.keys()))


class TensorDNA(TensorBase):
    def show(self, ctx=None, **kwargs):
        items = self.tolist()
        truncate_at = 50
        length = len(items)
        if length > truncate_at:
            midpoint = truncate_at//2
            items = items[:midpoint] + [".."] + items[-midpoint:]
        chars = [int_to_vocab[x] if x in int_to_vocab else str(x) for x in items]
        seq_str = "".join(chars)
        return f"{seq_str} [{length}]"


def fasta_seq_to_tensor(seq):
    seq_as_numpy = np.array(seq, 'c')
    seq_as_numpy = seq_as_numpy.view(np.uint8)
    # Ignore any characters in sequence which are below an ascii value of 'A' i.e. 65
    seq_as_numpy = seq_as_numpy[seq_as_numpy >= ord('A')]
    for character, value in vocab_to_int.items():
        seq_as_numpy[ seq_as_numpy == ord(character) ] = value
    seq_as_numpy = seq_as_numpy[ seq_as_numpy < len(vocab_to_int) ]
    seq_as_numpy = np.array(seq_as_numpy, dtype='u1')
    return TensorDNA(seq_as_numpy) # can I set the dtype here directly rather than going through numpy?


