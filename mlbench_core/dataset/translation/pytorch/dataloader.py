import os

import torchtext
import torchtext.datasets as nlp_datasets
from mlbench_core.dataset.translation.pytorch import config, WMT14Tokenizer


def _get_nmt_text(batch_first=False, include_lengths=False, tokenizer="spacy"):
    """ Returns the text fields for NMT

    Args:
        batch_first:

    Returns:

    """
    SRC_TEXT = WMT14Tokenizer(
        language="en",
        tokenizer=tokenizer,
        init_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        pad_token=config.PAD_TOKEN,
        unk_token=config.UNK_TOKEN,
        batch_first=batch_first,
        include_lengths=include_lengths,
    )

    TGT_TEXT = WMT14Tokenizer(
        language="de",
        tokenizer=tokenizer,
        init_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        pad_token=config.PAD_TOKEN,
        unk_token=config.UNK_TOKEN,
        batch_first=batch_first,
        include_lengths=include_lengths,
    )

    return SRC_TEXT, TGT_TEXT


def _construct_filter_pred(min_len, max_len):
    filter_pred = lambda x: not (
        len(vars(x)["src"]) < min_len or len(vars(x)["trg"]) < min_len
    )
    if max_len is not None:
        filter_pred = lambda x: not (
            len(vars(x)["src"]) < min_len
            or len(vars(x)["src"]) > max_len
            or len(vars(x)["trg"]) < min_len
            or len(vars(x)["trg"]) > max_len
        )

    return filter_pred


def _pad_vocabulary(math):
    if math == "fp16" or math == "manual_fp16":
        pad_vocab = 8
    elif math == "fp32":
        pad_vocab = 1
    return pad_vocab


class WMT14Dataset(nlp_datasets.WMT14):
    def __init__(
        self,
        root,
        download=True,
        train=False,
        validation=False,
        test=False,
        batch_first=False,
        include_lengths=True,
        min_len=0,
        max_len=None,
        math_precision="fp16",
        max_size=None,
    ):
        """WMT14 Dataset.

        Loads WMT14 dataset.
        Based on `torchtext.datasets.WMT14`

        Args:
            root (str): Root folder of WMT14 dataset
            download (bool): Download dataset
            train (bool): Whether to get the train or validation set.
                Default=True
            batch_first (bool): if True the model uses (batch,seq,feature)
                tensors, if false the model uses (seq, batch, feature)
        """
        self.train = train
        self.batch_first = batch_first
        self.fields = _get_nmt_text(
            batch_first=batch_first, include_lengths=include_lengths
        )
        self.root = root
        self.max_len = max_len
        self.min_len = min_len
        if download:
            path = self.download(root)
        else:
            path = os.path.join(root, "wmt14")

        for i in self.fields:
            i.build_vocab_from_file(
                os.path.join(path, config.VOCAB_FNAME),
                pad=_pad_vocabulary(math_precision),
                max_size=max_size,
            )

        if train:
            path = os.path.join(path, config.TRAIN_FNAME)
        elif validation:
            path = os.path.join(path, config.VAL_FNAME)
        elif test:
            path = os.path.join(path, config.TEST_FNAME)
        else:
            raise NotImplementedError()

        super(WMT14Dataset, self).__init__(
            path=path,
            fields=self.fields,
            exts=config.EXTS,
            filter_pred=_construct_filter_pred(min_len, max_len),
        )

    @property
    def vocab_size(self):
        return self.fields["src"].vocab_size

    def get_special_token_idx(self, token):
        return self.fields["src"].get_special_token_indices()[token]

    def get_raw_item(self, idx):
        return super().__getitem__(idx)

    def get_loader(
        self, batch_size=1, shuffle=False, device=None,
    ):

        train_iter = torchtext.data.BucketIterator(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            sort_within_batch=True,
            device=device,
            sort_key=lambda x: len(x.src),
        )
        return train_iter
