import os

import torchtext
from mlbench_core.dataset.translation.pytorch import config, WMT14Tokenizer
from torchtext.data import Example, Dataset


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

    return [("src", SRC_TEXT), ("trg", TGT_TEXT)]


def _construct_filter_pred(min_len, max_len):
    filter_pred = lambda x: not (len(x[0]) < min_len or len(x[1]) < min_len)
    if max_len is not None:
        filter_pred = lambda x: not (
            len(x[0]) < min_len
            or len(x[0]) > max_len
            or len(x[1]) < min_len
            or len(x[1]) > max_len
        )

    return filter_pred


def _pad_vocabulary(math):
    if math == "fp16" or math == "manual_fp16":
        pad_vocab = 8
    elif math == "fp32":
        pad_vocab = 1
    return pad_vocab


def process_data(path, filter_pred, fields, lazy=False):
    """
    Loads data from the input file.
    """

    src_path, trg_path = tuple(os.path.expanduser(path + x) for x in config.EXTS)
    examples = []
    with open(src_path, mode="r", encoding="utf-8") as src_file, open(
        trg_path, mode="r", encoding="utf-8"
    ) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_line, trg_line = src_line.strip(), trg_line.strip()

            should_consider = filter_pred((src_line, trg_line))
            if src_line != "" and trg_line != "" and should_consider:
                if lazy:
                    examples.append((src_line, trg_line))
                else:
                    examples.append(Example.fromlist([src_line, trg_line], fields))
    return examples


class WMT14Dataset(Dataset):
    urls = [
        (
            "https://drive.google.com/uc?export=download&"
            "id=0B_bZck-ksdkpM25jRUN2X2UxMm8",
            "wmt16_en_de.tar.gz",
        )
    ]
    name = "wmt14"

    def __init__(
        self,
        root,
        download=True,
        train=False,
        validation=False,
        lazy=False,
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

        self.batch_first = batch_first
        self.lazy = lazy
        self.fields = _get_nmt_text(
            batch_first=batch_first, include_lengths=include_lengths
        )

        super(WMT14Dataset, self).__init__(examples=[], fields=self.fields)

        self.list_fields = list(self.fields.items())
        self.max_len = max_len
        self.min_len = min_len
        if download:
            path = self.download(root)
        else:
            path = os.path.join(root, "wmt14")

        for i in self.fields.values():
            i.build_vocab_from_file(
                os.path.join(path, config.VOCAB_FNAME),
                pad=_pad_vocabulary(math_precision),
                max_size=max_size,
            )

        if train:
            path = os.path.join(path, config.TRAIN_FNAME)
        elif validation:
            path = os.path.join(path, config.VAL_FNAME)
        else:
            raise NotImplementedError()

        self.examples = process_data(
            path,
            filter_pred=_construct_filter_pred(min_len, max_len),
            fields=self.list_fields,
            lazy=lazy,
        )

    @property
    def vocab_size(self):
        return self.fields["src"].vocab_size

    def get_special_token_idx(self, token):
        return self.fields["src"].get_special_token_indices()[token]

    def get_raw_item(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.lazy:
            src_line, trg_line = self.examples[item]
            return Example.fromlist([src_line, trg_line], self.list_fields)
        else:
            return self.examples[item]

    def __iter__(self):
        for x in self.examples:
            if self.lazy:
                src_line, trg_line = x
                yield Example.fromlist([src_line, trg_line], self.list_fields)
            else:
                yield x

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
