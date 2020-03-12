import torch
import torch.distributed as dist
from mlbench_core.evaluation.pytorch.inference.beam_search import SequenceGenerator


def gather_predictions(preds):
    world_size = dist.get_world_size()
    if world_size > 1:
        all_preds = [preds.new(preds.size(0), preds.size(1)) for _ in range(world_size)]
        dist.all_gather(all_preds, preds)
        preds = torch.cat(all_preds)
    return preds


class Translator:
    """
    Translator that output translated sentences from GNMT model by using a Sequence Generator
    """

    def __init__(
        self,
        model,
        trg_tokenizer,
        BOS_idx,
        EOS_idx,
        beam_size=5,
        len_norm_factor=0.6,
        len_norm_const=5.0,
        cov_penalty_factor=0.1,
        max_seq_len=50,
    ):

        self.model = model
        self.tokenizer = trg_tokenizer
        self.insert_target_start = [BOS_idx]
        self.insert_src_start = [BOS_idx]
        self.insert_src_end = [EOS_idx]
        self.batch_first = model.batch_first
        self.beam_size = beam_size

        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            BOS_idx=BOS_idx,
            EOS_idx=EOS_idx,
            max_seq_len=max_seq_len,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor,
        )

    def get_detokenized_target(self, trg, batch_size):
        targets = []
        if self.batch_first:
            for i in range(batch_size):
                t = trg[i]
                targets.append(self.tokenizer.detokenize(t))
        else:
            for i in range(batch_size):
                t = trg[:, i]
                targets.append(self.tokenizer.detokenize(t))

        return targets

    def translate(self, src, trg):
        """ Given a source a target tokenized tensors, outputs the
        non-tokenized translation from the model, as well as the non-tokenized target

        Args:
            src:
            trg:

        Returns:

        """
        src, src_len = src
        trg, trg_len = trg
        device = next(self.model.parameters()).device

        if self.batch_first:
            batch_size = src.shape[0]
        else:
            batch_size = src.shape[1]

        bos = [self.insert_target_start] * (batch_size * self.beam_size)
        bos = torch.tensor(bos, dtype=torch.int64, device=device)
        if self.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)

        if self.beam_size == 1:
            generator = self.generator.greedy_search
        else:
            generator = self.generator.beam_search

        with torch.no_grad():
            context = self.model.encode(src, src_len)
            context = [context, src_len, None]
            preds, lengths, counter = generator(batch_size, bos, context)

        preds = gather_predictions(preds).cpu()

        output = []
        targets = []
        targets += self.get_detokenized_target(trg, batch_size)
        for pred in preds:
            pred = pred.tolist()
            detok = self.tokenizer.detokenize(pred)
            output.append(detok)

        return output, targets