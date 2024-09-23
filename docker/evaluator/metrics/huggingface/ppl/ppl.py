import math
import evaluate
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets


class ppl(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            features=datasets.Features({
                'predictions': datasets.Value('string'),
            }),
            description="",
            citation="",
        )
    
    def _download_and_prepare(self, dl_manager):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model_checkpoint = _tokenizer_checkpoint = "gpt2-large"
        self._model = AutoModelForCausalLM.from_pretrained(_model_checkpoint).eval().to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(_tokenizer_checkpoint)
        pass


    def _compute(self, predictions, **kwargs):
        with torch.no_grad():
            seq_ppls = []
            nlls = []
            total = 0
            token_error_s = 0
            seq_error_s = 0
            # predictions = tqdm(predictions, desc='Calculating the token-level and sequence-level perplexity scores...')
            for i, sample in enumerate(predictions):
                text_ids = self._tokenizer.encode(sample, return_tensors="pt", truncation=True, max_length=512).to(self._device)
                if text_ids.shape[1] > 1:
                    input_ids = text_ids[:, :-1]
                    target_ids = text_ids[:, 1:]
                    outputs = self._model(input_ids)
                    preds = outputs.logits[0]
                    calc_loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(preds, dim=1), target_ids[0])
                    neg_log_likelihood = calc_loss * input_ids.shape[1]
                    total += input_ids.shape[1]
                    nlls.append(neg_log_likelihood)

                    loss = self._model(text_ids, labels=text_ids)[0]
                    seq_ppl = np.exp(loss.cpu().detach().numpy())
                    if math.isnan(seq_ppl):
                        seq_error_s += 1
                    else:
                        seq_ppls.append(seq_ppl)
                else:
                    token_error_s += 1
            token_ppl = torch.exp(torch.stack(nlls).sum() / total)
            # return token_ppl.item(), token_error_s, sum(seq_ppl)/len(seq_ppl), seq_error_s

        metrics = {
            "s_ppl": sum(seq_ppls)/len(seq_ppls),
            "t_ppl": token_ppl.item()
        }
        return metrics