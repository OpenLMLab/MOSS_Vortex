
import os
import time
import threading
import queue
import json
import time
try:
    import thread
except ImportError:
    import _thread as thread
import websocket
from websocket import create_connection

import onnxruntime as ort
import numpy as np
import torch
import onnx
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

class Inferer:
    def __init__(self, model_path, ):
        self.num_layers, self.heads, self.hidden, self.vocab_size = 34, 24, 256, 51200
        self.stopwords = [torch.LongTensor([68, 12162, 29]), torch.LongTensor([68, 1219, 29])]

        self.ort_session = ort.InferenceSession(
            model_path,
            ort.SessionOptions(),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # self.ort_session.run()
        print('loaded')

    def top_k_top_p_filtering(self, logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1, ):
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits


    def topk_search(self, 
                input_ids, 
                attention_mask,
                temperature=0.7, 
                repetition_penalty=1.1, 
                top_k=0, 
                top_p=0.92, 
                max_iterations=1024,
                regulation_start=512,
                length_penalty=1,
                max_time=60):
        assert input_ids.dtype == torch.int64 and attention_mask.dtype == torch.int64

        self.bsz, self.seqlen = input_ids.shape
        self.past_seqlen = 1
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        last_token_indices = attention_mask.sum(1) - 1

        attention_mask = torch.cat([torch.ones((self.bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype), attention_mask], dim=1)
        stopwords = [sw.to(input_ids.device) for sw in self.stopwords]

        self.kvbuffer1 = torch.zeros((
            self.num_layers * 2,
            self.bsz,
            self.heads,
            self.seqlen + max_iterations + 1,
            self.hidden
        ), dtype=torch.float16, device='cuda').contiguous()
        self.kvbuffer2 = torch.zeros((
            self.num_layers * 2,
            self.bsz,
            self.heads,
            self.seqlen + max_iterations + 1,
            self.hidden
        ), dtype=torch.float16, device='cuda').contiguous()

        queue_for_stop_word = torch.empty(size=(self.bsz, 3), device=input_ids.device, dtype=input_ids.dtype)
        start_time = time.time()
        all_set = torch.tensor([False] * self.bsz, device=input_ids.device)
        for i in range(int(max_iterations)):
            logits = self.infer_(input_ids if i == 0 else new_generated_id, attention_mask)
            if i == 0:
                logits = logits.gather(1, last_token_indices.view(self.bsz, 1, 1).repeat(1, 1, self.vocab_size)).squeeze(1)
            else:
                logits = logits[:, -1, :]

            logits = logits / temperature

            if repetition_penalty > 1:
                score = logits.gather(1, input_ids)
                # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, input_ids, score)

            filtered_logits = self.top_k_top_p_filtering(logits, top_k, top_p)
            probabilities = torch.softmax(filtered_logits, dim=-1)

            cur_len = input_ids.size(1)
            if cur_len > int(regulation_start):
                for i in [1279, 68, 12162, 29]:
                    probabilities[:, i] = probabilities[:, i] * pow(length_penalty, cur_len - regulation_start)
            # print(probabilities)

            new_generated_id = torch.multinomial(probabilities, 1)
            # print(new_generated_id)
            input_ids = torch.cat([input_ids, new_generated_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((self.bsz, 1), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)

            # stop words components
            queue_for_stop_word = torch.cat([queue_for_stop_word[:, 1:], new_generated_id], dim=1)
            set_in_current_iteration = torch.tensor([False] * self.bsz, device=input_ids.device)
            for sw in stopwords:
                set_in_current_iteration |= (queue_for_stop_word == sw).all(1)
            all_set |= set_in_current_iteration
            if all_set.all().item():
                break
            elif time.time() - start_time > max_time:
                break

        return input_ids.cpu()


    def infer_(self, input_ids, attention_mask, device_id=0):
        outputs_logits = torch.empty((self.bsz, self.seqlen, self.vocab_size), dtype=torch.float32, device='cuda')
        io_binding = self.ort_session.io_binding()

        assert input_ids.is_contiguous() and input_ids.dtype == torch.int64 and input_ids.size(1) == self.seqlen
        assert attention_mask.is_contiguous() and attention_mask.dtype == torch.int64 and attention_mask.size(1) == self.seqlen + self.past_seqlen

        io_binding.bind_input(name='input_ids', device_type='cuda', device_id=device_id, element_type=np.int64,
                                          shape=input_ids.shape, buffer_ptr=input_ids.data_ptr())
        io_binding.bind_input(name='attention_mask', device_type='cuda', device_id=device_id, element_type=np.int64,
                              shape=attention_mask.shape, buffer_ptr=attention_mask.data_ptr())

        for _ in range(self.num_layers):
            io_binding.bind_input(name=f'past_key_values.{_}.key', device_type='cuda', device_id=device_id, element_type=torch.float16,
                                  shape=(self.bsz, self.heads, self.past_seqlen, self.hidden), buffer_ptr=self.kvbuffer1[2 * _].data_ptr())
            io_binding.bind_input(name=f'past_key_values.{_}.value', device_type='cuda', device_id=device_id, element_type=torch.float16,
                                  shape=(self.bsz, self.heads, self.past_seqlen, self.hidden), buffer_ptr=self.kvbuffer1[2 * _ + 1].data_ptr())

        io_binding.bind_output('logits', device_type='cuda', device_id=0, element_type=np.float32,
                                           shape=outputs_logits.shape, buffer_ptr=outputs_logits.data_ptr())

        for _ in range(self.num_layers):
            io_binding.bind_output(name=f'present.{_}.key', device_type='cuda', device_id=device_id, element_type=torch.float16,
                                  shape=(self.bsz, self.heads, self.past_seqlen + self.seqlen, self.hidden), buffer_ptr=self.kvbuffer2[2 * _].data_ptr())
            io_binding.bind_output(name=f'present.{_}.value', device_type='cuda', device_id=device_id, element_type=torch.float16,
                                  shape=(self.bsz, self.heads, self.past_seqlen + self.seqlen, self.hidden), buffer_ptr=self.kvbuffer2[2 * _ + 1].data_ptr())

        self.ort_session.run_with_iobinding(io_binding)
        self.kvbuffer1, self.kvbuffer2 = self.kvbuffer2, self.kvbuffer1
        self.past_seqlen += self.seqlen
        self.seqlen = 1

        return outputs_logits

