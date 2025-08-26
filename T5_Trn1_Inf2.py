import torch
import torch_xla.core.xla_model as xm
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention, T5Stack


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder,
                 decoder,
                 model_config,
                 batch_size,
                 max_length,
                 device,
                 num_beams,
                 tp_degree=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_config = model_config
        self.device = device
        self.num_beams = num_beams
        self.num_attention_heads_per_partition = model_config.num_heads
        self.tp_degree = tp_degree

    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=False, output_hidden_state=False)
        last_hidden_state = encoder_output["last_hidden_state"]
        encoder_hidden_states = torch.concat(
            [tensor.unsqueeze(0) for tensor in last_hidden_state])

        decoder_blocks = self.decoder.block
        present_key_value_states_sa = []
        present_key_value_states_ca = []
        for i, block in enumerate(decoder_blocks):
            cross_attention: T5LayerCrossAttention = block.layer[1]
            attention = cross_attention.EncDecAttention

            def shape(states):
                return states.view(self.batch_size, -1, self.num_attention_heads_per_partition, attention.key_value_proj_dim).transpose(1, 2)

            key_states = shape(attention.k(encoder_hidden_states))
            value_states = shape(attention.v(encoder_hidden_states))

            # cross_attn_kv_state
            present_key_value_states_ca.append(key_states)
            present_key_value_states_ca.append(value_states)

            present_key_value_states_sa.append(torch.zeros((self.batch_size,                                                     # key states
                                                            self.model_config.num_heads,
                                                            self.max_length-1,
                                                            self.model_config.d_kv), dtype=torch.float32, device=self.device))
            present_key_value_states_sa.append(torch.zeros((self.batch_size,                                                     # value states
                                                            self.model_config.num_heads,
                                                            self.max_length-1,
                                                            self.model_config.d_kv), dtype=torch.float32, device=self.device))

        return present_key_value_states_sa + present_key_value_states_ca


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: T5Stack, lm_head: torch.nn.Linear,
                 model_config,
                 num_beams: int,
                 max_length: int,
                 device: str,
                 tp_degree=None):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.model_dim = model_config.d_model
        self.device = device
        self.num_beams = num_beams
        self.batch_size = 1
        self.config = model_config

        num_heads = model_config.num_heads
        num_decoder_layers = model_config.num_decoder_layers

        self.num_attention_heads_per_partition = num_heads
        if device == "cpu":
            self.past_key_values_sa = [torch.ones(
                (num_beams, num_heads, max_length-1, model_config.d_kv), dtype=torch.float32) for _ in range(num_decoder_layers * 2)]
            self.past_key_values_ca = [torch.ones(
                (num_beams, num_heads, max_length, model_config.d_kv), dtype=torch.float32) for _ in range(num_decoder_layers * 2)]
        elif device == "xla":
            self.past_key_values_sa = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(
                (num_beams, self.num_attention_heads_per_partition, max_length-1, model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(num_decoder_layers * 2)])
            self.past_key_values_ca = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(
                (num_beams, self.num_attention_heads_per_partition, max_length, model_config.d_kv), dtype=torch.float32), requires_grad=False) for _ in range(num_decoder_layers * 2)])

    def update_past(self, past_key_values):
        new_past_sa = []
        new_past_ca = []
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i in range(len(new_past_layer[:2])):
                new_past_layer[i] = past_layer[i][:, :, 1:]
            new_past_sa += [new_past_layer[:2],]
            new_past_ca += [new_past_layer[2:],]
        return new_past_sa, new_past_ca

    def reorder_cache(self, past_key_values, beam_idx):
        for i in range(len(past_key_values)):
            gather_index = beam_idx.view(
                [beam_idx.shape[0], 1, 1, 1]).expand_as(past_key_values[i])
            past_key_values[i] = torch.gather(
                past_key_values[i], dim=0, index=gather_index)
        return past_key_values

    def forward(self,
                input_ids,
                decoder_attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                beam_idx,
                beam_scores,
                **kwargs):
        if self.num_beams > 1:
            # We reorder the cache based on the beams selected in each iteration. Required step for beam search.
            past_key_values_sa = self.reorder_cache(
                self.past_key_values_sa, beam_idx)
            past_key_values_ca = self.reorder_cache(
                self.past_key_values_ca, beam_idx)
        else:
            # We do not need to reorder for greedy sampling
            past_key_values_sa = self.past_key_values_sa
            past_key_values_ca = self.past_key_values_ca

        past_key_values = [[*past_key_values_sa[i*2:i*2+2], *past_key_values_ca[i*2:i*2+2]]
                           for i in range(0, int(len(past_key_values_ca)/2))]

        decoder_output = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False)
        last_hidden_state = decoder_output['last_hidden_state']
        past_key_values = decoder_output['past_key_values']

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            last_hidden_state = last_hidden_state * (self.model_dim**-0.5)

        lm_logits = self.lm_head(last_hidden_state)

        past_key_values_sa, past_key_values_ca = self.update_past(
            past_key_values)

        # We flatten the cache to a single array. This is required for the input output aliasing to work
        past_key_values_sa = [
            vec for kv_per_layer in past_key_values_sa for vec in kv_per_layer]
        past_key_values_ca = [
            vec for kv_per_layer in past_key_values_ca for vec in kv_per_layer]

        if self.device == "cpu":
            self.past_key_values_sa = past_key_values_sa
            self.past_key_values_ca = past_key_values_ca

        # We calculate topk inside the wrapper
        next_token_logits = lm_logits[:, -1, :]

        if self.num_beams > 1:
            # This section of beam search is run outside the decoder in the huggingface t5 implementation.
            # To maximize the computation within the neuron device, we move this within the wrapper
            logit_max, _ = torch.max(next_token_logits, dim=-1, keepdim=True)
            logsumexp = torch.log(
                torch.exp(next_token_logits - logit_max).sum(dim=-1, keepdim=True))
            next_token_scores = next_token_logits - logit_max - logsumexp
            next_token_scores = next_token_scores + \
                beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                self.batch_size, self.num_beams * vocab_size)
            next_token_scores = next_token_scores * 1

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(
                next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            return [next_token_scores, next_tokens, next_indices] + past_key_values_sa + past_key_values_ca
        else:
            # Greedy
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            return [next_tokens] + past_key_values_sa + past_key_values_ca
