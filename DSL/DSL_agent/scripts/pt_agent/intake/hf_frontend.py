"""Read local Hugging Face models without remote code or unrestricted pickle.

Third-party imports stay inside the intake entry point so importing pt_agent in
the main environment does not import transformers.
"""

from dataclasses import dataclass
from pathlib import Path


LOCAL_ONLY = dict(trust_remote_code=False, local_files_only=True)


@dataclass
class LocalHF:
    source: str
    config: object
    module: object
    processor: object
    input_kind: str


def load_local(model_path, *, attn_implementation=None) -> LocalHF:
    """Load a reviewed local model and its repository processor/tokenizer.

    A .bin checkpoint is read by torch's restricted weights-only loader and
    passed to the standard model loader as an already decoded state_dict. The
    standard loader handles buffers saved by older transformers versions; the
    checkpoint deserialization itself stays in the weights-only call below.
    """
    import torch
    import transformers

    path = Path(model_path).resolve()
    if not (path / 'config.json').is_file():
        raise ValueError(f'Local HF directory lacks config.json: {path}')
    config = transformers.AutoConfig.from_pretrained(str(path), **LOCAL_ONLY)
    attention = ({'attn_implementation': attn_implementation}
                 if attn_implementation is not None else {})
    architectures = tuple(config.architectures or ())
    if any(name.endswith('ForImageClassification') for name in architectures):
        factory = transformers.AutoModelForImageClassification
        processor_factory = transformers.AutoImageProcessor
        input_kind = 'image'
    elif any(name.endswith('ForSequenceClassification') for name in architectures):
        factory = transformers.AutoModelForSequenceClassification
        processor_factory = transformers.AutoTokenizer
        input_kind = 'tokens'
    elif any(name.endswith('ForCausalLM') for name in architectures):
        factory = transformers.AutoModelForCausalLM
        processor_factory = transformers.AutoTokenizer
        input_kind = 'tokens'
    else:
        factory = transformers.AutoModel
        processor_factory = transformers.AutoTokenizer
        input_kind = 'tokens'

    if ((path / 'model.safetensors').is_file()
            or (path / 'model.safetensors.index.json').is_file()):
        module = factory.from_pretrained(str(path), config=config,
                                         use_safetensors=True, **attention, **LOCAL_ONLY)
    else:
        binary = path / 'pytorch_model.bin'
        if not binary.is_file():
            raise ValueError(f'Local HF directory lacks supported weights: {path}')
        state = torch.load(binary, map_location='cpu', weights_only=True)
        if (not isinstance(state, dict) or not state
                or not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items())):
            raise ValueError('HF .bin must contain only a tensor state_dict')
        # With state_dict supplied, pinned transformers skips checkpoint reads
        # and handles saved nonpersistent buffers using its ordinary load rules.
        module = factory.from_pretrained(str(path), config=config, state_dict=state,
                                         use_safetensors=False, **attention, **LOCAL_ONLY)
    module = module.double().eval()
    try:
        processor = processor_factory.from_pretrained(str(path), **LOCAL_ONLY)
    except Exception as exc:
        raise ValueError(f'Repository processor/tokenizer could not load: {type(exc).__name__}: {exc}') from exc
    return LocalHF(str(path), config, module, processor, input_kind)


def resolve_local(source):
    path = Path(source).expanduser()
    if (path / 'config.json').is_file():
        return path.resolve()
    from huggingface_hub import snapshot_download
    # Hub names resolve only to an existing local cache, never to a new fetch.
    return Path(snapshot_download(str(source), local_files_only=True))


def boundary_module(local):
    """Expose original logits after the word-embedding client boundary."""
    import torch

    class ImageHead(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.original = local.module

        def forward(self, pixel_values):
            return self.original(pixel_values=pixel_values, return_dict=False)[0]

    class TextHead(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.original = local.module

        def forward(self, inputs_embeds, attention_mask):
            options = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=False)
            if any(name.endswith('ForCausalLM') for name in local.config.architectures or ()):
                options['use_cache'] = False
            return self.original(**options)[0]

    return (ImageHead() if local.input_kind == 'image' else TextHead()).double().eval()


@dataclass
class HFInputs:
    local: LocalHF
    raw: object
    info: object

    def get(self, split, n=None):
        indices = self.info.split[split]
        return self.select(indices if n is None else indices[:n])

    def select(self, indices):
        import numpy as np
        import torch
        from ..data.samples import encode_texts
        items, labels = self.raw.select(indices)
        if self.local.input_kind == 'image':
            from PIL import Image
            images = []
            for path in items:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            encoded = self.local.processor(images=images, return_tensors='np')
            arrays = {'pixel_values': np.asarray(encoded['pixel_values'], dtype=np.float64)}
        else:
            encoded = encode_texts(self.local.processor, items, self.info.sequence, task=self.info.task)
            ids = torch.as_tensor(np.array(encoded['input_ids']), dtype=torch.long)
            with torch.no_grad():
                embeddings = self.local.module.get_input_embeddings()(ids).cpu().numpy()
            arrays = dict(inputs_embeds=embeddings, attention_mask=np.asarray(encoded['attention_mask'], dtype=np.float64))
            if 'sel' in getattr(getattr(self.info,'graph',None),'inputs',()):
                mask=arrays['attention_mask'];sel=np.zeros_like(mask)
                sel[np.arange(len(mask)),mask.sum(axis=1).astype(int)-1]=1
                arrays['sel']=sel
        return arrays, labels


def import_local(source, samples):
    import torch
    from ..data.samples import choose_sequence
    from . import import_module
    from .probe import ImportedModel, model_info, rename_inputs
    from ..models.info import infer_task
    from types import SimpleNamespace

    local = load_local(resolve_local(source), attn_implementation='eager')
    split = samples.split()
    sequence = None
    if local.input_kind == 'tokens':
        if samples.kind != 'text':
            raise ValueError('A repository tokenizer requires original text samples')
        decision_indices = split['calibration'] + split['heldout']
        texts, _ = samples.select(decision_indices)
        tokenized = local.processor(texts, truncation=False, padding=False)
        sequence = choose_sequence([len(ids) for ids in tokenized['input_ids']],
                                   local.config.max_position_embeddings)
    elif samples.kind != 'image':
        raise ValueError('A repository image processor requires original image samples')
    task, _ = infer_task((), architectures=local.config.architectures or ())
    temporary = SimpleNamespace(sequence=sequence, task=task, split=split)
    prepared = HFInputs(local, samples, temporary)
    arrays, _ = prepared.select(split['calibration'][:1])
    wrapper = boundary_module(local)
    graph = import_module(wrapper, tuple(torch.from_numpy(value) for value in arrays.values()),
                          str(local.config.model_type).replace('-', '_'), frontend='aten')
    rename_inputs(graph, list(arrays))
    from .transformer import normalize
    causal=task=='causal_lm'
    graph=normalize(graph,last_position=causal,strict=False) if local.input_kind=='tokens' else graph
    handling = dict(mode='repository processor/tokenizer', kind=local.input_kind,
                    processor=type(local.processor).__name__,
                    calls=dict(return_tensors='np', **(dict(padding='max_length', truncation=True,
                               max_length=max(sequence['buckets']), return_attention_mask=True) if sequence else {})))
    if sequence:
        handling.update(embedding_boundary='client word embedding lookup',
                        embedding_class=type(local.module.get_input_embeddings()).__name__)
        if causal and 'sel' in graph.inputs:
            handling.update(selection_boundary='client encrypted one-hot at last real token',
                            selection_input='sel',single_lm_head=True)
    config_fields = ('architectures', 'model_type', 'num_hidden_layers', 'hidden_size', 'vocab_size', 'num_labels', 'id2label')
    info = model_info(dict(type='hf', path=str(source), local_path=local.source,
                          config={key: getattr(local.config, key, None) for key in config_fields}), graph,
                      split=split, has_labels=samples.labels is not None, handling=handling,
                      architectures=local.config.architectures or (), vocab_size=getattr(local.config, 'vocab_size', None),
                      sequence=sequence, assumptions=([dict(sequence=sequence)] if sequence else []))
    prepared.info = info
    return ImportedModel(info, prepared, local)
