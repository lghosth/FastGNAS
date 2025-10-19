from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple
SparseEmbedding = Tuple[int, Sequence[float]]

@dataclass
class SparseSpaceSpec:
    component_embeddings: Dict[str, Dict[str, SparseEmbedding]]
    option_decoder: Dict[int, Tuple[str, str]]
DEFAULT_REUSE_COMPONENTS: Tuple[Tuple[str, str], ...] = (('gnnConv2', 'gnnConv1'), ('activation2', 'activation1'), ('aggregation2', 'aggregation1'), ('normalize2', 'normalize1'))
try:
    from baseline.reference.DGNAP.search_space_manager.search_space import search_space_embeddings
    from baseline.reference.DGNAP.settings.config_file import config as ref_config, set_seed as ref_set_seed
    _HAS_REFERENCE = True
except Exception:
    search_space_embeddings = None
    ref_config = None
    ref_set_seed = None
    _HAS_REFERENCE = False

def _generate_unique_encoding(rng: random.Random, base_code: Sequence[int], existing: Iterable[Sequence[int]], noptioncode: int) -> Sequence[int]:
    option_encoding = list(base_code) + [rng.randint(0, 1) for _ in range(noptioncode)]
    while option_encoding in existing:
        option_encoding = list(base_code) + [rng.randint(0, 1) for _ in range(noptioncode)]
    return option_encoding

def build_sparse_search_space(search_space: Mapping[str, Sequence[str]], *, nfcode: int=4, noptioncode: int=4, seed: int=0, reuse_components: Sequence[Tuple[str, str]]=DEFAULT_REUSE_COMPONENTS, use_reference: bool=False) -> SparseSpaceSpec:
    if use_reference:
        if not _HAS_REFERENCE:
            raise ImportError('Reference DGNAP search space utilities are unavailable')
        if ref_config is not None:
            ref_config['param']['encoding_method'] = 'embedding'
            ref_config['param']['nfcode'] = str(max(nfcode, 1))
            ref_config['param']['noptioncode'] = str(max(noptioncode, 1))
        if ref_set_seed is not None:
            ref_set_seed(seed)
        base_space = {component: list(options) for component, options in search_space.items()}
        embeddings_dict, option_decoder_map = search_space_embeddings(base_space)
        component_embeddings: Dict[str, Dict[str, SparseEmbedding]] = {}
        option_decoder: Dict[int, Tuple[str, str]] = {}
        for component, option_map in embeddings_dict.items():
            comp_dict: Dict[str, SparseEmbedding] = {}
            for choice, (code, encoding) in option_map.items():
                if isinstance(encoding, (int, float)):
                    encoding_list: Sequence[float] = [float(encoding)]
                else:
                    encoding_list = [float(v) for v in encoding]
                comp_dict[choice] = (int(code), encoding_list)
                option_decoder[int(code)] = (component, choice)
            component_embeddings[component] = comp_dict
        return SparseSpaceSpec(component_embeddings=component_embeddings, option_decoder=option_decoder)
    rng = random.Random(seed)
    component_embeddings: Dict[str, Dict[str, SparseEmbedding]] = {}
    option_decoder: Dict[int, Tuple[str, str]] = {}
    function_codes: Dict[str, Sequence[int]] = {}
    component_choice_vectors: Dict[str, Dict[str, Sequence[int]]] = {}
    code_counter = 0
    reuse_map = {dst: src for dst, src in reuse_components}
    for component, choices in search_space.items():
        component_embeddings[component] = {}
        component_choice_vectors[component] = {}
        if component in reuse_map:
            src = reuse_map[component]
            if src not in component_choice_vectors:
                raise ValueError(f"Component '{component}' reuses '{src}', but source embeddings missing")
            for choice in choices:
                if choice not in component_choice_vectors[src]:
                    raise ValueError(f"Choice '{choice}' missing in reused component '{src}'")
                encoding = component_choice_vectors[src][choice]
                component_choice_vectors[component][choice] = encoding
                option_decoder[code_counter] = (component, choice)
                component_embeddings[component][choice] = (code_counter, list(encoding))
                code_counter += 1
            continue
        base_code = [rng.randint(0, 1) for _ in range(nfcode)]
        while base_code in function_codes.values():
            base_code = [rng.randint(0, 1) for _ in range(nfcode)]
        function_codes[component] = base_code
        existing_encodings: Dict[Tuple[int, ...], bool] = {}
        for choice in choices:
            option_encoding = _generate_unique_encoding(rng, base_code, existing_encodings, noptioncode)
            encoding_tuple = tuple(option_encoding)
            existing_encodings[encoding_tuple] = True
            component_choice_vectors[component][choice] = option_encoding
            option_decoder[code_counter] = (component, choice)
            component_embeddings[component][choice] = (code_counter, list(option_encoding))
            code_counter += 1
    return SparseSpaceSpec(component_embeddings=component_embeddings, option_decoder=option_decoder)