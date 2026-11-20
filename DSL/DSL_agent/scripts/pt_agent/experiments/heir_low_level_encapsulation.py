"""Prepare an explicit client-only low-level DenseToSparse construction.

The retained HEIR/Lattigo library remains untouched. This helper only changes
the separately owned lifetime client; it performs no parameter/key generation.
"""
import re


def adapt_client(text):
    old='phase("dense_to_sparse",func(){denseToSparse=kg.GenEvaluationKeyNew(extended,sparseSK)})'
    new='phase("dense_to_sparse",func(){denseToSparse=smallKG.GenEvaluationKeyNew(extended,sparseSK)})'
    if text.count(old)!=1 or new in text:
        raise ValueError('Exactly one retained original dense-to-sparse client construction is required')
    required=(
        'Q:p.Q()[:1],P:p.P()[:1]',
        'smallKG:=rlwe.NewKeyGenerator(sparse)',
        'sparseSK=smallKG.GenSecretKeyWithHammingWeightNew(b.EphemeralSecretWeight)',
        'sparseToDense=kg.GenEvaluationKeyNew(sparseSK,extended)',
        'rlk=kg.GenRelinearizationKeyNew(extended)',
    )
    if any(fragment not in text for fragment in required):
        raise ValueError('Retained same-secret/sparse-context/reverse-key client source differs')
    result=text.replace(old,new,1)
    result=result.replace('// Both calls deliberately use the original 6.1.0 dense key generator.',
        '// Explicit client-only variant: DenseToSparse uses Q0/P0; reverse key and RLK keep the full native context.')
    result=result.replace('same native RLK/encapsulation arithmetic, capacity-one on-demand Galois storage',
        'Explicit client-only Q0/P0 DenseToSparse; original full-context reverse key and RLK; capacity-one on-demand Galois storage')
    marker='result["actual_key_levels"]=map[string]any{'
    if result.count(marker)!=1:raise ValueError('Actual public key level recording must remain present exactly once')
    metadata=('result["encapsulation_construction_variant"]="client_low_level_dense_to_sparse_Q0P0; full_native_sparse_to_dense"\n'
              ' result["original_library_key_generator_unchanged"]=true\n'
              ' result["security_acceptance"]=nil\n '
              +marker)
    result=result.replace(marker,metadata,1)
    if len(re.findall(r'denseToSparse\s*=\s*smallKG\.GenEvaluationKeyNew\(extended,sparseSK\)',result))!=1:
        raise ValueError('Explicit low-level outgoing construction is ambiguous')
    return result
