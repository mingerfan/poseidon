"""Small source-only checks; no import of numeric or cryptographic runtimes."""
import unittest
from pt_agent.experiments.heir_low_level_encapsulation import adapt_client


SOURCE='''func prepare() {
 sparse,e:=rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{LogN:p.LogN(),Q:p.Q()[:1],P:p.P()[:1]})
 smallKG:=rlwe.NewKeyGenerator(sparse)
 phase("ephemeral",func(){sparseSK=smallKG.GenSecretKeyWithHammingWeightNew(b.EphemeralSecretWeight)})
 // Both calls deliberately use the original 6.1.0 dense key generator.
 phase("dense_to_sparse",func(){denseToSparse=kg.GenEvaluationKeyNew(extended,sparseSK)})
 phase("sparse_to_dense",func(){sparseToDense=kg.GenEvaluationKeyNew(sparseSK,extended)})
 phase("relinearization",func(){rlk=kg.GenRelinearizationKeyNew(extended)})
 result["actual_key_levels"]=map[string]any{}
}
'''


class ClientSourceChecks(unittest.TestCase):
    def test_only_forward_key_context_changes_and_actual_levels_stay_recorded(self):
        result=adapt_client(SOURCE)
        self.assertIn('denseToSparse=smallKG.GenEvaluationKeyNew(extended,sparseSK)',result)
        self.assertIn('sparseToDense=kg.GenEvaluationKeyNew(sparseSK,extended)',result)
        self.assertIn('rlk=kg.GenRelinearizationKeyNew(extended)',result)
        self.assertIn('result["actual_key_levels"]',result);self.assertIn('result["security_acceptance"]=nil',result)
        self.assertEqual(result.count('GenSecretKeyWithHammingWeightNew'),1)

    def test_missing_or_duplicated_target_is_rejected(self):
        target='phase("dense_to_sparse",func(){denseToSparse=kg.GenEvaluationKeyNew(extended,sparseSK)})'
        for source in (SOURCE.replace(target,''),SOURCE+target):
            with self.assertRaises(ValueError):adapt_client(source)

    def test_missing_sparse_context_or_changed_reverse_key_is_rejected(self):
        for source in (SOURCE.replace('Q:p.Q()[:1]','Q:p.Q()'),SOURCE.replace('sparseToDense=kg.GenEvaluationKeyNew','sparseToDense=smallKG.GenEvaluationKeyNew')):
            with self.assertRaises(ValueError):adapt_client(source)


if __name__=='__main__':unittest.main()
