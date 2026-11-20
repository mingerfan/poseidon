// Copy only this N17 test and the DaCapo trusted main.go to a NEW isolated
// module. Tests use public parameter constructors, plaintext encoding and small
// metadata/files. No secret/key generator, ciphertext or evaluator is created.
// R31 N16 parameter tests belong to their separate module.
package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func dacapoPublicRecord(t *testing.T) {
	t.Helper()
	originalRecord, originalOutput, originalEvents := record, outputPath, eventPath
	originalSchedule, originalSequence, originalParents := checkpoints, sequence, parents
	originalPrecision, originalS2C := precisionBits, s2cPrecisionBits
	originalInverse, originalDegree := mod1InverseDegree, mod1PolynomialDegree
	originalRatio, originalCache, originalDivisor := logMessageRatio, bootstrapCacheCapacity, bootstrapInputDivisor
	directory := t.TempDir()
	outputPath, eventPath = filepath.Join(directory, "result.json"), filepath.Join(directory, "events.jsonl")
	record = map[string]any{"status": "running", "inline_scalar_broadcasts": 0}
	for _, name := range []string{"keygen_api_attempts", "keygen_api_completed", "plaintext_encoding_attempts", "plaintext_encoding_completed", "encryption_attempts", "encryption_completed", "compute_attempts", "compute_completed", "bootstrap_attempts", "bootstrap_completed", "final_decryption_attempts", "final_decryption_completed"} {
		record[name] = 0
	}
	checkpoints, _ = newCheckpointSchedule(1)
	sequence, parents = 0, nil
	precisionBits, s2cPrecisionBits, mod1InverseDegree, mod1PolynomialDegree = 51, 0, 0, 0
	logMessageRatio, bootstrapCacheCapacity, bootstrapInputDivisor = 0, 1, 0
	t.Cleanup(func() {
		record, outputPath, eventPath = originalRecord, originalOutput, originalEvents
		checkpoints, sequence, parents = originalSchedule, originalSequence, originalParents
		precisionBits, s2cPrecisionBits = originalPrecision, originalS2C
		mod1InverseDegree, mod1PolynomialDegree = originalInverse, originalDegree
		logMessageRatio, bootstrapCacheCapacity, bootstrapInputDivisor = originalRatio, originalCache, originalDivisor
	})
}

func dacapoWriteJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil { t.Fatal(err) }
	if err = os.WriteFile(path, data, 0600); err != nil { t.Fatal(err) }
}

func dacapoSmallPlan(t *testing.T) (Plan, string, string) {
	t.Helper()
	var plan Plan
	err := json.Unmarshal([]byte(`{
		"format_version":1,"target":{"target_id":"dacapo-own-native-public-fixture"},
		"values":[
		{"id":"x","kind":"ciphertext","context":"dacapo-original-n17-r51-export","level":16,"scale_log2":40,"components":2,"ntt":true,"place":{"kind":"host","rank":0}},
		{"id":"scalar","kind":"plaintext","context":"dacapo-original-n17-r51-export","level":16,"scale_log2":87,"components":1,"ntt":true,"place":{"kind":"host","rank":0}},
		{"id":"out","kind":"ciphertext","context":"dacapo-original-n17-r51-export","level":16,"scale_log2":127,"components":2,"ntt":true,"place":{"kind":"host","rank":0}}],
		"external_inputs":["x"],"initialization":[{"kind":"encode","ordinal":0,"output":"scalar","payload":{"kind":"inline","values":[0.125]}}],
		"execution":[{"kind":"compute","ordinal":1,"op":"mul_cp","inputs":["x","scalar"],"output":"out","place":{"kind":"host","rank":0}}],
		"finalization":[],"final_outputs":["out"],"plaintext_bundle":{"id":"public-n17","version":1}}`), &plan)
	if err != nil { t.Fatal(err) }
	directory := t.TempDir()
	bundle := filepath.Join(directory, "bundle")
	if err = os.Mkdir(bundle, 0700); err != nil { t.Fatal(err) }
	path := filepath.Join(directory, "plan.json")
	dacapoWriteJSON(t, path, plan)
	dacapoWriteJSON(t, filepath.Join(bundle, "manifest.json"), Manifest{Format: 1, ID: "public-n17", Version: 1})
	return plan, path, bundle
}

func dacapoProofs(t *testing.T, planPath string) (string, string, map[string]any, map[string]any) {
	t.Helper()
	directory := t.TempDir()
	sites := map[string]any{"status": "scalar_sites_verified", "source_factory": "pt_agent.experiments.r13_common_models:resnet20_silu", "source_plan": planPath, "original_dsl_scalar_broadcast_interpretation": true, "original_plan_modified": false,
		"sites": []map[string]any{{"value_id": "scalar", "interpretation": "full-slot-scalar", "source_path": "/trusted/source.py", "source_line": 72}}}
	reference := map[string]any{"status": "strict_math_passed", "source_factory": "pt_agent.experiments.r13_common_models:resnet20_silu", "source_plan": planPath, "sample_ids": []string{"train:0", "train:1", "train:2"}, "original_poly_preserved": true, "original_pack_bb32_preserved": true, "original_plan_modified": false, "all_three_passed": true, "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12}
	a, b := filepath.Join(directory, "sites.json"), filepath.Join(directory, "math.json")
	dacapoWriteJSON(t, a, sites)
	dacapoWriteJSON(t, b, reference)
	return a, b, sites, reference
}

func TestDaCapoN17PublicParametersPreserveOriginalNativeDefaults(t *testing.T) {
	dacapoPublicRecord(t)
	p, bp, err := parameters()
	if err != nil { t.Fatal(err) }
	if p.N() != 131072 || p.MaxLevel() != 16 || p.QCount() != 17 || p.PCount() != 3 || p.LogDefaultScale() != 51 || logSlots != 16 {
		t.Fatal("original N17 residual contract changed")
	}
	for i, prime := range p.Q() {
		want := 51.0
		if i == 0 { want = 55 }
		if math.Abs(math.Log2(float64(prime))-want) > 1e-4 { t.Fatalf("unexpected residual prime %d", i) }
	}
	if bp.BootstrappingParameters.N() != p.N() || bp.BootstrappingParameters.PCount() != 4 || bp.LogMaxSlots() != 16 {
		t.Fatal("original bootstrap N17/slots16/P4 contract changed")
	}
	for _, prime := range append(append([]uint64{}, p.P()...), bp.BootstrappingParameters.P()...) {
		if math.Abs(math.Log2(float64(prime))-61) > 1e-4 { t.Fatal("original nominal P61 changed") }
	}
	literal := bp.Mod1ParametersLiteral
	if literal.Mod1InvDegree != bootstrapping.DefaultMod1InvDegree || literal.Mod1Degree != bootstrapping.DefaultMod1Degree || literal.LogMessageRatio != bootstrapping.DefaultLogMessageRatio || literal.DoubleAngle != bootstrapping.DefaultDoubleAngle || literal.K != bootstrapping.DefaultK {
		t.Fatal("original native Mod1 defaults changed")
	}
	for _, name := range []string{"keygen_api_attempts", "keygen_api_completed", "plaintext_encoding_attempts", "encryption_attempts", "compute_attempts", "bootstrap_attempts", "final_decryption_attempts"} {
		if record[name] != 0 { t.Fatalf("public constructor executed %s", name) }
	}
	if record["security_acceptance"] != nil { t.Fatal("public parameters asserted security acceptance") }
}

func TestDaCapoLoaderPreservesHighOriginalScales(t *testing.T) {
	_, path, bundle := dacapoSmallPlan(t)
	_, values, encodes, _, last, err := loadPlan(path, bundle)
	if err != nil { t.Fatal(err) }
	if values["scalar"].Scale != 87 || values["out"].Scale != 127 || encodes["scalar"].Values[0] != 0.125 || last["x"] != 0 {
		t.Fatal("loader relabelled an original scale, scalar or last use")
	}
}

func TestDaCapoLoaderRejectsWrongContextLevelAndScale(t *testing.T) {
	for _, kind := range []string{"context", "level", "scale"} {
		t.Run(kind, func(t *testing.T) {
			plan, path, bundle := dacapoSmallPlan(t)
			switch kind {
			case "context": plan.Values[0].Context = "r31-logN16"
			case "level": plan.Values[0].Level = 17
			case "scale": plan.Values[2].Scale = 126
			}
			dacapoWriteJSON(t, path, plan)
			if _, _, _, _, _, err := loadPlan(path, bundle); err == nil { t.Fatal("accepted invalid original metadata") }
		})
	}
}

func TestDaCapoInlinePlaintextCacheBroadcastsEverySlot(t *testing.T) {
	dacapoPublicRecord(t)
	p, _, err := parameters()
	if err != nil { t.Fatal(err) }
	encoder := ckks.NewEncoder(p)
	cache := plaintextCache{p: p, encoder: encoder, values: map[string]Value{
		"a": {Kind: "plaintext", Level: 0, Scale: 51}, "b": {Kind: "plaintext", Level: 0, Scale: 51}},
		encodes: map[string]EncodePayload{"a": {Kind: "inline", Values: []float64{0.125}}, "b": {Kind: "inline", Values: []float64{-0.375}}}}
	for _, item := range []struct { name string; value float64 }{{"a", 0.125}, {"a", 0.125}, {"b", -0.375}} {
		pt, err := cache.get(item.name)
		if err != nil { t.Fatal(err) }
		decoded := make([]float64, 1<<logSlots)
		if err = encoder.Decode(pt, decoded); err != nil { t.Fatal(err) }
		for index, value := range decoded {
			if math.Abs(value-item.value) > 1e-12+1e-12*math.Abs(item.value) { t.Fatalf("scalar broadcast differs at slot%d", index) }
		}
	}
	if cache.loads != 2 || cache.hits != 1 || cache.evictions != 1 || record["inline_scalar_broadcasts"] != 2 {
		t.Fatal("one-plaintext cache accounting differs")
	}
	if record["keygen_api_attempts"] != 0 || record["encryption_attempts"] != 0 || record["final_decryption_attempts"] != 0 { t.Fatal("plaintext test created keys or used ciphertexts") }
}

func TestDaCapoSourceProofsBindBothCurrentPlanPaths(t *testing.T) {
	plan, path, _ := dacapoSmallPlan(t)
	a, b, sites, reference := dacapoProofs(t, path)
	if err := authorizeOriginalSource(plan, path, false, a, b); err != nil { t.Fatal(err) }
	foreign := filepath.Join(t.TempDir(), "foreign-plan.json")
	dacapoWriteJSON(t, foreign, plan)
	for _, item := range []struct { path string; proof map[string]any }{{a, sites}, {b, reference}} {
		item.proof["source_plan"] = foreign
		dacapoWriteJSON(t, item.path, item.proof)
		if err := authorizeOriginalSource(plan, path, false, a, b); err == nil { t.Fatal("accepted a different plan proof with identical IDs") }
		item.proof["source_plan"] = path
		dacapoWriteJSON(t, item.path, item.proof)
	}
}

func TestDaCapoSourcePathAliasesAndInvalidPaths(t *testing.T) {
	_, path, _ := dacapoSmallPlan(t)
	alias := filepath.Join(t.TempDir(), "alias.json")
	if err := os.Symlink(path, alias); err != nil { t.Fatal(err) }
	if !sameSourcePlan(alias, path) || !sameSourcePlan(path, alias) { t.Fatal("canonical path aliases disagree") }
	for _, proof := range []string{"plan.json", "", filepath.Dir(path), filepath.Join(t.TempDir(), "missing.json")} {
		if sameSourcePlan(proof, path) { t.Fatalf("accepted invalid proof path %q", proof) }
	}
}

func TestDaCapoOriginalProofRejectsWrongSamplesAndScalarSites(t *testing.T) {
	plan, path, _ := dacapoSmallPlan(t)
	a, b, sites, reference := dacapoProofs(t, path)
	reference["sample_ids"] = []string{"test:0", "train:1", "train:2"}
	dacapoWriteJSON(t, b, reference)
	if err := authorizeOriginalSource(plan, path, false, a, b); err == nil { t.Fatal("accepted report or heldout evidence") }
	reference["sample_ids"] = []string{"train:0", "train:1", "train:2"}
	dacapoWriteJSON(t, b, reference)
	sites["sites"] = []map[string]any{{"value_id": "foreign", "interpretation": "full-slot-scalar", "source_path": "/trusted/source.py", "source_line": 72}}
	dacapoWriteJSON(t, a, sites)
	if err := authorizeOriginalSource(plan, path, false, a, b); err == nil { t.Fatal("accepted missing original scalar site") }
}

func TestDaCapoPublicScalarFixtureStaysExplicitAndSmall(t *testing.T) {
	plan, _, _ := dacapoSmallPlan(t)
	if err := authorizeOriginalSource(plan, "", true, "", ""); err != nil { t.Fatal(err) }
	plan.Target = json.RawMessage(`{"target_id":"original-model"}`)
	if err := authorizeOriginalSource(plan, "", true, "", ""); err == nil { t.Fatal("fixture override accepted original-model target") }
	plan.Target = json.RawMessage(`{"target_id":"dacapo-own-native-public-fixture"}`)
	plan.Execution = make([]Instruction, 13)
	if err := authorizeOriginalSource(plan, "", true, "", ""); err == nil { t.Fatal("fixture override accepted more than12operations") }
}

func TestDaCapoFinalSelectionUsesPhysical32UnitsWithoutChangingInput(t *testing.T) {
	decoded := make([]float64, 16)
	indices, wanted := make([]int, 10), make([]float64, 10)
	for i := range decoded { decoded[i] = float64(i-5)/32 }
	for i := range indices { indices[i], wanted[i] = i, float64(i-5) }
	raw, scaled, err := selectFinalOutput(decoded, indices, 32)
	if err != nil { t.Fatal(err) }
	for i := range indices {
		if raw[i] != float64(i-5)/32 || scaled[i] != wanted[i] { t.Fatal("final output conversion lost physical32 units") }
	}
	comparison, err := compare(wanted, scaled)
	if err != nil || comparison["passed"] != true || comparison["absolute_tolerance"] != 1e-5 || comparison["relative_tolerance"] != 1e-4 {
		t.Fatal("original CKKS comparison units or tolerance changed")
	}
	raw[0], scaled[1] = 100, 100
	if decoded[0] != -5.0/32 || decoded[1] != -4.0/32 { t.Fatal("final reporting conversion mutated decoded data") }
}

func TestDaCapoFinalSelectionRejectsInvalidFactorsIndicesAndNonfinite(t *testing.T) {
	for _, factor := range []float64{0, 2, -32, math.NaN(), math.Inf(1)} {
		if _, _, err := selectFinalOutput([]float64{1}, []int{0}, factor); err == nil { t.Fatal("accepted invalid output factor") }
	}
	for _, indices := range [][]int{{-1}, {1}} {
		if _, _, err := selectFinalOutput([]float64{1}, indices, 1); err == nil { t.Fatal("accepted invalid final index") }
	}
	for _, value := range []float64{math.NaN(), math.Inf(1), math.MaxFloat64} {
		if _, _, err := selectFinalOutput([]float64{value}, []int{0}, 32); err == nil { t.Fatal("accepted nonfinite selected/scaled result") }
	}
}

func TestDaCapoOriginalExpectedContractRequiresSameSourceAndTimes32(t *testing.T) {
	_, planPath, _ := dacapoSmallPlan(t)
	_, mathPath, _, _ := dacapoProofs(t, planPath)
	factor := 32.0
	reference := Expected{Indices: make([]int, 10), Values: make([]float64, 10), Scope: "Original physical logits from accepted mathematical probe",
		OutputMultiplier: &factor, OutputDecoder: "original-physical32-first10", SourcePlan: planPath, SourceMathProof: mathPath,
		SourceFactory: "pt_agent.experiments.r13_common_models:resnet20_silu", SampleID: "train:0"}
	for i := range reference.Indices { reference.Indices[i] = i }
	if got, err := expectedOutputMultiplier(reference, planPath, mathPath, false); err != nil || got != 32 { t.Fatal("rejected original final decoder contract", err) }
	foreign := filepath.Join(t.TempDir(), "different.json")
	dacapoWriteJSON(t, foreign, map[string]bool{"different": true})
	for _, kind := range []string{"missing", "factor", "scope", "plan", "math", "decoder", "sample", "indices"} {
		bad := reference
		switch kind {
		case "missing": bad.OutputMultiplier = nil
		case "factor": one := 1.0; bad.OutputMultiplier = &one
		case "scope": bad.Scope = ""
		case "plan": bad.SourcePlan = foreign
		case "math": bad.SourceMathProof = foreign
		case "decoder": bad.OutputDecoder = "unscaled"
		case "sample": bad.SampleID = "test:0"
		case "indices": bad.Indices = []int{1}
		}
		if _, err := expectedOutputMultiplier(bad, planPath, mathPath, false); err == nil { t.Fatalf("accepted invalid original contract: %s", kind) }
	}
	if _, err := expectedOutputMultiplier(reference, "", "", true); err == nil { t.Fatal("public fixture accepted multiplier32") }
	if got, err := expectedOutputMultiplier(Expected{}, "", "", true); err != nil || got != 1 { t.Fatal("public fixture default multiplier changed") }
}
