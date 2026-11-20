// Copy beside the N16 R31 main.go and its public-parameter/checkpoint tests in
// a NEW isolated module. Constructors expose public Q/P and metadata only.
// This test never generates a key, encodes a plaintext or evaluates a cipher.
package main

import (
	"math"
	"reflect"
	"testing"
)

func TestNativeC2S52PreservesDepthResidualAndP61Headroom(t *testing.T) {
	original := c2sPrecisionBits
	t.Cleanup(func() { c2sPrecisionBits = original })
	c2sPrecisionBits = 0
	legacy, legacyBoot := nativeParameterFixture(t, 61)
	c2sPrecisionBits = 52
	p, bp := nativeParameterFixture(t, 61)
	bigp, oldp := bp.BootstrappingParameters, legacyBoot.BootstrappingParameters
	if p.N() != 65536 || p.QCount() != 9 || p.PCount() != 3 || bigp.QCount() != 28 || bigp.PCount() != 4 {
		t.Fatal("C2S52 changed N or public chain dimensions")
	}
	if !reflect.DeepEqual(legacy.Q(), p.Q()) || !reflect.DeepEqual(legacy.P(), p.P()) ||
		!reflect.DeepEqual(oldp.Q()[:24], bigp.Q()[:24]) || !reflect.DeepEqual(oldp.P(), bigp.P()) {
		t.Fatal("C2S52 changed residual primes, earlier bootstrap Q or P61")
	}
	if !reflect.DeepEqual(bp.CoeffsToSlotsParameters.Levels, []int{1, 1, 1, 1}) ||
		!reflect.DeepEqual(bp.SlotsToCoeffsParameters.Levels, []int{1, 1, 1}) ||
		bp.CoeffsToSlotsParameters.LevelQ != legacyBoot.CoeffsToSlotsParameters.LevelQ ||
		bp.SlotsToCoeffsParameters.LevelQ != legacyBoot.SlotsToCoeffsParameters.LevelQ {
		t.Fatal("C2S52 changed the original four/three transform depths or levels")
	}
	for _, prime := range bigp.Q()[24:] {
		if math.Abs(math.Log2(float64(prime))-52) > 1e-4 { t.Fatal("C2S52 did not replace exactly the four nominal56-bit tail primes") }
	}
	actual, previous := publicPrimeLog2(bigp.Q(), bigp.P()), publicPrimeLog2(oldp.Q(), oldp.P())
	if actual > 1747 || math.Abs(actual-1744) > 1e-3 || math.Abs(previous-actual-16) > 1e-3 || math.Abs(actual-bigp.LogQP()) > 1e-9 {
		t.Fatalf("unexpected complete public QP for C2S52/P61: %.15f (previous %.15f)", actual, previous)
	}
	short := publicPrimeLog2(bigp.Q()[:1], bigp.P()[:1])
	if math.Abs(short-116) > 1e-6 || short > 121 {
		t.Fatalf("unexpected H32 generator public Q0P0 scope: %.15f", short)
	}
	policy := record["bootstrap_precision_policy"].(map[string]any)
	if policy["explicit_c2s_override"] != true || policy["source_default_c2s_bits"] != 56 ||
		!reflect.DeepEqual(policy["coeffs_to_slots_log_scales"], [][]int{{52}, {52}, {52}, {52}}) ||
		!reflect.DeepEqual(policy["slots_to_coeffs_log_scales"], [][]int{{39}, {39}, {39}}) {
		t.Fatal("C2S and S2C precision attribution differ")
	}
	if bp.Mod1ParametersLiteral.Mod1InvDegree != 15 || bp.Mod1ParametersLiteral.LogMessageRatio != 2 ||
		bp.Mod1ParametersLiteral.Mod1Degree != 30 || bp.EphemeralSecretWeight != 32 || record["security_acceptance"] != nil {
		t.Fatal("C2S52 changed the fixed approximation, sparse sampler policy or security scope")
	}
}
