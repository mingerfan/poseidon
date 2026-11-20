// Copy this file and r31_lattigo_checkpoint_test.go beside the trusted main.go
// in a NEW isolated module. These tests construct public parameters and rings
// only. They never create a key generator, secret, encoder, evaluator or
// ciphertext. Do not use t.Parallel: the trusted runner has package globals.
package main

import (
	"math"
	"reflect"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func nativeParameterFixture(t *testing.T, auxiliaryBits int) (ckks.Parameters, bootstrapping.Parameters) {
	t.Helper()
	checkpointFixture(t, 1)
	originalPrecision, originalS2C := precisionBits, s2cPrecisionBits
	originalInverse, originalDegree := mod1InverseDegree, mod1PolynomialDegree
	originalMessageRatio, originalAuxiliary := logMessageRatio, auxiliaryPrimeBits
	originalCache, originalDivisor := bootstrapCacheCapacity, bootstrapInputDivisor
	t.Cleanup(func() {
		precisionBits, s2cPrecisionBits = originalPrecision, originalS2C
		mod1InverseDegree, mod1PolynomialDegree = originalInverse, originalDegree
		logMessageRatio, auxiliaryPrimeBits = originalMessageRatio, originalAuxiliary
		bootstrapCacheCapacity, bootstrapInputDivisor = originalCache, originalDivisor
	})
	precisionBits, s2cPrecisionBits = 50, 0
	mod1InverseDegree, mod1PolynomialDegree = 15, 0
	logMessageRatio, auxiliaryPrimeBits = 2, auxiliaryBits
	bootstrapCacheCapacity, bootstrapInputDivisor = 32, 8
	for _, counter := range []string{
		"keygen_api_attempts", "keygen_api_completed", "plaintext_encoding_attempts",
		"plaintext_encoding_completed", "encryption_attempts", "encryption_completed",
		"compute_attempts", "compute_completed", "bootstrap_attempts", "bootstrap_completed",
		"final_decryption_attempts", "final_decryption_completed",
	} {
		record[counter] = 0
	}
	p, bp, err := parameters()
	if err != nil {
		t.Fatal(err)
	}
	for _, counter := range []string{
		"keygen_api_attempts", "keygen_api_completed", "plaintext_encoding_attempts",
		"plaintext_encoding_completed", "encryption_attempts", "encryption_completed",
		"compute_attempts", "compute_completed", "bootstrap_attempts", "bootstrap_completed",
		"final_decryption_attempts", "final_decryption_completed",
	} {
		if record[counter] != 0 {
			t.Fatalf("public parameter construction invoked %s: %v", counter, record[counter])
		}
	}
	if record["security_acceptance"] != nil {
		t.Fatal("public parameter construction asserted security acceptance")
	}
	return p, bp
}

// Sum the actual public prime logarithms with compensation. Nominal LogP=57
// primes can lie on either side of 2^57; integer bit_length is not their nominal
// size and is not the complete log2(QP) used by the published table.
func publicPrimeLog2(primeGroups ...[]uint64) float64 {
	var sum, compensation float64
	for _, primes := range primeGroups {
		for _, prime := range primes {
			term := math.Log2(float64(prime)) - compensation
			next := sum + term
			compensation = (next - sum) - term
			sum = next
		}
	}
	return sum
}

func requirePublicPrimeSubset(t *testing.T, name string, subset, full []uint64) {
	t.Helper()
	available := make(map[uint64]bool, len(full))
	for _, prime := range full {
		available[prime] = true
	}
	for _, prime := range subset {
		if !available[prime] {
			t.Fatalf("%s prime %d is absent from the full bootstrap basis", name, prime)
		}
	}
}

func TestNativeParameterLegacyP61PublicRecord(t *testing.T) {
	if auxiliaryPrimeBits != 61 {
		t.Fatalf("legacy package default changed: %d", auxiliaryPrimeBits)
	}
	p, bp := nativeParameterFixture(t, 61)
	bigp := bp.BootstrappingParameters
	if p.QCount() != 9 || p.PCount() != 3 || bigp.QCount() != 28 || bigp.PCount() != 4 {
		t.Fatal("legacy precision50/Inv15/MR2 public chain shape changed")
	}
	if !reflect.DeepEqual(p.LogPi(), []int{61, 61, 61}) || !reflect.DeepEqual(bigp.LogPi(), []int{61, 61, 61, 61}) {
		t.Fatal("legacy auxiliary prime sizes changed")
	}
	actual := publicPrimeLog2(bigp.Q(), bigp.P())
	if math.Abs(actual-1759.9999852481214) > 1e-8 || actual <= 1747 {
		t.Fatalf("legacy complete QP differs from the saved old90 public record: %.15f", actual)
	}
	if math.Abs(actual-bigp.LogQP()) > 1e-9 {
		t.Fatal("legacy LogQP omits or disagrees with actual Q/P primes")
	}
}

func TestNativeDualP57PreservesQAndAuxiliarySubgroup(t *testing.T) {
	legacy, legacyBoot := nativeParameterFixture(t, 61)
	p, bp := nativeParameterFixture(t, 57)
	bigp := bp.BootstrappingParameters
	if !reflect.DeepEqual(legacy.Q(), p.Q()) || !reflect.DeepEqual(legacyBoot.BootstrappingParameters.Q(), bigp.Q()) {
		t.Fatal("dual P57 changed the residual or bootstrap Q primes")
	}
	if p.PCount() != 3 || bigp.PCount() != 4 {
		t.Fatal("dual P57 changed auxiliary prime counts")
	}
	if !reflect.DeepEqual(p.LogPi(), []int{57, 57, 57}) || !reflect.DeepEqual(bigp.LogPi(), []int{57, 57, 57, 57}) {
		t.Fatal("dual P57 did not select both nominal57 auxiliary groups")
	}
	for _, primes := range [][]uint64{p.P(), bigp.P()} {
		for _, prime := range primes {
			if math.Abs(math.Log2(float64(prime))-57) > 1e-6 {
				t.Fatalf("unexpected nominal57 prime: %d", prime)
			}
		}
	}
	requirePublicPrimeSubset(t, "residual P", p.P(), bigp.P())
	if reflect.DeepEqual(legacy.P(), p.P()) || reflect.DeepEqual(legacyBoot.BootstrappingParameters.P(), bigp.P()) {
		t.Fatal("explicit P57 retained a legacy P group")
	}
}

func TestNativeDualP57CompleteMainModulusBound(t *testing.T) {
	p, bp := nativeParameterFixture(t, 57)
	bigp := bp.BootstrappingParameters
	if p.N() != 65536 || bigp.N() != p.N() {
		t.Fatal("same-N main-key extension premise changed")
	}
	if !reflect.DeepEqual(p.Q(), bigp.Q()[:p.QCount()]) {
		t.Fatal("residual Q is not a bootstrap prefix")
	}
	requirePublicPrimeSubset(t, "same-main-secret residual P", p.P(), bigp.P())
	actual := publicPrimeLog2(bigp.Q(), bigp.P())
	if math.IsNaN(actual) || math.IsInf(actual, 0) || actual > 1747 {
		t.Fatalf("complete dense main-key QP exceeds the published N65536 table entry: %.15f", actual)
	}
	if math.Abs(actual-1743.9999852481253) > 1e-6 || math.Abs(actual-bigp.LogQP()) > 1e-9 {
		t.Fatalf("complete P57 QP disagrees with actual-prime calculation: %.15f", actual)
	}
	if residual := publicPrimeLog2(p.Q(), p.P()); math.Abs(residual-626.0000000386499) > 1e-6 || residual > actual {
		t.Fatalf("unexpected residual P57 complete QP: %.15f", residual)
	}
	// This is a dense-main-secret published-table screen. H32 is assessed
	// separately; no security estimate or full-system certificate is produced.
}

func TestNativeDualP57EphemeralPrefixPublicModulus(t *testing.T) {
	_, bp := nativeParameterFixture(t, 57)
	bigp := bp.BootstrappingParameters
	short, err := rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{
		LogN: bigp.LogN(), Q: bigp.Q()[:1], P: bigp.P()[:1],
	})
	if err != nil {
		t.Fatal(err)
	}
	if short.N() != 65536 || short.QCount() != 1 || short.PCount() != 1 || short.Q()[0] != bigp.Q()[0] || short.P()[0] != bigp.P()[0] {
		t.Fatal("the wrapper's sparse-generator parameter prefix changed")
	}
	actual := publicPrimeLog2(short.Q(), short.P())
	if math.Abs(actual-112) > 1e-6 || actual > 121 || math.Abs(actual-short.LogQP()) > 1e-9 {
		t.Fatalf("H32 wrapper generator does not use the expected short Q0P0 modulus: %.15f", actual)
	}
	if bp.EphemeralSecretWeight != 32 {
		t.Fatal("explicit GenSecretKeyWithHammingWeightNew override would no longer request H32")
	}
	// No H32 key is sampled. The <=121 assertion checks the source's intended
	// low-modulus encapsulation range, not a sparse-secret security certificate.
}

func TestNativeParameterSamplerMetadataAndPolicyBothGroups(t *testing.T) {
	for _, auxiliaryBits := range []int{61, 57} {
		name := "P61"
		if auxiliaryBits == 57 {
			name = "P57"
		}
		t.Run(name, func(t *testing.T) {
			p, bp := nativeParameterFixture(t, auxiliaryBits)
			dense, ok := p.Xs().(ring.Ternary)
			if !ok || dense.P != 2.0/3.0 || dense.H != 0 {
				t.Fatalf("residual main-secret sampler metadata changed: %#v", p.Xs())
			}
			configured, ok := bp.BootstrappingParameters.Xs().(ring.Ternary)
			if !ok || configured.H != 192 || configured.P != 0 {
				t.Fatalf("configured bootstrap distribution changed: %#v", bp.BootstrappingParameters.Xs())
			}
			for _, distribution := range []ring.DistributionParameters{p.Xe(), bp.BootstrappingParameters.Xe()} {
				noise, ok := distribution.(ring.DiscreteGaussian)
				if !ok || noise.Sigma != 3.2 || noise.Bound != 19.2 {
					t.Fatalf("actual library error metadata changed: %#v", distribution)
				}
			}
			if p.LogDefaultScale() != 50 || bp.Mod1ParametersLiteral.Mod1InvDegree != 15 || bp.Mod1ParametersLiteral.Mod1Degree != 30 || bp.Mod1ParametersLiteral.LogMessageRatio != 2 || bp.EphemeralSecretWeight != 32 {
				t.Fatal("precision50/Inv15/MR2/H32 policy changed with P selection")
			}
			attribution := record["secret_sampler_attribution"].(map[string]any)
			if attribution["sampled_hamming_weight"] != nil || attribution["bootstrap_configured_Xs"] != bp.BootstrappingParameters.Xs() {
				t.Fatal("parameter metadata claimed a sampled Hamming weight or lost configured H192 attribution")
			}
		})
	}
}
