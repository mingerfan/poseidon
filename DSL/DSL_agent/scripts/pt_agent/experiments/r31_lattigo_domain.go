// Public plaintext Encode only. No secret, evaluation key or ciphertext exists.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/big"
	"os"
	"runtime"
	"time"

	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type vectorRecord struct {
	Path    string  `json:"path"`
	Sample  string  `json:"sample"`
	Ordinal int     `json:"ordinal"`
	Mean    float64 `json:"mean"`
	MeanAbs float64 `json:"mean_abs"`
}

// Positive Taylor-tail bounds use native mod1_parameters.go:139-155.
// They cover inverse truncation with an ideal sine. Native cosine/sine
// approximation, scale rounding and encrypted arithmetic retain uncertainty.
type inverseBudget struct {
	Divisor    int     `json:"divisor"`
	Degree     int     `json:"inverse_degree"`
	Components int     `json:"components"`
	MaxInput   float64 `json:"maximum_original_coefficient_abs"`
	MaxTail    float64 `json:"maximum_inverse_tail_upper_bound"`
	SumTail    float64 `json:"sum_inverse_tail_upper_bound"`
	SumSquares float64 `json:"sum_squared_inverse_tail_upper_bound"`
	DCTail     float64 `json:"dc_inverse_tail_upper_bound"`
	MaxNonDC   float64 `json:"maximum_non_dc_inverse_tail_upper_bound"`
	DCBias     float64 `json:"ideal_sine_dc_inverse_bias_float64"`
}

func (b *inverseBudget) observe(coefficient float64, index int) {
	t := coefficient / (4 * float64(b.Divisor))
	if math.Abs(t) >= .25 {
		panic("Inverse Taylor principal branch is not certified")
	}
	z := math.Sin(2 * math.Pi * t)
	a := 1 / (2 * math.Pi)
	polynomial := a * z
	for i := 3; i <= b.Degree; i += 2 {
		a *= float64((i-2)*(i-2)) / float64(i*(i-1))
		polynomial += a * math.Pow(z, float64(i))
	}
	next := b.Degree + 2
	a *= float64((next-2)*(next-2)) / float64(next*(next-1))
	// Successive absolute Taylor terms have ratio strictly below z^2.
	tail := 4 * float64(b.Divisor) * math.Abs(a*math.Pow(z, float64(next))) / (1 - z*z)
	b.Components++
	b.MaxInput = math.Max(b.MaxInput, math.Abs(coefficient))
	b.MaxTail = math.Max(b.MaxTail, tail)
	b.SumTail += tail
	b.SumSquares += tail * tail
	if index == 0 {
		b.DCTail = tail
		b.DCBias = 4 * float64(b.Divisor) * (polynomial - t)
	} else {
		b.MaxNonDC = math.Max(b.MaxNonDC, tail)
	}
}

func main() {
	manifestPath := flag.String("manifest", "", "Original mathematical preboot vectors manifest")
	parametersPath := flag.String("parameters", "", "Saved actual native constructor output")
	output := flag.String("output", "", "Terminal JSON")
	rowsPath := flag.String("rows", "", "Per-vector plaintext encode observations")
	inputDivisor := flag.Int("input-divisor", 0, "optional public value division2 with scale51; zero preserves original scale50 audit")
	budgetPath := flag.String("inverse-budget-rows", "", "optional public coefficient Taylor inverse budget JSONL; no keys")
	flag.Parse()
	origin := time.Now()
	result := map[string]any{"status": "running", "error": nil, "public_encode_attempts": 0, "public_encode_completed": 0,
		"keygen_runs": 0, "encryptions": 0, "bootstraps": 0, "decryptions": 0, "intermediate_decryptions": 0,
		"extra_integrity_checks": 0, "security_acceptance": nil, "bootstrap_domain_acceptance": nil,
		"scope": "Public native Encoder.Encode and centered plaintext coefficients for270 original mathematical prebootstrap vectors. No ciphertext or secret exists."}
	exit := 0
	defer func() {
		if value := recover(); value != nil {
			result["status"] = "error"
			result["error"] = fmt.Sprint(value)
			exit = 1
		}
		result["elapsed_ns"] = time.Since(origin).Nanoseconds()
		var memory runtime.MemStats
		runtime.ReadMemStats(&memory)
		result["go_memstats"] = map[string]uint64{"heap_alloc": memory.HeapAlloc, "heap_sys": memory.HeapSys, "sys": memory.Sys}
		encoded, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			panic(err)
		}
		if err = os.WriteFile(*output, append(encoded, '\n'), 0600); err != nil {
			panic(err)
		}
		if exit != 0 {
			os.Exit(exit)
		}
	}()
	if *inputDivisor != 0 && *inputDivisor != 2 {
		panic("Explicit input divisor is zero or2")
	}
	divisor := 1.
	scaleBits := 50
	if *inputDivisor == 2 {
		divisor = 2
		scaleBits = 51
	}
	result["input_value_divisor"] = divisor
	result["encode_scale_log2"] = scaleBits
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		panic(err)
	}
	var manifest struct {
		Vectors []vectorRecord `json:"vectors"`
		Count   int            `json:"count"`
	}
	if err = json.Unmarshal(raw, &manifest); err != nil {
		panic(err)
	}
	if manifest.Count != 270 || len(manifest.Vectors) != 270 {
		panic("Expected all270 original preboot vectors")
	}
	raw, err = os.ReadFile(*parametersPath)
	if err != nil {
		panic(err)
	}
	var source struct {
		BootstrapParameters struct {
			ResidualParameters json.RawMessage `json:"ResidualParameters"`
		} `json:"bootstrap_parameters"`
	}
	if err = json.Unmarshal(raw, &source); err != nil {
		panic(err)
	}
	var params ckks.Parameters
	if err = params.UnmarshalJSON(source.BootstrapParameters.ResidualParameters); err != nil {
		panic(err)
	}
	if params.LogN() != 16 || params.LogDefaultScale() != 50 {
		panic("Expected exact native N16/scale50 residual parameters")
	}
	result["residual_parameters"] = params
	result["encode_level"] = 1
	result["logical_log_slots"] = 14
	result["message_ratio"] = 4
	result["mod1_neighborhood_half_width"] = .25
	encoder := ckks.NewEncoder(params)
	pt := ckks.NewPlaintext(params, 1)
	pt.LogDimensions = ring.Dimensions{Rows: 0, Cols: 14}
	pt.Scale = params.NewScale(math.Exp2(float64(scaleBits)))
	ringQ := params.RingQ().AtLevel(1)
	poly := ringQ.NewPoly()
	coeffs := make([]*big.Int, 32768)
	for i := range coeffs {
		coeffs[i] = new(big.Int)
	}
	stream, err := os.OpenFile(*rowsPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		panic(err)
	}
	defer stream.Close()
	var budgetStream *os.File
	if *budgetPath != "" {
		if divisor != 1 {
			panic("Inverse budgets start from the unchanged original scale50 coefficients")
		}
		budgetStream, err = os.OpenFile(*budgetPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			panic(err)
		}
		defer budgetStream.Close()
		result["inverse_budget_rows"] = 0
		result["inverse_budget_scope"] = "All32768 public real coefficients of each original prebootstrap vector; positive Taylor remainder upper bound under ideal sine and native degree recurrence. Sum is a conservative triangle bound for slot error from inverse truncation alone. Native sine/cosine approximation and CKKS/scale errors remain unknown."
	}
	maxCoefficient, maxDCError := 0., 0.
	allMessageDomain, allFourierBounds := true, true
	for index, row := range manifest.Vectors {
		raw, err = os.ReadFile(row.Path)
		if err != nil {
			panic(err)
		}
		if len(raw) != 16384*8 {
			panic("Original preboot vector length differs")
		}
		values := make([]float64, 16384)
		for i := range values {
			values[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
			values[i] /= divisor
			if math.IsNaN(values[i]) || math.IsInf(values[i], 0) {
				panic("Nonfinite original mathematical vector")
			}
		}
		result["public_encode_attempts"] = index + 1
		start := time.Now()
		if err = encoder.Encode(values, pt); err != nil {
			panic(err)
		}
		ringQ.INTT(pt.Value, poly)
		// Native sparse embedding has gap N/(2*logicalSlots)=2. These are
		// ordinary plaintext polynomial coefficients, not decrypted values.
		ringQ.PolyToBigintCentered(poly, 2, coeffs)
		maximum, dc := 0., 0.
		budgets := []*inverseBudget{}
		if budgetStream != nil {
			for _, div := range []int{2, 4, 8, 16} {
				for _, degree := range []int{7, 15, 31} {
					budgets = append(budgets, &inverseBudget{Divisor: div, Degree: degree})
				}
			}
		}
		for i, integer := range coeffs {
			coefficient, _ := new(big.Float).SetInt(integer).Float64()
			coefficient = math.Ldexp(coefficient, -scaleBits)
			if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) {
				panic("Nonfinite encoded coefficient")
			}
			maximum = math.Max(maximum, math.Abs(coefficient))
			for _, budget := range budgets {
				budget.observe(coefficient, i)
			}
			if i == 0 {
				dc = coefficient
			}
		}
		for _, budget := range budgets {
			entry := map[string]any{"sample_id": row.Sample, "ordinal": row.Ordinal, "statistics": budget,
				"additional_sine_cosine_approximation_error": nil, "encrypted_arithmetic_error": nil,
				"security_acceptance": nil, "formal_accuracy_acceptance": nil,
				"scale_down_input_scale_log2": 50 + int(math.Log2(float64(budget.Divisor)))}
			if err = json.NewEncoder(budgetStream).Encode(entry); err != nil {
				panic(err)
			}
			result["inverse_budget_rows"] = result["inverse_budget_rows"].(int) + 1
		}
		dcError := math.Abs(dc - row.Mean/divisor)
		bound := row.MeanAbs/divisor + 1e-12 + math.Ldexp(.5, -scaleBits)
		boundPassed := maximum <= bound && dcError <= 1e-12+1e-12*math.Abs(row.Mean/divisor)
		// The native cosine nodes cover integer centers +/-1/MessageRatio.
		// This checks the plaintext message component; secret-dependent lift
		// integers and encryption/evaluation error retain separate uncertainty.
		messagePassed := maximum+math.Ldexp(.5, -scaleBits) < 1
		observed := map[string]any{"index": index, "sample_id": row.Sample, "ordinal": row.Ordinal, "coefficient_count": 32768,
			"maximum_plaintext_coefficient_abs": maximum, "dc": dc, "dc_mean_error": dcError, "mean_abs_bound": row.MeanAbs / divisor,
			"normalized_message_fraction_max": maximum / 4, "mod1_neighborhood_half_width": .25,
			"plaintext_message_in_mod1_neighborhood": messagePassed, "fourier_bound_and_dc_passed": boundPassed,
			"duration_ns": time.Since(start).Nanoseconds()}
		line, err := json.Marshal(observed)
		if err != nil {
			panic(err)
		}
		if _, err = stream.Write(append(line, '\n')); err != nil {
			panic(err)
		}
		if err = stream.Sync(); err != nil {
			panic(err)
		}
		result["public_encode_completed"] = index + 1
		maxCoefficient = math.Max(maxCoefficient, maximum)
		maxDCError = math.Max(maxDCError, dcError)
		allMessageDomain = allMessageDomain && messagePassed
		allFourierBounds = allFourierBounds && boundPassed
	}
	result["maximum_plaintext_coefficient_abs"] = maxCoefficient
	result["maximum_dc_mean_error"] = maxDCError
	result["all_plaintext_message_domains_passed"] = allMessageDomain
	result["all_fourier_bounds_and_dc_passed"] = allFourierBounds
	result["domain_scope"] = "Only the known plaintext message component before270 bootstraps. Secret-dependent integer lift K16, ciphertext error and final model tolerance remain unverified by this plaintext audit."
	if !allFourierBounds {
		panic("Public Encode coefficients disagree with conservative normalized Fourier bound/DC")
	}
	result["status"] = "ok"
}
