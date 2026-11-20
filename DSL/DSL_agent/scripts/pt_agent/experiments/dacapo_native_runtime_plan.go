// Own-client DaCapo binding, derived from the unchanged R31 adapter source.
// Changes: N17/max16/scale51/slots16; source-preserved arbitrary scales; inline
// singleton full-slot broadcast with explicit source/math gates. No library edit.
// Independent RuntimePlan adapter. Built only in a new results work directory.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

const logSlots = 16
const variant = "dacapo-own-lattigo-n17-q55-51x16-p61x3-scale51-slots16-lazy1"

var checkInputSSA bool
var traceKeyRequests bool
var bootstrapInputDivisor int // zero preserves the original native adapter

type inputMetadataSnapshot struct {
	metadata      rlwe.MetaData
	level, degree int
}

func snapshotInput(ct *rlwe.Ciphertext) inputMetadataSnapshot {
	m := *ct.MetaData
	// MetaData.CopyNew is a struct copy. Give this observation its own
	// big.Float mantissa so a later native write cannot change the snapshot.
	m.Scale = rlwe.NewScale(ct.Scale)
	return inputMetadataSnapshot{metadata: m, level: ct.Level(), degree: ct.Degree()}
}

func inputUnchanged(ct *rlwe.Ciphertext, before inputMetadataSnapshot) bool {
	return ct.Level() == before.level && ct.Degree() == before.degree &&
		ct.Scale.Equal(before.metadata.Scale) && ct.IsNTT == before.metadata.IsNTT &&
		ct.IsMontgomery == before.metadata.IsMontgomery &&
		ct.IsBatched == before.metadata.IsBatched && ct.LogDimensions == before.metadata.LogDimensions
}

type Value struct {
	ID         string `json:"id"`
	Kind       string `json:"kind"`
	Context    string `json:"context"`
	Level      int    `json:"level"`
	Scale      int    `json:"scale_log2"`
	Components int    `json:"components"`
	NTT        bool   `json:"ntt"`
	Place      struct {
		Kind string `json:"kind"`
		Rank int    `json:"rank"`
	} `json:"place"`
}
type EncodePayload struct {
	Kind string `json:"kind"`
	Content string `json:"content"`
	Values []float64 `json:"values"`
}

type Instruction struct {
	Kind    string   `json:"kind"`
	Ordinal int      `json:"ordinal"`
	Op      string   `json:"op"`
	Inputs  []string `json:"inputs"`
	Output  string   `json:"output"`
	Attrs   struct {
		Steps          int    `json:"steps"`
		Level          int    `json:"target_level"`
		Scale          int    `json:"target_scale_log2"`
		Components     int    `json:"target_components"`
		Implementation string `json:"implementation"`
        OperatorProfile string `json:"operator_profile"`
	} `json:"attrs"`
	Payload EncodePayload `json:"payload"`
    Place struct { Kind string `json:"kind"`; Rank int `json:"rank"` } `json:"place"`
}
type Plan struct {
	Version        int             `json:"format_version"`
	Target         json.RawMessage `json:"target"`
	Values         []Value         `json:"values"`
	Inputs         []string        `json:"external_inputs"`
	Initialization []Instruction   `json:"initialization"`
	Execution      []Instruction   `json:"execution"`
	Finalization   []Instruction   `json:"finalization"`
	Outputs        []string        `json:"final_outputs"`
	Bundle         struct {
		ID      string `json:"id"`
		Version int    `json:"version"`
	} `json:"plaintext_bundle"`
}
type Blob struct {
	Content string `json:"content"`
	Bytes   int    `json:"byte_length"`
}
type Manifest struct {
	Format  int    `json:"bundle_format_version"`
	ID      string `json:"bundle_id"`
	Version int    `json:"version"`
	Blobs   []Blob `json:"blobs"`
}
type Expected struct {
	Indices []int     `json:"indices"`
	Values  []float64 `json:"values"`
	Scope   string    `json:"scope"`
	OutputMultiplier *float64 `json:"output_multiplier,omitempty"`
	OutputDecoder string `json:"output_decoder,omitempty"`
	SourcePlan string `json:"source_plan,omitempty"`
	SourceMathProof string `json:"source_math_proof,omitempty"`
	SourceFactory string `json:"source_factory,omitempty"`
	SampleID string `json:"sample_id,omitempty"`
}

var record map[string]any
var outputPath, eventPath string
var origin = time.Now()
var sequence int
var eventMu sync.Mutex
var parents []int
var bootstrapCacheCapacity = 1
var precisionBits = 51
var s2cPrecisionBits = 0     // zero retains the native default three layers of39 bits
var mod1InverseDegree = 0    // zero retains the native default without inverse polynomial
var mod1PolynomialDegree = 0 // zero retains the native default degree30
var logMessageRatio = 0      // zero retains the native default MR8

// The interval counts existing save requests, including nested native stages.
// Every stage event and in-memory SSA observation is still appended. A killed
// process may lose the full-record observations since its last checkpoint;
// returned errors and the normal terminal path always force a checkpoint.
const maxCheckpointInterval = 1000000

type checkpointSchedule struct {
	Interval        int    `json:"interval"`
	Unit            string `json:"unit"`
	Requests        int    `json:"requests"`
	WriteAttempts   int    `json:"write_attempts"`
	WritesCompleted int    `json:"writes_completed"`
	SkippedRequests int    `json:"skipped_requests"`
	ForcedRequests  int    `json:"forced_requests"`
	WriteErrors     int    `json:"write_errors"`
}

var checkpoints = checkpointSchedule{Interval: 1, Unit: "existing_save_request"}

func newCheckpointSchedule(interval int) (checkpointSchedule, error) {
	if interval < 1 || interval > maxCheckpointInterval {
		return checkpointSchedule{}, fmt.Errorf("checkpoint interval must be between1 and%d", maxCheckpointInterval)
	}
	return checkpointSchedule{Interval: interval, Unit: "existing_save_request"}, nil
}

func (schedule *checkpointSchedule) request(force bool) bool {
	schedule.Requests++
	if force {
		schedule.ForcedRequests++
	}
	if !force && schedule.Requests%schedule.Interval != 0 {
		schedule.SkippedRequests++
		return false
	}
	schedule.WriteAttempts++
	return true
}

// cacheOrder is shared by the real-key cache and the key-free regression.
func cacheOrder(order []uint64, g uint64, capacity int) (next []uint64, hit bool, evicted *uint64) {
	if capacity < 1 {
		panic("positive cache capacity required")
	}
	for _, item := range order {
		if item == g {
			hit = true
		} else {
			next = append(next, item)
		}
	}
	if !hit && len(next) >= capacity {
		old := next[0]
		evicted = &old
		next = next[1:]
	}
	next = append(next, g)
	return
}
func cacheSelftest() error {
	for _, capacity := range []int{1, 2, 16, 32, 40, 44, 48} {
		order := []uint64{}
		hits, evictions := 0, 0
		for i := 0; i < capacity; i++ {
			var hit bool
			var evicted *uint64
			order, hit, evicted = cacheOrder(order, uint64(i), capacity)
			if hit || evicted != nil {
				return fmt.Errorf("initial cache fill failed")
			}
		}
		for i := 0; i < capacity; i++ {
			var hit bool
			var evicted *uint64
			order, hit, evicted = cacheOrder(order, uint64(i), capacity)
			if !hit || evicted != nil {
				return fmt.Errorf("resident cache hit failed")
			}
			hits++
		}
		var hit bool
		var evicted *uint64
		order, hit, evicted = cacheOrder(order, 1000, capacity)
		if hit || evicted == nil || *evicted != 0 || len(order) != capacity {
			return fmt.Errorf("LRU capacity eviction failed")
		}
		evictions++
		if hits != capacity || evictions != 1 {
			return fmt.Errorf("cache counters differ")
		}
	}
	record["cache_selftest_passed"] = true
	record["cache_selftest_capacities"] = []int{1, 2, 16, 32, 40, 44, 48}
	return nil
}

func save(force ...bool) {
	forced := len(force) != 0 && force[0]
	writeNow := checkpoints.request(forced)
	record["checkpoint_policy"] = checkpoints
	if !writeNow {
		return
	}
	completed := false
	checkpoints.WritesCompleted++
	record["checkpoint_policy"] = checkpoints
	defer func() {
		if !completed {
			checkpoints.WritesCompleted--
			checkpoints.WriteErrors++
			record["checkpoint_policy"] = checkpoints
		}
	}()
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		panic(err)
	}
	temporary := outputPath + ".tmp"
	if err = os.WriteFile(temporary, append(data, '\n'), 0600); err != nil {
		panic(err)
	}
	if err = os.Rename(temporary, outputPath); err != nil {
		panic(err)
	}
	completed = true
}

// This mode exercises only small maps, event JSONL and temporary files. It runs
// before plan/parameter loading and creates no keys, ciphertext or evaluator.
func checkpointSelftest() (err error) {
	record["checkpoint_selftest_passed"] = false
	directory, err := os.MkdirTemp(filepath.Dir(outputPath), "checkpoint-selftest-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(directory)
	originalRecord, originalOutput, originalEvents := record, outputPath, eventPath
	originalCheckpoints, originalSequence, originalParents := checkpoints, sequence, parents
	defer func() {
		if failure := recover(); failure != nil {
			err = fmt.Errorf("checkpoint selftest panic: %v", failure)
		}
		record, outputPath, eventPath = originalRecord, originalOutput, originalEvents
		checkpoints, sequence, parents = originalCheckpoints, originalSequence, originalParents
		if err == nil {
			record["checkpoint_selftest_passed"] = true
			record["checkpoint_selftest_intervals"] = []int{1, 3, 1000}
			record["checkpoint_selftest_crypto_calls"] = 0
		}
	}()
	for _, interval := range []int{1, 3, 1000} {
		checkpoints, err = newCheckpointSchedule(interval)
		if err != nil {
			return err
		}
		record = map[string]any{"status": "running", "probe": 0}
		outputPath = filepath.Join(directory, fmt.Sprintf("interval-%d.json", interval))
		for i := 1; i <= interval; i++ {
			record["probe"] = i
			save()
			if checkpoints.WritesCompleted != i/interval {
				return fmt.Errorf("periodic checkpoint count differs")
			}
		}
		record["probe"] = interval + 1
		save(true)
		var observed map[string]any
		if err = readJSON(outputPath, &observed); err != nil {
			return err
		}
		if observed["probe"] != float64(interval+1) || checkpoints.WritesCompleted != 2 || checkpoints.ForcedRequests != 1 {
			return fmt.Errorf("forced checkpoint did not persist latest values")
		}
	}
	checkpoints, _ = newCheckpointSchedule(1000)
	record = map[string]any{"status": "running", "failed_stage": nil, "probe": 0}
	outputPath, eventPath = filepath.Join(directory, "error.json"), filepath.Join(directory, "error-events.jsonl")
	if stage("checkpoint.selftest_error", nil, func() error {
		record["probe"] = 7
		return fmt.Errorf("intentional key-free selftest error")
	}) == nil {
		return fmt.Errorf("selftest error was not returned")
	}
	var observed map[string]any
	if err = readJSON(outputPath, &observed); err != nil {
		return err
	}
	if observed["probe"] != float64(7) || observed["failed_stage"] != "checkpoint.selftest_error" || checkpoints.ForcedRequests != 1 {
		return fmt.Errorf("stage error checkpoint did not flush")
	}
	return nil
}
func emit(row map[string]any) {
	eventMu.Lock()
	defer eventMu.Unlock()
	row["pid"] = os.Getpid()
	row["elapsed_ns"] = time.Since(origin).Nanoseconds()
	row["clock_domain"] = "Go monotonic relative to this process origin; distinct from Python perf_counter_ns"
	f, err := os.OpenFile(eventPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if err = json.NewEncoder(f).Encode(row); err != nil {
		panic(err)
	}
}
func count(name string) { record[name] = record[name].(int) + 1 }
func stage(name string, metadata map[string]any, fn func() error) (err error) {
	sequence++
	id := sequence
	var parent any = nil
	if len(parents) > 0 {
		parent = parents[len(parents)-1]
	}
	parents = append(parents, id)
	record["active_stage"] = name
	emit(map[string]any{"event_id": id, "parent_event_id": parent, "stage": name, "type": "begin", "metadata": metadata})
	save()
	begin := time.Now()
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
		duration := time.Since(begin).Nanoseconds()
		parents = parents[:len(parents)-1]
		status := "ok"
		if err != nil {
			status = "error"
			record["failed_stage"] = name
		}
		emit(map[string]any{"event_id": id, "parent_event_id": parent, "stage": name, "type": "end", "status": status, "duration_ns": duration, "error": fmt.Sprint(err)})
		save(err != nil)
	}()
	return fn()
}
func keygen(name string, fn func()) error {
	count("keygen_api_attempts")
	return stage(name, nil, func() error { fn(); count("keygen_api_completed"); return nil })
}
func readJSON(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

var safeLeaf = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

func blobPath(bundle, content string) (string, error) {
	parts := strings.SplitN(content, ":", 2)
	if len(parts) != 2 || !safeLeaf.MatchString(parts[1]) {
		return "", fmt.Errorf("unsafe bundle content locator")
	}
	return filepath.Join(bundle, "data", parts[1]+".bin"), nil
}
func readVector(path string) ([]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) != (1<<logSlots)*8 {
		return nil, fmt.Errorf("packed vector must contain exactly65536 float64 elements")
	}
	values := make([]float64, 1<<logSlots)
	for i := range values {
		v := math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite packed input at %d", i)
		}
		values[i] = v
	}
	return values, nil
}
func payload(p ckks.Parameters) int64 {
	lq, lp := p.QCount()-1, p.PCount()-1
	rows := p.BaseRNSDecompositionVectorSize(lq, lp)
	columns := p.BaseTwoDecompositionVectorSize(lq, lp, 0)
	n := 0
	for i := 0; i < rows; i++ {
		n += columns[i]
	}
	return int64(n) * 2 * int64(p.N()) * int64(p.QCount()+p.PCount()) * 8
}
func summary(p ckks.Parameters) map[string]any {
	qb, pb := []int{}, []int{}
	for _, q := range p.Q() {
		qb = append(qb, bits.Len64(q))
	}
	for _, q := range p.P() {
		pb = append(pb, bits.Len64(q))
	}
	return map[string]any{"parameters": p, "n": p.N(), "q": p.Q(), "p": p.P(), "q_bits": qb, "p_bits": pb, "log_qp": p.LogQP(), "max_level": p.MaxLevel(), "log_default_scale": p.LogDefaultScale(), "physical_log_slots": p.LogMaxSlots(), "logical_log_slots": logSlots, "secret_distribution_config": p.Xs(), "evaluation_key_uint64_payload_bytes": payload(p)}
}
func parameters() (p ckks.Parameters, bp bootstrapping.Parameters, err error) {
	err = stage("parameters.construct", nil, func() error {
		logQ := []int{55}
		for range 16 {
			logQ = append(logQ, precisionBits)
		}
		p, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: 17, LogQ: logQ, LogP: []int{61, 61, 61}, LogDefaultScale: precisionBits})
		if err != nil {
			return err
		}
		bootLiteral := bootstrapping.ParametersLiteral{LogN: utils.Pointy(17), LogSlots: utils.Pointy(logSlots), LogP: []int{61, 61, 61, 61}}
		if s2cPrecisionBits != 0 {
			bootLiteral.SlotsToCoeffsFactorizationDepthAndLogScales = [][]int{{s2cPrecisionBits}, {s2cPrecisionBits}, {s2cPrecisionBits}}
		}
		if mod1InverseDegree != 0 {
			bootLiteral.Mod1InvDegree = utils.Pointy(mod1InverseDegree)
		}
		if mod1PolynomialDegree != 0 {
			bootLiteral.Mod1Degree = utils.Pointy(mod1PolynomialDegree)
		}
		if logMessageRatio != 0 {
			bootLiteral.LogMessageRatio = utils.Pointy(logMessageRatio)
		}
		bp, err = bootstrapping.NewParametersFromLiteral(p, bootLiteral)
		return err
	})
	if err == nil {
		record["residual_parameters"] = summary(p)
		record["bootstrap_parameters"] = bp
		record["bootstrap_parameter_summary"] = summary(bp.BootstrappingParameters)
		effectiveS2C := bootstrapping.DefaultSlotsToCoeffsLogScale
		if s2cPrecisionBits != 0 {
			effectiveS2C = s2cPrecisionBits
		}
		record["bootstrap_precision_policy"] = map[string]any{"slots_to_coeffs_log_scales": [][]int{{effectiveS2C}, {effectiveS2C}, {effectiveS2C}}, "explicit_s2c_override": s2cPrecisionBits != 0, "source_default_s2c_bits": bootstrapping.DefaultSlotsToCoeffsLogScale, "mod1_algorithm_changed": mod1InverseDegree != 0 || mod1PolynomialDegree != 0, "mod1_inverse_degree": bp.Mod1ParametersLiteral.Mod1InvDegree, "explicit_mod1_inverse_override": mod1InverseDegree != 0, "mod1_degree": bp.Mod1ParametersLiteral.Mod1Degree, "explicit_mod1_degree_override": mod1PolynomialDegree != 0}
		record["bootstrap_precision_policy"].(map[string]any)["log_message_ratio"] = bp.Mod1ParametersLiteral.LogMessageRatio
		record["bootstrap_precision_policy"].(map[string]any)["explicit_log_message_ratio_override"] = logMessageRatio != 0
		record["security_acceptance"] = nil
		record["concrete_prime_binding_verified"] = false
		record["prime_binding_scope"] = "Actual Lattigo constructor primes only; original Poseidon operator placeholder does not bind concrete Q/P"
		record["secret_sampler_attribution"] = map[string]any{"residual_actual_generation": "No secret generation in parameters mode; execute uses residual GenSecretKeyNew with residual configured Xs", "bootstrap_configured_Xs": bp.BootstrappingParameters.Xs(), "same_n_bootstrap_actual_secret": "execute extends the same residual secret basis; configured bootstrap H192 does not create a second H192 secret", "sampled_hamming_weight": nil}
		save()
	}
	return
}

type lazyKeys struct {
	mu                                 sync.Mutex
	p                                  ckks.Parameters
	sk                                 *rlwe.SecretKey
	kg                                 *rlwe.KeyGenerator
	allowed                            map[uint64]bool
	list                               []uint64
	cache                              map[uint64]*rlwe.GaloisKey
	order                              []uint64
	capacity                           int
	name                               string
	gets, hits, generations, evictions int
}

func newLazy(name string, p ckks.Parameters, sk *rlwe.SecretKey, list []uint64, capacities ...int) *lazyKeys {
	allowed := map[uint64]bool{}
	for _, g := range list {
		allowed[g] = true
	}
	list = nil
	for g := range allowed {
		list = append(list, g)
	}
	sort.Slice(list, func(i, j int) bool { return list[i] < list[j] })
	capacity := 1
	if len(capacities) > 0 {
		capacity = capacities[0]
	}
	return &lazyKeys{p: p, sk: sk, kg: rlwe.NewKeyGenerator(p), allowed: allowed, list: list, name: name, capacity: capacity, cache: map[uint64]*rlwe.GaloisKey{}}
}
func (l *lazyKeys) get(g uint64) (*rlwe.GaloisKey, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.gets++
	if !l.allowed[g] {
		return nil, fmt.Errorf("unlisted Galois element %d", g)
	}
	next, hit, evicted := cacheOrder(l.order, g, l.capacity)
	if traceKeyRequests {
		var parent any
		if len(parents) > 0 {
			parent = parents[len(parents)-1]
		}
		emit(map[string]any{"type": "galois_request", "stage": l.name + ".get", "parent_event_id": parent,
			"cache": l.name, "call_index": l.gets, "galois_element": g, "hit": hit, "evicted": evicted,
			"scope": "Public Galois element/cache decisions only; no secret or key coefficients"})
	}
	if hit {
		l.hits++
		l.order = next
		return l.cache[g], nil
	}
	if evicted != nil {
		delete(l.cache, *evicted)
		l.evictions++
		runtime.GC()
	}
	var key *rlwe.GaloisKey
	err := keygen(l.name+".galois", func() { key = l.kg.GenGaloisKeyNew(g, l.sk) })
	if err != nil {
		return nil, err
	}
	l.cache[g] = key
	l.order = next
	l.generations++
	record[l.name+"_cache"] = l.summary()
	return key, nil
}
func (l *lazyKeys) summary() map[string]any {
	return map[string]any{"capacity": l.capacity, "get_calls": l.gets, "hits": l.hits, "generated": l.generations, "evictions": l.evictions, "allowed_keys": len(l.list), "resident_keys": len(l.cache), "serialized_keys": 0}
}
func (l *lazyKeys) attach(mem *rlwe.MemEvaluationKeySet) {
	mem.GetGaloisKeyFunc = l.get
	mem.GetGaloisKeysListFunc = func() []uint64 { return append([]uint64{}, l.list...) }
}
func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
func buildBootstrap(p ckks.Parameters, bp bootstrapping.Parameters, sk *rlwe.SecretKey) (evaluator *bootstrapping.Evaluator, lazy *lazyKeys, err error) {
	bigp := bp.BootstrappingParameters
	kg := rlwe.NewKeyGenerator(bigp)
	var extended *rlwe.SecretKey
	err = stage("bootstrap.extend_same_secret", nil, func() error {
		extended = rlwe.NewSecretKey(bigp)
		q, aux := bigp.RingQ(), bigp.RingP()
		buffer := q.NewPoly()
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(q, q, sk.Value.Q, buffer, extended.Value.Q)
		rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(q, aux, sk.Value.Q, buffer, extended.Value.P)
		return nil
	})
	if err != nil {
		return
	}
	var d2s, s2d *rlwe.EvaluationKey
	if bp.EphemeralSecretWeight != 0 {
		var sparse rlwe.Parameters
		sparse, err = rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{LogN: bigp.LogN(), Q: bigp.Q()[:1], P: bigp.P()[:1]})
		if err != nil {
			return
		}
		ks := rlwe.NewKeyGenerator(sparse)
		var auxiliary *rlwe.SecretKey
		if err = keygen("bootstrap.ephemeral_secret", func() { auxiliary = ks.GenSecretKeyWithHammingWeightNew(bp.EphemeralSecretWeight) }); err != nil {
			return
		}
		if err = keygen("bootstrap.dense_to_sparse", func() { d2s = ks.GenEvaluationKeyNew(extended, auxiliary) }); err != nil {
			return
		}
		if err = keygen("bootstrap.sparse_to_dense", func() { s2d = kg.GenEvaluationKeyNew(auxiliary, extended) }); err != nil {
			return
		}
		record["bootstrap_ephemeral_sampler"] = map[string]any{"kind": "fixed Hamming weight", "h": bp.EphemeralSecretWeight, "source": "GenSecretKeyWithHammingWeightNew; independent auxiliary key"}
	}
	var rlk *rlwe.RelinearizationKey
	if err = keygen("bootstrap.relinearization_key", func() { rlk = kg.GenRelinearizationKeyNew(extended) }); err != nil {
		return
	}
	lazy = newLazy("bootstrap_lazy", bigp, extended, bp.GaloisElements(bigp), bootstrapCacheCapacity)
	mem := rlwe.NewMemEvaluationKeySet(rlk)
	lazy.attach(mem)
	keys := &bootstrapping.EvaluationKeys{EvkDenseToSparse: d2s, EvkSparseToDense: s2d, MemEvaluationKeySet: mem}
	var dftBegin time.Time
	var dftID int
	dft.R27MatrixObserver = func(literal dft.MatrixLiteral, begin bool) {
		if begin {
			dftBegin = time.Now()
			sequence++
			dftID = sequence
			emit(map[string]any{"event_id": dftID, "stage": "bootstrap.dft_matrix", "type": "begin", "literal": literal})
		} else {
			emit(map[string]any{"event_id": dftID, "stage": "bootstrap.dft_matrix", "type": "end", "duration_ns": time.Since(dftBegin).Nanoseconds()})
			count("dft_matrices_completed")
			save()
		}
	}
	defer func() { dft.R27MatrixObserver = nil }()
	err = stage("bootstrap.evaluator_constructor", nil, func() error { var e error; evaluator, e = bootstrapping.NewEvaluator(bp, keys); return e })
	record["bootstrap_constructor_cache"] = lazy.summary()
	record["bootstrap_checkkeys_real_keys_generated"] = lazy.generations
	return
}

func loadPlan(path, bundle string) (plan Plan, values map[string]Value, encodes map[string]EncodePayload, manifest map[string]Blob, last map[string]int, err error) {
	if err = readJSON(path, &plan); err != nil {
		return
	}
	if plan.Version != 1 || len(plan.Inputs) != 1 || len(plan.Outputs) != 1 || len(plan.Finalization) != 0 {
		err = fmt.Errorf("unsupported RuntimePlan contract")
		return
	}
	values = map[string]Value{}
	encodes = map[string]EncodePayload{}
	manifest = map[string]Blob{}
	last = map[string]int{}
	for _, v := range plan.Values {
		if _, exists := values[v.ID]; exists || v.ID == "" || !v.NTT || v.Level < 0 || v.Level > 16 || v.Scale < 0 || v.Scale > 1023 || v.Context != "dacapo-original-n17-r51-export" || v.Place.Kind != "host" || v.Place.Rank != 0 || (v.Kind != "plaintext" && v.Kind != "ciphertext") || (v.Kind == "plaintext" && v.Components != 1) || (v.Kind == "ciphertext" && v.Components != 2 && v.Components != 3) {
			err = fmt.Errorf("invalid value %s", v.ID)
			return
		}
		values[v.ID] = v
	}
	var m Manifest
	if err = readJSON(filepath.Join(bundle, "manifest.json"), &m); err != nil {
		return
	}
	if m.Format != 1 || m.Version != 1 || m.ID != plan.Bundle.ID || plan.Bundle.Version != 1 {
		err = fmt.Errorf("bundle metadata mismatch")
		return
	}
	for _, b := range m.Blobs {
		if _, exists := manifest[b.Content]; exists || b.Bytes != (1<<logSlots)*8 {
			err = fmt.Errorf("invalid blob metadata")
			return
		}
		if _, err = blobPath(bundle, b.Content); err != nil {
			return
		}
		manifest[b.Content] = b
	}
    for _, id := range []string{plan.Inputs[0],plan.Outputs[0]} {
        v, ok := values[id]
        if !ok || v.Kind != "ciphertext" || v.Components != 2 { err=fmt.Errorf("invalid ciphertext input/output metadata");return }
    }
	seen := map[string]bool{plan.Inputs[0]: true}
	for n, i := range plan.Initialization {
        if i.Ordinal != n { err=fmt.Errorf("initialization ordinal differs");return }
		if i.Kind != "encode" || seen[i.Output] || values[i.Output].Kind != "plaintext" {
			err = fmt.Errorf("invalid initialization")
			return
		}
		if i.Payload.Kind == "bundle" {
            if _, ok := manifest[i.Payload.Content]; !ok { err = fmt.Errorf("blob absent from manifest"); return }
        } else if i.Payload.Kind == "inline" {
            if len(i.Payload.Values) != 1 || math.IsNaN(i.Payload.Values[0]) || math.IsInf(i.Payload.Values[0], 0) {
                err = fmt.Errorf("only finite singleton inline payloads have an explicit interpretation"); return
            }
        } else { err = fmt.Errorf("unsupported plaintext payload"); return }
		seen[i.Output] = true
		encodes[i.Output] = i.Payload
	}
	arity := map[string]int{"mod_switch": 1, "rotate": 1, "mul_cp": 2, "rescale": 1, "add_cc": 2, "add_cp": 2, "boot": 1, "mul_cc": 2, "relinearize": 1, "negate": 1}
	for n, i := range plan.Execution {
		size, ok := arity[i.Op]
		if !ok || i.Kind != "compute" || len(i.Inputs) != size || seen[i.Output] || i.Ordinal != len(plan.Initialization)+n || i.Place.Kind != "host" || i.Place.Rank != 0 {
			err = fmt.Errorf("unsupported instruction %d", i.Ordinal)
			return
		}
		for _, id := range i.Inputs {
			if !seen[id] {
				err = fmt.Errorf("unproduced input %s", id)
				return
			}
			last[id] = n
		}
        if err=validateSymbolicOperation(i,values);err!=nil{return}
		seen[i.Output] = true
	}
	if len(seen) != len(values) || !seen[plan.Outputs[0]] {
		err = fmt.Errorf("unproduced values or output")
	}
	return
}

// Verify the complete original symbolic contract before any constructor or key.
// Native arithmetic separately validates the actual scale recurrence after each op.
func validateSymbolicOperation(op Instruction, values map[string]Value) error {
    a, ok := values[op.Inputs[0]]
    out, outOK := values[op.Output]
    if !ok || !outOK || a.Kind!="ciphertext" || out.Kind!="ciphertext" {
        return fmt.Errorf("ciphertext operand/output metadata missing at ordinal%d",op.Ordinal)
    }
    level,scale,components:=a.Level,a.Scale,a.Components
    if len(op.Inputs)==2 {
        b,exists:=values[op.Inputs[1]]
        requiredKind:="ciphertext"
        if strings.HasSuffix(op.Op,"_cp") { requiredKind="plaintext" }
        if !exists || b.Kind!=requiredKind || b.Level!=level || b.Context!=a.Context {
            return fmt.Errorf("binary operand contract differs at ordinal%d",op.Ordinal)
        }
        if strings.HasPrefix(op.Op,"add") {
            if b.Scale!=scale { return fmt.Errorf("nominal addition scales differ at ordinal%d",op.Ordinal) }
            components=max(components,b.Components)
        } else { scale+=b.Scale;components+=b.Components-1 }
    }
    switch op.Op {
    case "mod_switch", "rescale":
        if op.Attrs.Level<0 || op.Attrs.Level>=level { return fmt.Errorf("invalid level drop at ordinal%d",op.Ordinal) }
        if op.Op=="rescale" {
            if op.Attrs.Level!=level-1 || op.Attrs.Scale!=scale-51 { return fmt.Errorf("original rescale must consume one nominal51-bit prime at ordinal%d",op.Ordinal) }
            scale-=51
        }
        level=op.Attrs.Level
    case "boot":
        if level!=3 || scale!=51 || components!=2 || op.Attrs.Implementation!="native" || op.Attrs.OperatorProfile!="dacapo-original-l3-to16-s51-export" || op.Attrs.Level!=16 || op.Attrs.Scale!=51 || op.Attrs.Components!=2 {
            return fmt.Errorf("original bootstrap contract must be3/51 to16/51 at ordinal%d",op.Ordinal)
        }
        level,scale,components=16,51,2
    case "relinearize":
        if components!=3 { return fmt.Errorf("relinearize requires3components at ordinal%d",op.Ordinal) }
        components=2
    case "rotate":
        if components!=2 { return fmt.Errorf("rotate requires2components at ordinal%d",op.Ordinal) }
    }
    if out.Level!=level || out.Scale!=scale || out.Components!=components {
        return fmt.Errorf("original output metadata equation differs at ordinal%d",op.Ordinal)
    }
    return nil
}

type plaintextCache struct {
	p                      ckks.Parameters
	encoder                *ckks.Encoder
	bundle                 string
	values                 map[string]Value
	encodes                map[string]EncodePayload
	name                   string
	current                *rlwe.Plaintext
	hits, loads, evictions int
	peak                   int64
}

func (c *plaintextCache) get(id string) (pt *rlwe.Plaintext, err error) {
	if c.current != nil && c.name == id {
		c.hits++
		return c.current, nil
	}
	if c.current != nil {
		c.current = nil
		c.evictions++
	}
	v, ok := c.values[id]
	if !ok || v.Kind != "plaintext" {
		return nil, fmt.Errorf("unknown plaintext %s", id)
	}
	payload, ok := c.encodes[id]
	if !ok {
		return nil, fmt.Errorf("plaintext payload missing")
	}
	count("plaintext_encoding_attempts")
	err = stage("plaintext.read_encode", map[string]any{"value_id": id, "content_locator": payload.Content, "payload_kind": payload.Kind, "level": v.Level, "scale_log2": v.Scale}, func() error {
        var raw []float64
        if payload.Kind == "inline" {
            raw = make([]float64, 1<<logSlots)
            for i := range raw { raw[i] = payload.Values[0] }
            record["inline_scalar_broadcasts"] = record["inline_scalar_broadcasts"].(int)+1
        } else {
            path, e := blobPath(c.bundle, payload.Content)
            if e != nil { return e }
            raw, e = readVector(path)
            if e != nil { return e }
        }
		pt = ckks.NewPlaintext(c.p, v.Level)
		pt.Scale = rlwe.NewScale(math.Exp2(float64(v.Scale)))
		pt.LogDimensions.Cols = logSlots
		if e := c.encoder.Encode(raw, pt); e != nil {
			return e
		}
		c.loads++
		count("plaintext_encoding_completed")
		return nil
	})
	if err != nil {
		return nil, err
	}
	c.name = id
	c.current = pt
	bytes := int64(c.p.N()) * int64(v.Level+1) * 8
	if bytes > c.peak {
		c.peak = bytes
	}
	record["plaintext_cache"] = c.summary()
	return
}
func (c *plaintextCache) summary() map[string]any {
	return map[string]any{"capacity": 1, "loads": c.loads, "hits": c.hits, "evictions": c.evictions, "resident_plaintexts": boolInt(c.current != nil), "peak_coefficient_bytes": c.peak, "raw_vectors_cached": 0}
}
func validateCipher(ct *rlwe.Ciphertext, v Value, predicted rlwe.Scale) error {
	if ct == nil || ct.Level() != v.Level || ct.Degree()+1 != v.Components || !ct.IsNTT || ct.LogDimensions.Cols != logSlots {
		return fmt.Errorf("ciphertext metadata mismatch for %s", v.ID)
	}
	if ct.Scale.Cmp(predicted) != 0 {
		return fmt.Errorf("actual native scale recurrence mismatch for %s", v.ID)
	}
	drift := math.Abs(ct.Scale.Log2() - float64(v.Scale))
	if drift > record["max_nominal_scale_log2_deviation"].(float64) {
		record["max_nominal_scale_log2_deviation"] = drift
	}
	return nil
}
func compare(want, got []float64) (map[string]any, error) {
	if len(want) == 0 || len(want) != len(got) {
		return nil, fmt.Errorf("final output shape mismatch")
	}
	maximum, ratio := 0., 0.
	passed := true
	for i, x := range want {
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(got[i]) || math.IsInf(got[i], 0) {
			return nil, fmt.Errorf("nonfinite final output")
		}
		delta := math.Abs(got[i] - x)
		bound := 1e-5 + 1e-4*math.Abs(x)
		maximum = math.Max(maximum, delta)
		ratio = math.Max(ratio, delta/bound)
		passed = passed && delta <= bound
	}
	return map[string]any{"passed": passed, "elements": len(want), "max_absolute_error": maximum, "max_error_ratio": ratio, "absolute_tolerance": 1e-5, "relative_tolerance": 1e-4}, nil
}

func expectedOutputMultiplier(reference Expected, planPath, mathProofPath string, fixture bool) (float64, error) {
	multiplier := 1.0
	if reference.OutputMultiplier != nil { multiplier = *reference.OutputMultiplier }
	if math.IsNaN(multiplier) || math.IsInf(multiplier, 0) || (multiplier != 1 && multiplier != 32) {
		return 0, fmt.Errorf("final output multiplier must be finite1 or32")
	}
	if fixture {
		if multiplier != 1 { return 0, fmt.Errorf("public fixture final outputs retain multiplier1") }
		return multiplier, nil
	}
	if reference.OutputMultiplier == nil || multiplier != 32 || reference.Scope == "" ||
		reference.OutputDecoder != "original-physical32-first10" ||
		reference.SourceFactory != "pt_agent.experiments.r13_common_models:resnet20_silu" ||
		!sameSourcePlan(reference.SourcePlan, planPath) || !sameSourcePlan(reference.SourceMathProof, mathProofPath) {
		return 0, fmt.Errorf("original final logits require explicit first10 times32 and same-source mathematical proof")
	}
	if reference.SampleID != "train:0" && reference.SampleID != "train:1" && reference.SampleID != "train:2" {
		return 0, fmt.Errorf("original final reference must identify an accepted training probe")
	}
	if len(reference.Indices) != 10 || len(reference.Values) != 10 { return 0, fmt.Errorf("original final decoder requires exactly10 logits") }
	for i, index := range reference.Indices { if index != i { return 0, fmt.Errorf("original final decoder must select the first10 slots") } }
	return multiplier, nil
}

// This pure conversion is called only after the sole final decrypt/decode.
// It changes reporting units, without touching a ciphertext or its metadata.
func selectFinalOutput(decoded []float64, indices []int, multiplier float64) ([]float64, []float64, error) {
	if math.IsNaN(multiplier) || math.IsInf(multiplier, 0) || (multiplier != 1 && multiplier != 32) {
		return nil, nil, fmt.Errorf("unsupported final output multiplier")
	}
	raw, scaled := make([]float64, len(indices)), make([]float64, len(indices))
	for i, index := range indices {
		if index < 0 || index >= len(decoded) { return nil, nil, fmt.Errorf("final output index out of range") }
		raw[i] = decoded[index]
		scaled[i] = raw[i] * multiplier
		if math.IsNaN(raw[i]) || math.IsInf(raw[i], 0) || math.IsNaN(scaled[i]) || math.IsInf(scaled[i], 0) {
			return nil, nil, fmt.Errorf("nonfinite final output conversion")
		}
	}
	return raw, scaled, nil
}

func execute(plan Plan, values map[string]Value, encodes map[string]EncodePayload, last map[string]int, bundle, input, expected string, p ckks.Parameters, bp bootstrapping.Parameters, allowBoot bool, planPath, mathProofPath string, fixture bool) error {
	hasBoot := false
	rotationMap := map[int]bool{}
	for _, i := range plan.Execution {
		hasBoot = hasBoot || i.Op == "boot"
		if i.Op == "rotate" {
			r := (i.Attrs.Steps%(1<<logSlots) + (1 << logSlots)) % (1 << logSlots)
			if r != 0 {
				rotationMap[r] = true
			}
		}
	}
	if hasBoot && !allowBoot {
		return fmt.Errorf("bootstrap execution was not explicitly selected")
	}
	var x []float64
	var reference Expected
	var outputMultiplier float64
	if err := stage("input.read", nil, func() error {
		var e error
		x, e = readVector(input)
		if e != nil {
			return e
		}
		if e = readJSON(expected, &reference); e != nil {
			return e
		}
		if len(reference.Indices) != len(reference.Values) || len(reference.Values) == 0 {
			return fmt.Errorf("final reference shape mismatch")
		}
		for n, i := range reference.Indices {
			if i < 0 || i >= 1<<logSlots || math.IsNaN(reference.Values[n]) || math.IsInf(reference.Values[n], 0) {
				return fmt.Errorf("invalid final reference")
			}
		}
		outputMultiplier, e = expectedOutputMultiplier(reference, planPath, mathProofPath, fixture)
		if e != nil { return e }
		record["output_multiplier"] = outputMultiplier
		record["final_decoder_contract"] = map[string]any{"decoder": reference.OutputDecoder, "source_plan": reference.SourcePlan,
			"source_math_proof": reference.SourceMathProof, "source_factory": reference.SourceFactory,
			"sample_id": reference.SampleID, "scope": reference.Scope, "public_fixture": fixture,
			"application_phase": "Only after the sole final decrypt/decode; original decoded logit units"}
		return nil
	}); err != nil {
		return err
	}
	kg := rlwe.NewKeyGenerator(p)
	var sk *rlwe.SecretKey
	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	if err := keygen("residual.secret_key", func() { sk = kg.GenSecretKeyNew() }); err != nil {
		return err
	}
	record["actual_residual_secret_generation"] = map[string]any{"method": "GenSecretKeyNew", "sampler": p.Xs(), "sampled_hamming_weight": nil}
	if err := keygen("residual.public_key", func() { pk = kg.GenPublicKeyNew(sk) }); err != nil {
		return err
	}
	if err := keygen("residual.relinearization_key", func() { rlk = kg.GenRelinearizationKeyNew(sk) }); err != nil {
		return err
	}
	gals := []uint64{}
	for r := range rotationMap {
		gals = append(gals, p.GaloisElement(r))
	}
	lazy := newLazy("residual_lazy", p, sk, gals)
	mem := rlwe.NewMemEvaluationKeySet(rlk)
	lazy.attach(mem)
	eval := ckks.NewEvaluator(p, mem)
	var bootstrapper *bootstrapping.Evaluator
	var bootLazy *lazyKeys
	if hasBoot {
		var err error
		bootstrapper, bootLazy, err = buildBootstrap(p, bp, sk)
		if err != nil {
			return err
		}
	}
	pool := map[string]*rlwe.Ciphertext{}
	encoder := ckks.NewEncoder(p)
	cache := &plaintextCache{p: p, encoder: encoder, bundle: bundle, values: values, encodes: encodes}
	count("encryption_attempts")
	if err := stage("input.encode_encrypt", nil, func() error {
		v := values[plan.Inputs[0]]
		pt := ckks.NewPlaintext(p, v.Level)
		pt.Scale = rlwe.NewScale(math.Exp2(float64(v.Scale)))
		pt.LogDimensions.Cols = logSlots
		if err := encoder.Encode(x, pt); err != nil {
			return err
		}
		ct, err := rlwe.NewEncryptor(p, pk).EncryptNew(pt)
		if err != nil {
			return err
		}
		if err = validateCipher(ct, v, pt.Scale); err != nil {
			return err
		}
		pool[v.ID] = ct
		count("encryption_completed")
		return nil
	}); err != nil {
		return err
	}
	x = nil
	peakCount, peakBytes := 0, int64(0)
	observe := func() {
		bytes := int64(0)
		for _, ct := range pool {
			bytes += int64(ct.Degree()+1) * int64(ct.Level()+1) * int64(p.N()) * 8
		}
		if len(pool) > peakCount {
			peakCount = len(pool)
		}
		if bytes > peakBytes {
			peakBytes = bytes
		}
		record["ciphertext_pool"] = map[string]any{"live": len(pool), "peak_live": peakCount, "peak_coefficient_bytes": peakBytes}
	}
	observe()
	attempts := map[string]int{}
	completed := map[string]int{}
	record["op_attempts_by_kind"] = attempts
	record["op_completed_by_kind"] = completed
	for index, op := range plan.Execution {
		a, ok := pool[op.Inputs[0]]
		if !ok {
			return fmt.Errorf("missing live ciphertext %s", op.Inputs[0])
		}
		if err := validateCipher(a, values[op.Inputs[0]], a.Scale); err != nil {
			return fmt.Errorf("input %s: %w", op.Inputs[0], err)
		}
		v := values[op.Output]
		var b *rlwe.Ciphertext
		var pt *rlwe.Plaintext
		if len(op.Inputs) == 2 {
			if values[op.Inputs[1]].Kind == "plaintext" {
				var err error
				pt, err = cache.get(op.Inputs[1])
				if err != nil {
					return err
				}
			} else {
				b = pool[op.Inputs[1]]
				if b == nil {
					return fmt.Errorf("missing second ciphertext")
				}
				if err := validateCipher(b, values[op.Inputs[1]], b.Scale); err != nil {
					return fmt.Errorf("second input %s: %w", op.Inputs[1], err)
				}
			}
		}
		count("compute_attempts")
		attempts[op.Op]++
		err := stage("runtime.compute", map[string]any{"ordinal": op.Ordinal, "op": op.Op, "output": op.Output, "index": index}, func() error {
			var aBefore, bBefore inputMetadataSnapshot
			if checkInputSSA {
				aBefore = snapshotInput(a)
				if b != nil {
					bBefore = snapshotInput(b)
				}
			}
			var out *rlwe.Ciphertext
			predicted := a.Scale
			var err error
			switch op.Op {
			case "mod_switch":
				if op.Attrs.Level < 0 || op.Attrs.Level > a.Level() {
					return fmt.Errorf("invalid mod-switch target")
				}
				out = a.CopyNew()
				eval.DropLevel(out, a.Level()-op.Attrs.Level)
			case "rotate":
				out = ckks.NewCiphertext(p, a.Degree(), a.Level())
				r := (op.Attrs.Steps%(1<<logSlots) + (1 << logSlots)) % (1 << logSlots)
				if r == 0 {
					out = a.CopyNew()
				} else {
					err = eval.Rotate(a, r, out)
				}
			case "mul_cp":
				out = ckks.NewCiphertext(p, a.Degree(), a.Level())
				predicted = a.Scale.Mul(pt.Scale)
				err = eval.Mul(a, pt, out)
			case "mul_cc":
				out = ckks.NewCiphertext(p, a.Degree()+b.Degree(), a.Level())
				predicted = a.Scale.Mul(b.Scale)
				err = eval.Mul(a, b, out)
			case "rescale":
				if op.Attrs.Level != a.Level()-1 {
					return fmt.Errorf("rescale must consume one prime")
				}
				out = ckks.NewCiphertext(p, a.Degree(), op.Attrs.Level)
				predicted = a.Scale.Div(rlwe.NewScale(p.Q()[a.Level()]))
				err = eval.Rescale(a, out)
			case "add_cc":
				out = ckks.NewCiphertext(p, max(a.Degree(), b.Degree()), a.Level())
				if b.Scale.Cmp(predicted) > 0 {
					predicted = b.Scale
				}
				err = eval.Add(a, b, out)
			case "add_cp":
				out = ckks.NewCiphertext(p, a.Degree(), a.Level())
				if pt.Scale.Cmp(predicted) > 0 {
					predicted = pt.Scale
				}
				err = eval.Add(a, pt, out)
			case "relinearize":
				out = ckks.NewCiphertext(p, 1, a.Level())
				err = eval.Relinearize(a, out)
			case "negate":
				out = ckks.NewCiphertext(p, a.Degree(), a.Level())
				err = eval.Mul(a, -1, out)
			case "boot":
				if bootstrapper == nil || op.Attrs.Implementation != "native" {
					return fmt.Errorf("native bootstrap unavailable")
				}
				count("bootstrap_attempts")
				entryLevel, entryScale := a.Level(), a.Scale.Log2()
				// BootstrapMany can reuse shared ciphertext storage. Preserve the
				// input SSA value for any later consumers in the original plan.
				bootstrapInput := a.CopyNew()
				if bootstrapInputDivisor != 0 {
					if err = stage("bootstrap.input_value_normalization", map[string]any{"ordinal": op.Ordinal,
						"decoded_value_transform": fmt.Sprintf("x -> x/%d", bootstrapInputDivisor), "original_scale_log2": entryScale, "value_divisor": bootstrapInputDivisor}, func() error {
						// Only the COPY changes represented value. Scale.Mul owns its new big.Float.
						bootstrapInput.Scale = bootstrapInput.Scale.Mul(rlwe.NewScale(bootstrapInputDivisor))
						count("bootstrap_input_value_divisions")
						return nil
					}); err != nil {
						return err
					}
				}
				internalInputScale := bootstrapInput.Scale.Log2()
				valueDivisor := 1
				if bootstrapInputDivisor != 0 {
					valueDivisor = bootstrapInputDivisor
				}
				beforeCache := bootLazy.summary()
				bootstrapBegin := time.Now()
				count("native_bootstrap_api_attempts")
				out, err = bootstrapper.Bootstrap(bootstrapInput)
				if err == nil {
					count("native_bootstrap_api_completed")
				}
				bootstrapDuration := time.Since(bootstrapBegin).Nanoseconds()
				if a.Level() != entryLevel || a.Scale.Log2() != entryScale {
					return fmt.Errorf("native bootstrap mutated its original SSA input metadata")
				}
				var memory runtime.MemStats
				runtime.ReadMemStats(&memory)
				row := map[string]any{"index": record["bootstrap_attempts"], "ordinal": op.Ordinal, "duration_ns": bootstrapDuration,
					"input_level": entryLevel, "input_scale_log2": entryScale, "input_ssa_preserved": true,
					"target_level": v.Level, "cache_before": beforeCache, "cache_after": bootLazy.summary(),
					"internal_input_scale_log2": internalInputScale, "input_value_divisor": valueDivisor,
					"native_completed": err == nil, "go_heap_alloc": memory.HeapAlloc, "go_heap_sys": memory.HeapSys, "go_sys": memory.Sys,
					"error": fmt.Sprint(err), "scope": "Actual native bootstrap call; includes lazy-key generation on cache misses"}
				record["bootstrap_observations"] = append(record["bootstrap_observations"].([]map[string]any), row)
				if err == nil {
					count("bootstrap_completed")
					record["last_bootstrap_native_output_level"] = out.Level()
					row["native_output_level"] = out.Level()
					row["native_output_scale_log2"] = out.Scale.Log2()
					if out.Level() < v.Level {
						return fmt.Errorf("native bootstrap output level below plan target")
					}
					eval.DropLevel(out, out.Level()-v.Level)
					predicted = p.DefaultScale()
					count("bootstrap_post_drop_level_operations")
					if bootstrapInputDivisor != 0 {
						if err = stage("bootstrap.output_integer_restore", map[string]any{"ordinal": op.Ordinal,
							"decoded_value_transform": fmt.Sprintf("NativeBoot(x/%d) -> %d*NativeBoot(x/%d)", bootstrapInputDivisor, bootstrapInputDivisor, bootstrapInputDivisor), "level": out.Level(), "integer_multiplier": bootstrapInputDivisor}, func() error {
							if e := eval.Mul(out, bootstrapInputDivisor, out); e != nil {
								return e
							}
							count("bootstrap_output_integer_restores")
							return nil
						}); err != nil {
							return err
						}
					}
					row["restored_output_scale_log2"] = out.Scale.Log2()
					row["restored_output_level"] = out.Level()
					row["output_integer_multiplier"] = valueDivisor
				}
			default:
				return fmt.Errorf("unsupported op %s", op.Op)
			}
			if err != nil {
				return err
			}
			if checkInputSSA {
				unchanged := inputUnchanged(a, aBefore) && (b == nil || inputUnchanged(b, bBefore))
				record["input_ssa_observations"] = append(record["input_ssa_observations"].([]map[string]any),
					map[string]any{"ordinal": op.Ordinal, "op": op.Op, "inputs": op.Inputs,
						"metadata_preserved": unchanged, "input_level": aBefore.level,
						"input_components": aBefore.degree + 1, "input_scale_log2": aBefore.metadata.Scale.Log2(),
						"scope": "Input level, degree, native scale and encoding flags; coefficients verified by final branched mathematical comparison"})
				if !unchanged {
					return fmt.Errorf("native operation mutated its input SSA metadata at ordinal %d", op.Ordinal)
				}
			}
			if err = validateCipher(out, v, predicted); err != nil {
				return err
			}
			pool[op.Output] = out
			observe()
			count("compute_completed")
			completed[op.Op]++
			for _, id := range op.Inputs {
				if last[id] == index && id != plan.Outputs[0] {
					delete(pool, id)
				}
			}
			if _, used := last[op.Output]; !used && op.Output != plan.Outputs[0] {
				delete(pool, op.Output)
			}
			record["residual_lazy_cache"] = lazy.summary()
			if bootLazy != nil {
				record["bootstrap_lazy_cache"] = bootLazy.summary()
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	cache.current = nil
	observe()
	record["plaintext_cache"] = cache.summary()
	out := pool[plan.Outputs[0]]
	if out == nil {
		return fmt.Errorf("final ciphertext missing")
	}
	count("final_decryption_attempts")
	if err := stage("final.decrypt_decode_compare", nil, func() error {
		pt := rlwe.NewDecryptor(p, sk).DecryptNew(out)
		count("final_decryption_completed")
		decoded := make([]float64, 1<<logSlots)
		if err := encoder.Decode(pt, decoded); err != nil {
			return err
		}
		raw, got, err := selectFinalOutput(decoded, reference.Indices, outputMultiplier)
		if err != nil { return err }
		comparison, err := compare(reference.Values, got)
		if err != nil {
			return err
		}
		record["comparison"] = comparison
		record["reference_scope"] = reference.Scope
		record["final_selected_values"] = got
		record["final_selected_raw_values"] = raw
		record["final_selected_scaled_values"] = got
		record["final_output_metadata"] = map[string]any{"level": out.Level(), "scale_log2": out.Scale.Log2(), "components": out.Degree() + 1, "logical_log_slots": out.LogDimensions.Cols}
		if comparison["passed"] != true {
			return fmt.Errorf("original CKKS comparison failed")
		}
		return nil
	}); err != nil {
		return err
	}
	record["encrypted_plan_completed"] = true
	return nil
}


// Resolve only filesystem paths. No content, digest or archive verification.
func canonicalSourcePlanPath(path string) (string, error) {
    absolute, err := filepath.Abs(path)
    if err != nil { return "", err }
    resolved, err := filepath.EvalSymlinks(absolute)
    if err != nil { return "", err }
    info, err := os.Stat(resolved)
    if err != nil { return "", err }
    if !info.Mode().IsRegular() { return "", fmt.Errorf("source plan path must name a regular file") }
    return filepath.Clean(resolved), nil
}

func sameSourcePlan(proofPath, actualPath string) bool {
    if !filepath.IsAbs(proofPath) { return false }
    proof, err := canonicalSourcePlanPath(proofPath)
    if err != nil { return false }
    actual, err := canonicalSourcePlanPath(actualPath)
    return err == nil && proof == actual
}

func authorizeOriginalSource(plan Plan, planPath string, fixture bool, sitePath, mathPath string) error {
    inline := map[string]bool{}
    for _, op := range plan.Initialization { if op.Payload.Kind == "inline" { inline[op.Output]=true } }
    if fixture {
        var target map[string]any; if err:=json.Unmarshal(plan.Target,&target);err!=nil{return err}
        if len(plan.Execution) > 12 || target["target_id"] != "dacapo-own-native-public-fixture" { return fmt.Errorf("explicit public scalar fixture is limited to12operations") }
        return nil
    }
    var sites struct {
        Status string `json:"status"`
        Factory string `json:"source_factory"`
        SourcePlan string `json:"source_plan"`
        Interpretation bool `json:"original_dsl_scalar_broadcast_interpretation"`
        Modified *bool `json:"original_plan_modified"`
        Sites []struct { ID string `json:"value_id"`; Interpretation string `json:"interpretation"`; Path string `json:"source_path"`; Line int `json:"source_line"` } `json:"sites"`
    }
    var reference struct {
        Status string `json:"status"`
        Factory string `json:"source_factory"`
        SourcePlan string `json:"source_plan"`
        Samples []string `json:"sample_ids"`
        Poly bool `json:"original_poly_preserved"`
        Packing bool `json:"original_pack_bb32_preserved"`
        Modified *bool `json:"original_plan_modified"`
        AllPassed bool `json:"all_three_passed"`
        Absolute float64 `json:"absolute_tolerance"`
        Relative float64 `json:"relative_tolerance"`
    }
    if sitePath=="" || mathPath=="" { return fmt.Errorf("original model scalar provenance and three-probe strict mathematical evidence are required") }
    if err:=readJSON(sitePath,&sites);err!=nil{return err}
    if err:=readJSON(mathPath,&reference);err!=nil{return err}
    if !sameSourcePlan(sites.SourcePlan, planPath) || !sameSourcePlan(reference.SourcePlan, planPath) {
        return fmt.Errorf("scalar and mathematical evidence must identify this original source plan")
    }
    factory:="pt_agent.experiments.r13_common_models:resnet20_silu"
    if sites.Status!="scalar_sites_verified" || sites.Factory!=factory || !sites.Interpretation || sites.Modified == nil || *sites.Modified || len(sites.Sites)!=len(inline) {
        return fmt.Errorf("scalar-site provenance is incomplete")
    }
    seen:=map[string]bool{}
    for _, site:=range sites.Sites {
        if !inline[site.ID] || seen[site.ID] || site.Interpretation!="full-slot-scalar" || site.Path=="" || site.Line<=0 {
            return fmt.Errorf("invalid original scalar site %s",site.ID)
        }
        seen[site.ID]=true
    }
    if reference.Status!="strict_math_passed" || reference.Factory!=factory || !reference.Poly || !reference.Packing || reference.Modified == nil || *reference.Modified || !reference.AllPassed || reference.Absolute!=1e-12 || reference.Relative!=1e-12 || len(reference.Samples)!=3 || reference.Samples[0]!="train:0" || reference.Samples[1]!="train:1" || reference.Samples[2]!="train:2" {
        return fmt.Errorf("three original Poly/physical32 mathematical probes have not passed")
    }
    return nil
}

func main() {
	mode := flag.String("mode", "parameters", "parameters, execute, cache-selftest or checkpoint-selftest")
	planPath := flag.String("plan", "", "original RuntimePlan JSON")
	bundle := flag.String("bundle", "", "original plaintext bundle directory")
	input := flag.String("input", "", "packed float64 input")
	expected := flag.String("expected", "", "final expected indices and values JSON")
	output := flag.String("output", "", "terminal JSON")
	events := flag.String("events", "", "native event JSONL")
	allowBoot := flag.Bool("allow-bootstrap", false, "explicitly execute native bootstrap ops")
    scalarFixture := flag.Bool("public-scalar-fixture", false, "explicit public fixture of at most12ops; no original-model acceptance")
    scalarProof := flag.String("scalar-site-proof", "", "independent original DSL scalar-site provenance JSON")
    sourceMath := flag.String("source-math-proof", "", "three original Poly/packed training probes strict mathematical evidence")
	cacheCapacity := flag.Int("bootstrap-cache", 1, "real bootstrap Galois LRU capacity")
	precision := flag.Int("precision-bits", 51, "original explicit DaCapo native residual prime bits51")
	s2cPrecision := flag.Int("s2c-bits", 0, "optional explicit three-layer SlotsToCoeffs precision50; zero preserves native default39")
	inverseDegree := flag.Int("mod1-inv-degree", 0, "optional explicit Mod1 inverse polynomial degree7 or15; zero preserves native default")
	polynomialDegree := flag.Int("mod1-degree", 0, "optional explicit Mod1 polynomial degree63; zero preserves native default30")
	messageRatio := flag.Int("log-message-ratio", 0, "optional explicit MR2 on precision50/inverse7; zero preserves native MR8")
	ssaCheck := flag.Bool("check-input-ssa", false, "observe input metadata preservation around every native operation")
	inputDivisor := flag.Int("bootstrap-input-divisor", 0, "legacy explicit2*NativeBoot(x/2) wrapper; zero preserves default")
	normalizationLog2 := flag.Int("bootstrap-normalization-log2", 0, "explicit dyadic value normalization log2(1..3); zero preserves default")
	keyTrace := flag.Bool("trace-key-requests", false, "record public Galois request order and cache decisions")
	checkpointInterval := flag.Int("checkpoint-interval", 1, "existing whole-record save requests per checkpoint:1..1000000; events retained and terminal/errors forced")
	flag.Parse()
	checkInputSSA = *ssaCheck
	traceKeyRequests = *keyTrace
	bootstrapInputDivisor = *inputDivisor
	if *normalizationLog2 > 0 && *normalizationLog2 <= 3 {
		bootstrapInputDivisor = 1 << *normalizationLog2
	}
	bootstrapCacheCapacity = *cacheCapacity
	precisionBits = *precision
	s2cPrecisionBits = *s2cPrecision
	mod1InverseDegree = *inverseDegree
	mod1PolynomialDegree = *polynomialDegree
	logMessageRatio = *messageRatio
	outputPath, eventPath = *output, *events
	record = map[string]any{"round": "DACAPO-NATIVE", "mode": *mode, "variant": variant, "status": "running", "active_stage": nil, "failed_stage": nil, "error": nil, "extra_integrity_checks": 0, "intermediate_decryptions": 0, "get_log_enabled": false, "keygen_api_attempts": 0, "keygen_api_completed": 0, "plaintext_encoding_attempts": 0, "plaintext_encoding_completed": 0, "encryption_attempts": 0, "encryption_completed": 0, "compute_attempts": 0, "compute_completed": 0, "bootstrap_attempts": 0, "bootstrap_completed": 0, "bootstrap_post_drop_level_operations": 0, "final_decryption_attempts": 0, "final_decryption_completed": 0, "dft_matrices_completed": 0, "max_nominal_scale_log2_deviation": 0.0, "security_acceptance": nil, "accuracy_acceptance": nil, "encrypted_plan_completed": false, "original_target_relabelled": false, "scale_policy": "Validate symbolic plan integer scale equations separately; retain native scale after division by actual Q primes and validate exact native Scale recurrence. No metadata reset to powers of two."}
	record["final_decoder_protocol"] = 1
	record["supported_final_output_multipliers"] = []int{1, 32}
	record["original_final_decoder"] = "original-physical32-first10"
	if bootstrapCacheCapacity != 1 {
		record["variant"] = fmt.Sprintf("dacapo-own-lattigo-n17-q55-%dx16-p61x3-scale%d-slots16-bootcache%d-residualcache1-plaintext1", precisionBits, precisionBits, bootstrapCacheCapacity)
	} else if precisionBits != 51 {
		record["variant"] = fmt.Sprintf("dacapo-own-lattigo-n17-q55-%dx16-p61x3-scale%d-slots16-lazy1", precisionBits, precisionBits)
	}
	if s2cPrecisionBits != 0 {
		record["variant"] = record["variant"].(string) + "-s2c50x3"
	}
	if mod1InverseDegree != 0 {
		record["variant"] = record["variant"].(string) + fmt.Sprintf("-mod1inv%d", mod1InverseDegree)
	}
	if mod1PolynomialDegree != 0 {
		record["variant"] = record["variant"].(string) + "-mod1degree63"
	}
	if logMessageRatio != 0 {
		record["variant"] = record["variant"].(string) + "-mr2"
	}
	if bootstrapInputDivisor != 0 {
		record["variant"] = record["variant"].(string) + fmt.Sprintf("-inputdiv%d", bootstrapInputDivisor)
	}
	record["precision_bits"] = precisionBits
	record["bootstrap_cache_capacity"] = bootstrapCacheCapacity
	record["bootstrap_observations"] = []map[string]any{}
	record["input_ssa_check_enabled"] = checkInputSSA
	record["input_ssa_observations"] = []map[string]any{}
	record["public_key_request_trace_enabled"] = traceKeyRequests
	record["bootstrap_input_value_divisions"] = 0
	record["bootstrap_output_integer_restores"] = 0
	record["bootstrap_input_divisor"] = bootstrapInputDivisor
	record["bootstrap_normalization_log2"] = *normalizationLog2
	record["native_bootstrap_api_attempts"] = 0
	record["native_bootstrap_api_completed"] = 0
	record["requested_checkpoint_interval"] = *checkpointInterval
	record["checkpoint_interruption_scope"] = "Events are appended at every original event point; all SSA rows remain in RAM and forced terminal/error checkpoints. Abrupt termination may lose whole-record updates since the latest checkpoint; interval counts save requests, not operations or seconds."
	if bootstrapInputDivisor != 0 {
		record["bootstrap_wrapper_contract"] = fmt.Sprintf("%d*NativeBoot(x/%d); temporary input copy Scale*=%d changes its decoded value; integerMul%d restores output value without consuming a level", bootstrapInputDivisor, bootstrapInputDivisor, bootstrapInputDivisor, bootstrapInputDivisor)
	}
	exit := 0
	defer func() {
		if v := recover(); v != nil {
			record["status"] = "error"
			record["error"] = fmt.Sprint(v)
			exit = 1
		}
		record["elapsed_ns"] = time.Since(origin).Nanoseconds()
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		record["go_memstats"] = map[string]uint64{"heap_alloc": m.HeapAlloc, "heap_sys": m.HeapSys, "sys": m.Sys, "total_alloc": m.TotalAlloc}
		save(true)
		if exit != 0 {
			os.Exit(exit)
		}
	}()
	if outputPath == "" || eventPath == "" {
		panic("explicit output/event paths required")
	}
	var checkpointErr error
	checkpoints, checkpointErr = newCheckpointSchedule(*checkpointInterval)
	if checkpointErr != nil {
		// Keep the valid default schedule so the terminal error can be saved.
		checkpoints, _ = newCheckpointSchedule(1)
		panic(checkpointErr)
	}
	if bootstrapCacheCapacity < 1 || bootstrapCacheCapacity > 48 {
		panic("bootstrap cache capacity must be between1 and48")
	}
	if precisionBits != 51 || s2cPrecisionBits != 0 || mod1InverseDegree != 0 || mod1PolynomialDegree != 0 || logMessageRatio != 0 || bootstrapInputDivisor != 0 {
		panic("Original DaCapo native binding preserves precision51 and native bootstrap defaults")
	}
	if bootstrapCacheCapacity != 1 { panic("Original DaCapo binding uses the existing R27 cache1 policy") }
	if s2cPrecisionBits != 0 && (s2cPrecisionBits != 50 || precisionBits != 50) {
		panic("S2C50 override requires explicit residual precision50")
	}
	if mod1InverseDegree != 0 && ((mod1InverseDegree != 7 && mod1InverseDegree != 15) || precisionBits != 50 || s2cPrecisionBits != 0) {
		panic("Inverse7/15 override requires residual precision50 with native S2C39")
	}
	if mod1PolynomialDegree != 0 && (mod1PolynomialDegree != 63 || precisionBits != 50 || s2cPrecisionBits != 0 || mod1InverseDegree != 0) {
		panic("Degree63 override changes only native Mod1 degree under residual precision50")
	}
	if logMessageRatio != 0 && (logMessageRatio != 2 || precisionBits != 50 || s2cPrecisionBits != 0 || (mod1InverseDegree != 7 && mod1InverseDegree != 15) || mod1PolynomialDegree != 0) {
		panic("MR2 override requires precision50/inverse7-or15 with native S2C39 and degree30")
	}
	if *normalizationLog2 < 0 || *normalizationLog2 > 3 || (*normalizationLog2 != 0 && *inputDivisor != 0) {
		panic("Explicit normalization log2 is0..3 and excludes the legacy divisor flag")
	}
	if *inputDivisor != 0 && *inputDivisor != 2 {
		panic("Legacy input divisor is0 or2")
	}
	if bootstrapInputDivisor != 0 && logMessageRatio != 2 {
		panic("Dyadic value normalization requires explicitMR2/inverse7-or15")
	}
	if *mode == "checkpoint-selftest" {
		if err := checkpointSelftest(); err != nil {
			panic(err)
		}
		record["status"] = "ok"
		return
	}
	if *mode == "cache-selftest" {
		if err := cacheSelftest(); err != nil {
			panic(err)
		}
		record["status"] = "ok"
		return
	}
	plan, values, encodes, _, last, err := loadPlan(*planPath, *bundle)
	if err != nil {
		panic(err)
	}
    record["inline_scalar_broadcasts"] = 0
    record["scalar_broadcast_interpretation"] = "OriginalDSLscalarbroadcast interpretation; distinct from the old Poseidon vector-encoder path"
    record["source_math_and_scalar_sites_verified"] = false
    if *mode == "execute" {
        if err := authorizeOriginalSource(plan, *planPath, *scalarFixture, *scalarProof, *sourceMath); err != nil { panic(err) }
        record["source_math_and_scalar_sites_verified"] = !*scalarFixture
        record["public_fixture_only"] = *scalarFixture
    }
	record["original_target"] = plan.Target
	p, bp, err := parameters()
	if err != nil {
		panic(err)
	}
	rotations := map[int]bool{}
	for _, op := range plan.Execution {
		if op.Op == "rotate" {
			step := (op.Attrs.Steps%(1<<logSlots) + (1 << logSlots)) % (1 << logSlots)
			if step != 0 {
				rotations[step] = true
			}
		}
	}
	steps := []int{}
	for step := range rotations {
		steps = append(steps, step)
	}
	sort.Ints(steps)
	residualGals := []uint64{}
	for _, step := range steps {
		residualGals = append(residualGals, p.GaloisElement(step))
	}
	bootstrapGals := bp.GaloisElements(bp.BootstrappingParameters)
	record["required_rotation_steps"] = steps
	record["residual_galois_elements"] = residualGals
	record["bootstrap_galois_elements"] = bootstrapGals
	record["evaluation_key_payload_estimate"] = map[string]any{"residual_one_key_bytes": payload(p), "bootstrap_one_key_bytes": payload(bp.BootstrappingParameters), "residual_resident_all_galois_bytes": int64(len(residualGals)) * payload(p), "bootstrap_resident_all_galois_bytes": int64(len(bootstrapGals)) * payload(bp.BootstrappingParameters), "lazy_two_cache_keys_bytes": payload(p) + payload(bp.BootstrappingParameters), "scope": "uint64 gadget coefficient payload only; relinearization/encapsulation keys, DFT/evaluator scratch and allocator overhead excluded", "bootstrap_dft_peak_memory_bytes": nil}
	save()
	if *mode == "execute" {
		err = execute(plan, values, encodes, last, *bundle, *input, *expected, p, bp, *allowBoot, *planPath, *sourceMath, *scalarFixture)
	} else if *mode != "parameters" {
		err = fmt.Errorf("unsupported mode")
	}
	if err != nil {
		record["status"] = "error"
		record["error"] = err.Error()
		exit = 1
	} else {
		record["status"] = "ok"
		record["active_stage"] = nil
	}
}
