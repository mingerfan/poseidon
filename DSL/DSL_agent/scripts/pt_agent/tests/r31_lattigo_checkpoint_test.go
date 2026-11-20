// Copy this file beside the trusted main.go in a new isolated module before
// running the tests. These tests use only maps, temporary files and event JSON;
// no Lattigo constructor, key, evaluator or ciphertext is called.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func checkpointFixture(t *testing.T, interval int) string {
	t.Helper()
	originalRecord, originalOutput, originalEvents := record, outputPath, eventPath
	originalCheckpoints, originalSequence, originalParents := checkpoints, sequence, parents
	directory := t.TempDir()
	record = map[string]any{"status": "running", "failed_stage": nil, "probe": 0,
		"keygen_api_attempts": 0, "encryption_attempts": 0,
		"compute_attempts": 0, "final_decryption_attempts": 0}
	outputPath, eventPath = filepath.Join(directory, "result.json"), filepath.Join(directory, "events.jsonl")
	var err error
	checkpoints, err = newCheckpointSchedule(interval)
	if err != nil {
		t.Fatal(err)
	}
	sequence, parents = 0, nil
	t.Cleanup(func() {
		record, outputPath, eventPath = originalRecord, originalOutput, originalEvents
		checkpoints, sequence, parents = originalCheckpoints, originalSequence, originalParents
	})
	return directory
}

func checkpointRecord(t *testing.T) map[string]any {
	t.Helper()
	var observed map[string]any
	if err := readJSON(outputPath, &observed); err != nil {
		t.Fatal(err)
	}
	return observed
}

func TestCheckpointScheduleRejectsInvalidInterval(t *testing.T) {
	for _, interval := range []int{-1, 0, maxCheckpointInterval + 1} {
		if _, err := newCheckpointSchedule(interval); err == nil {
			t.Fatalf("accepted interval %d", interval)
		}
	}
	for _, interval := range []int{1, 1000, maxCheckpointInterval} {
		if schedule, err := newCheckpointSchedule(interval); err != nil || schedule.Interval != interval {
			t.Fatalf("rejected valid interval %d: %v", interval, err)
		}
	}
}

func TestCheckpointDefaultSavesEveryOriginalRequest(t *testing.T) {
	checkpointFixture(t, 1)
	for i := 1; i <= 4; i++ {
		record["probe"] = i
		save()
		if checkpointRecord(t)["probe"] != float64(i) {
			t.Fatal("default request was skipped")
		}
	}
	if checkpoints.Requests != 4 || checkpoints.WritesCompleted != 4 || checkpoints.SkippedRequests != 0 {
		t.Fatalf("default counters differ: %+v", checkpoints)
	}
}

func TestCheckpointIntervalCountsRequestsAndForcesTerminal(t *testing.T) {
	checkpointFixture(t, 3)
	for i := 1; i <= 5; i++ {
		record["probe"] = i
		save()
	}
	if checkpointRecord(t)["probe"] != float64(3) {
		t.Fatal("periodic snapshot differs")
	}
	record["status"] = "ok"
	save(true)
	observed := checkpointRecord(t)
	if observed["probe"] != float64(5) || observed["status"] != "ok" {
		t.Fatal("terminal values were not flushed")
	}
	if checkpoints.Requests != 6 || checkpoints.WritesCompleted != 2 || checkpoints.SkippedRequests != 4 || checkpoints.ForcedRequests != 1 {
		t.Fatalf("interval counters differ: %+v", checkpoints)
	}
	policy := observed["checkpoint_policy"].(map[string]any)
	if policy["writes_completed"] != float64(2) || policy["requests"] != float64(6) || policy["unit"] != "existing_save_request" {
		t.Fatal("persisted policy differs from completed writes")
	}
}

func TestCheckpointStageErrorPersistsAndKeepsEveryEvent(t *testing.T) {
	checkpointFixture(t, 1000)
	if err := stage("small.first", nil, func() error { record["probe"] = 1; return nil }); err != nil {
		t.Fatal(err)
	}
	if err := stage("small.failure", nil, func() error { record["probe"] = 7; return fmt.Errorf("expected fixture failure") }); err == nil {
		t.Fatal("error was lost")
	}
	observed := checkpointRecord(t)
	if observed["probe"] != float64(7) || observed["failed_stage"] != "small.failure" || checkpoints.ForcedRequests != 1 {
		t.Fatal("error did not force latest full record")
	}
	data, err := os.ReadFile(eventPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected four original events, got %d", len(lines))
	}
	for index, line := range lines {
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatal(err)
		}
		if event["type"] != []string{"begin", "end", "begin", "end"}[index] {
			t.Fatal("original event order changed")
		}
	}
}

func TestCheckpointPanicForcesErrorAndRetainsSSA(t *testing.T) {
	checkpointFixture(t, 1000)
	record["input_ssa_observations"] = []map[string]any{}
	if err := stage("small.panic", nil, func() error {
		for i := 0; i < 3; i++ {
			record["input_ssa_observations"] = append(record["input_ssa_observations"].([]map[string]any), map[string]any{"ordinal": i, "metadata_preserved": true})
			save()
		}
		panic("expected fixture panic")
	}); err == nil {
		t.Fatal("panic was not returned as a stage error")
	}
	observed := checkpointRecord(t)
	if len(observed["input_ssa_observations"].([]any)) != 3 || observed["failed_stage"] != "small.panic" {
		t.Fatal("forced error lost SSA rows")
	}
	if checkpoints.WritesCompleted != 1 || checkpoints.SkippedRequests != 4 {
		t.Fatalf("panic checkpoint counters differ: %+v", checkpoints)
	}
}

func TestCheckpointWriteFailureCountsAndCanFlushError(t *testing.T) {
	directory := checkpointFixture(t, 1000)
	outputPath = filepath.Join(directory, "absent", "result.json")
	failed := false
	func() {
		defer func() { failed = recover() != nil }()
		save(true)
	}()
	if !failed || checkpoints.WriteAttempts != 1 || checkpoints.WritesCompleted != 0 || checkpoints.WriteErrors != 1 {
		t.Fatalf("write failure counters differ: %+v", checkpoints)
	}
	outputPath = filepath.Join(directory, "recovered.json")
	record["status"], record["error"] = "error", "preserved fixture write failure"
	save(true)
	observed := checkpointRecord(t)
	if observed["status"] != "error" || observed["error"] != "preserved fixture write failure" {
		t.Fatal("error flush did not persist")
	}
	policy := observed["checkpoint_policy"].(map[string]any)
	if policy["write_errors"] != float64(1) || policy["writes_completed"] != float64(1) || policy["write_attempts"] != float64(2) {
		t.Fatal("error flush counters differ")
	}
}

func TestCheckpointKeyFreeSelftestRestoresOuterState(t *testing.T) {
	checkpointFixture(t, 1000)
	originalOutput, originalEvents, originalSchedule := outputPath, eventPath, checkpoints
	record["probe"] = 123
	if err := checkpointSelftest(); err != nil {
		t.Fatal(err)
	}
	if record["probe"] != 123 || outputPath != originalOutput || eventPath != originalEvents || checkpoints != originalSchedule {
		t.Fatal("selftest changed outer state")
	}
	if record["checkpoint_selftest_passed"] != true || record["checkpoint_selftest_crypto_calls"] != 0 {
		t.Fatal("selftest acceptance missing")
	}
	for _, name := range []string{"keygen_api_attempts", "encryption_attempts", "compute_attempts", "final_decryption_attempts"} {
		if record[name] != 0 {
			t.Fatalf("selftest invoked %s", name)
		}
	}
}
