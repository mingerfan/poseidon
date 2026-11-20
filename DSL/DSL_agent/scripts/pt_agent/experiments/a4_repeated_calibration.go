// Appended only to a fresh private calibration source. No per-op disk writes.
type calibrationTarget struct {
    Cell int `json:"cell_index"`
    Iteration int `json:"iteration"`
    Ordinal int `json:"target_ordinal"`
}
var calibrationTargets=map[int]calibrationTarget{}
var calibrationSamples=[]map[string]any{}
var calibrationOperationBegin time.Time

func calibrationLoadTargets(path string) error {
    if path=="" {return fmt.Errorf("explicit calibration metadata required")}
    var data struct {Targets []calibrationTarget `json:"target_operations"`}
    if err:=readJSON(path,&data);err!=nil {return err}
    if len(data.Targets)!=200 {return fmt.Errorf("require200 explicit repeated targets")}
    counts:=map[int]int{}
    for _,target:=range data.Targets {
        if _,ok:=calibrationTargets[target.Ordinal];ok {return fmt.Errorf("duplicate calibration target")}
        calibrationTargets[target.Ordinal]=target;counts[target.Cell]++
    }
    five,ten:=0,0
    for _,count:=range counts {if count==5 {five++} else if count==10 {ten++} else {return fmt.Errorf("target repetition differs")}}
    if five!=38 || ten!=1 {return fmt.Errorf("require38 ordinary cells and one bootstrap cell")}
    return nil
}
func calibrationStage(metadata map[string]any,fn func() error) (err error) {
    began:=time.Now()
    ordinal,ok:=metadata["ordinal"].(int)
    if !ok {return fmt.Errorf("missing operation ordinal")}
    target,measured:=calibrationTargets[ordinal]
    defer func(){
        if r:=recover();r!=nil {err=fmt.Errorf("panic: %v",r)}
        if measured {
            ended:=time.Now()
            calibrationSamples=append(calibrationSamples,map[string]any{
                "ordinal":ordinal,"cell_index":target.Cell,"iteration":target.Iteration,
                "operation_ns":ended.Sub(calibrationOperationBegin).Nanoseconds(),
                "compute_ns":ended.Sub(began).Nanoseconds(),"completed":err==nil})
        }
    }()
    return fn()
}
func calibrationPublish() {
    record["calibration_observations"]=calibrationSamples
    record["calibration_target_count"]=len(calibrationSamples)
    record["calibration_persistence_during_inference"]=false
    record["calibration_scope"]="Fresh prepared-disk bounded-cache public target samples; all target durations stored in RAM and persisted after final decrypt; operation includes plaintext lookup and key loads; compute excludes pre-stage plaintext lookup"
}
