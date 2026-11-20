// Appended to a private generated R31 client. No original runtime or SDK edits.
var preparedRoot string
var preparedTimed bool
var preparedBegin time.Time
var preparedKeygenNS, preparedEncodeNS, preparedWriteNS int64
var preparedBytes int64
var preparedKeygenAtStart, preparedEncodeAtStart int
var preparedPlainPaths = map[string]string{}
var preparedKeySets = map[string]bool{}
var preparedConstantsComplete bool

func preparedBump(name string) {
    n,_:=record[name].(int);record[name]=n+1
}
func preparedSilentStage(name string) bool {
    return preparedTimed || name=="input.encode_encrypt" || name=="runtime.compute" ||
        name=="final.decrypt_decode_compare" || name=="bootstrap.input_value_normalization" ||
        name=="bootstrap.output_integer_restore"
}
func preparedDiskAllowance() error {
    var s syscall.Statfs_t
    if err:=syscall.Statfs(preparedRoot,&s);err!=nil {return err}
    if uint64(s.Bavail)*uint64(s.Bsize)<10<<30 {return fmt.Errorf("resource_limit: prepared disk reserve10GiB")}
    if preparedBytes>36<<30 {return fmt.Errorf("resource_limit: prepared disk artifacts36GiB")}
    return nil
}
func preparedWriteFile(path string, object io.WriterTo) error {
    if preparedTimed {return fmt.Errorf("prepared store write forbidden during inference")}
    if err:=preparedDiskAllowance();err!=nil {return err}
    f,err:=os.OpenFile(path+".partial",os.O_WRONLY|os.O_CREATE|os.O_EXCL,0600)
    if err!=nil {return err}
    began:=time.Now()
    n,writeErr:=object.WriteTo(f)
    if writeErr==nil {writeErr=f.Sync()}
    closeErr:=f.Close()
    preparedWriteNS+=time.Since(began).Nanoseconds()
    if writeErr!=nil {return writeErr};if closeErr!=nil {return closeErr}
    if err=os.Rename(path+".partial",path);err!=nil {return err}
    preparedBytes+=n
    return preparedDiskAllowance()
}
func preparedWriteKeys(l *lazyKeys) error {
    if preparedTimed || preparedKeySets[l.name] {return fmt.Errorf("duplicate/late prepared keyset")}
    directory:=filepath.Join(preparedRoot,l.name)
    if err:=os.Mkdir(directory,0700);err!=nil {return err}
    for _,g:=range l.list {
        var key *rlwe.GaloisKey
        if err:=keygen(l.name+".galois.prepare",func(){key=l.kg.GenGaloisKeyNew(g,l.sk)});err!=nil {return err}
        if err:=preparedWriteFile(filepath.Join(directory,fmt.Sprintf("%d.bin",g)),key);err!=nil {return err}
        preparedBump("prepared_keys_generated")
        key=nil;runtime.GC()
    }
    l.kg=nil;l.sk=nil
    preparedKeySets[l.name]=true
    return nil
}
func preparedReadKey(l *lazyKeys,g uint64) (*rlwe.GaloisKey,error) {
    if !preparedKeySets[l.name] || !l.allowed[g] || l.kg!=nil || l.sk!=nil {
        return nil,fmt.Errorf("unknown or unsealed prepared key")
    }
    f,err:=os.Open(filepath.Join(preparedRoot,l.name,fmt.Sprintf("%d.bin",g)))
    if err!=nil {return nil,err};defer f.Close()
    key:=new(rlwe.GaloisKey)
    if _,err=key.ReadFrom(f);err!=nil {return nil,err}
    if key.GaloisElement!=g || key.LevelQ()!=l.p.MaxLevel() || key.LevelP()!=l.p.PCount()-1 {
        return nil,fmt.Errorf("prepared key public metadata differs")
    }
    return key,nil
}
func preparedWritePlaintexts(c *plaintextCache,init []Instruction) error {
    if preparedTimed || preparedConstantsComplete {return fmt.Errorf("duplicate/late constant preparation")}
    directory:=filepath.Join(preparedRoot,"plaintexts")
    if err:=os.Mkdir(directory,0700);err!=nil {return err}
    unique:=map[string]string{}
    for _,op:=range init {
        v,ok:=c.values[op.Output]
        if !ok || v.Kind!="plaintext" || op.Kind!="encode" {return fmt.Errorf("unknown encode value")}
        content,ok:=c.encodes[op.Output]
        if !ok {return fmt.Errorf("unknown encode payload")}
        // Existing locator string is an opaque identifier; no digest is calculated.
        k:=fmt.Sprintf("%d:%d:%s",v.Level,v.Scale,content)
        path,exists:=unique[k]
        if !exists {
            rawPath,err:=blobPath(c.bundle,content);if err!=nil {return err}
            raw,err:=readVector(rawPath);if err!=nil {return err}
            pt:=ckks.NewPlaintext(c.p,v.Level)
            pt.Scale=rlwe.NewScale(math.Exp2(float64(v.Scale)));pt.LogDimensions.Cols=logSlots
            count("plaintext_encoding_attempts");began:=time.Now()
            if err=c.encoder.Encode(raw,pt);err!=nil {return err}
            preparedEncodeNS+=time.Since(began).Nanoseconds();count("plaintext_encoding_completed")
            path=filepath.Join(directory,fmt.Sprintf("%d.bin",len(unique)))
            if err=preparedWriteFile(path,pt);err!=nil {return err}
            unique[k]=path;raw=nil;pt=nil
            if len(unique)%16==0 {runtime.GC()}
        }
        preparedPlainPaths[op.Output]=path
    }
    record["prepared_unique_plaintexts"]=len(unique)
    record["prepared_plaintext_value_ids"]=len(preparedPlainPaths)
    preparedConstantsComplete=true;runtime.GC()
    return nil
}
func preparedReadPlaintext(c *plaintextCache,id string) (*rlwe.Plaintext,error) {
    if !preparedConstantsComplete {return nil,fmt.Errorf("constants are not fully prepared")}
    path,ok:=preparedPlainPaths[id];if !ok {return nil,fmt.Errorf("unprepared plaintext value")}
    if c.current!=nil && c.name==id {c.hits++;return c.current,nil}
    if c.current!=nil {c.current=nil;c.evictions++}
    v,ok:=c.values[id];if !ok || v.Kind!="plaintext" {return nil,fmt.Errorf("unknown plaintext metadata")}
    f,err:=os.Open(path);if err!=nil {return nil,err};defer f.Close()
    pt:=new(rlwe.Plaintext)
    if _,err=pt.ReadFrom(f);err!=nil {return nil,err}
    if pt.Level()!=v.Level || !pt.IsNTT || pt.LogDimensions.Cols!=logSlots ||
        !pt.Scale.Equal(rlwe.NewScale(math.Exp2(float64(v.Scale)))) {return nil,fmt.Errorf("prepared plaintext public metadata differs")}
    c.current=pt;c.name=id;c.loads++;preparedBump("prepared_plaintext_loads")
    bytes:=int64(c.p.N())*int64(v.Level+1)*8;if bytes>c.peak {c.peak=bytes}
    return pt,nil
}
func preparedStartInference() error {
    if preparedTimed || !preparedConstantsComplete || !preparedKeySets["residual_lazy"] || !preparedKeySets["bootstrap_lazy"] {
        return fmt.Errorf("incomplete prepared key/constant stores")
    }
    preparedKeygenAtStart,_=record["keygen_api_completed"].(int)
    preparedEncodeAtStart,_=record["plaintext_encoding_completed"].(int)
    record["keygen_completed_at_inference_start"]=preparedKeygenAtStart
    record["constant_encodes_completed_at_inference_start"]=preparedEncodeAtStart
    record["prepared_key_generation_s"]=float64(preparedKeygenNS)/1e9
    record["prepared_constant_encoding_s"]=float64(preparedEncodeNS)/1e9
    record["prepared_store_writes_s"]=float64(preparedWriteNS)/1e9
    record["prepared_store_serialized_bytes"]=preparedBytes
    record["per_operation_recording_enabled"]=false
    record["prepared_inference_scope"]="input encryption through final decrypt/decode; includes bounded-cache loading of already generated keys/encoded constants from disk; excludes preparation, input encoding, quality comparison, and final persistence; OS page-cache state unforced"
    record["prepared_setup_before_encrypt_s"]=time.Since(origin).Seconds()
    save(true)
    preparedTimed=true;preparedBegin=time.Now()
    return nil
}
func preparedStopInference(completed bool) error {
    if !preparedTimed {return fmt.Errorf("inference clock not active")}
    elapsed:=time.Since(preparedBegin).Nanoseconds();preparedTimed=false
    record["ckks_inference_observed_ns"]=elapsed
    record["ckks_inference_completed"]=completed
    record["ckks_inference_s"]=nil
    keys,_:=record["keygen_api_completed"].(int)
    encodes,_:=record["plaintext_encoding_completed"].(int)
    record["keygen_completed_at_inference_end"]=keys
    record["constant_encodes_completed_at_inference_end"]=encodes
    unchanged:=keys==preparedKeygenAtStart && encodes==preparedEncodeAtStart
    record["no_keygen_or_constant_encoding_during_inference"]=unchanged
    if !unchanged {return fmt.Errorf("key generation or constant encoding occurred during inference")}
    if completed {record["ckks_inference_s"]=float64(elapsed)/1e9}
    return nil
}
