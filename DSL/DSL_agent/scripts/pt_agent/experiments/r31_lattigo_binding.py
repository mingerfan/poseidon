"""Explicit independent native precision bindings; original plan stays intact."""
import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeBinding:
    precision_bits:int=43
    s2c_bits:int|None=None
    mod1_inv_degree:int|None=None
    mod1_degree:int|None=None
    log_message_ratio:int|None=None
    bootstrap_input_divisor:int|None=None
    bootstrap_normalization_log2:int=0
    auxiliary_prime_bits:int=61
    c2s_bits:int|None=None

    def __post_init__(self):
        if type(self.auxiliary_prime_bits) is not int or self.auxiliary_prime_bits not in (57,61):
            raise ValueError('Explicit residual/bootstrap auxiliary primes use57 or61 bits')
        if type(self.precision_bits) is not int or self.precision_bits not in (43,50):
            raise ValueError('The explicit native precision binding is43 or50')
        if self.s2c_bits is not None and (type(self.s2c_bits) is not int or self.s2c_bits!=50 or self.precision_bits!=50):
            raise ValueError('The optional three-layer S2C50 binding requires residual precision50')
        if self.mod1_inv_degree is not None and (type(self.mod1_inv_degree) is not int or self.mod1_inv_degree not in (7,15) or self.precision_bits!=50 or self.s2c_bits is not None):
            raise ValueError('The optional inverse7/15 binding requires residual precision50 and native S2C39')
        if self.mod1_degree is not None and (type(self.mod1_degree) is not int or self.mod1_degree!=63 or self.precision_bits!=50 or self.s2c_bits is not None or self.mod1_inv_degree is not None):
            raise ValueError('The optional degree63 binding changes only native Mod1 degree under residual precision50')
        if self.log_message_ratio is not None and (type(self.log_message_ratio) is not int or self.log_message_ratio!=2 or self.precision_bits!=50 or self.mod1_inv_degree not in (7,15) or self.s2c_bits is not None or self.mod1_degree is not None):
            raise ValueError('The optional message-ratio2 binding changes only MR8 on the explicit50/inverse7/15 policy')
        if self.bootstrap_input_divisor is not None and (type(self.bootstrap_input_divisor) is not int or self.bootstrap_input_divisor!=2 or self.log_message_ratio!=2):
            raise ValueError('The optional divisor2 is the explicit2*Boot(x/2) MR2/inverse7 wrapper')

        if type(self.bootstrap_normalization_log2) is not int or self.bootstrap_normalization_log2 not in (0,1,2,3):
            raise ValueError('Explicit log2 value normalization is0 through3')
        if self.bootstrap_normalization_log2 and (self.bootstrap_input_divisor is not None or self.log_message_ratio!=2):
            raise ValueError('Explicit log2 normalization requires MR2 and excludes the legacy divisor option')
        if self.c2s_bits is not None and (type(self.c2s_bits) is not int or self.c2s_bits!=52
                or self.precision_bits!=50 or self.mod1_inv_degree!=15 or self.log_message_ratio!=2
                or self.bootstrap_normalization_log2!=3 or self.bootstrap_input_divisor is not None
                or self.auxiliary_prime_bits!=61 or self.s2c_bits is not None or self.mod1_degree is not None):
            raise ValueError('Explicit C2S52x4 requires precision50/Inv15/MR2/div8/P61 with native S2C39')

    @property
    def value_divisor(self):
        return (1<<self.bootstrap_normalization_log2) if self.bootstrap_normalization_log2 else (self.bootstrap_input_divisor or 0)

    def variant(self,capacity=1):
        prefix='lattigo-n16-q55-'+str(self.precision_bits)+'x8-p'+str(self.auxiliary_prime_bits)+'x3-scale'+str(self.precision_bits)+'-logicalslots14-'
        value=prefix+('lazy1' if capacity==1 else 'bootcache'+str(capacity)+'-residualcache1-plaintext1')
        if self.s2c_bits is not None:value+='-s2c50x3'
        if self.mod1_inv_degree is not None:value+='-mod1inv'+str(self.mod1_inv_degree)
        if self.mod1_degree is not None:value+='-mod1degree63'
        if self.log_message_ratio is not None:value+='-mr2'
        if self.value_divisor:value+='-inputdiv'+str(self.value_divisor)
        if self.auxiliary_prime_bits!=61:value+='-bootp'+str(self.auxiliary_prime_bits)+'x4'
        if self.c2s_bits is not None:value+='-c2s52x4'
        return value

    def metadata(self):
        value=dict(log_n=16,log_q=[55]+[self.precision_bits]*8,log_p=[self.auxiliary_prime_bits]*3,
            log_default_scale=self.precision_bits,logical_log_slots=14,bootstrap_log_p=[self.auxiliary_prime_bits]*4,
            original_compiler_parameters_unchanged=True,security_acceptance=None)
        if self.s2c_bits is not None:
            value['bootstrap_slots_to_coeffs_factorization_depth_and_log_scales']=[[self.s2c_bits] for _ in range(3)]
        if self.c2s_bits is not None:
            value['bootstrap_coeffs_to_slots_factorization_depth_and_log_scales']=[[self.c2s_bits] for _ in range(4)]
        if self.mod1_inv_degree is not None:value['bootstrap_mod1_inv_degree']=self.mod1_inv_degree
        if self.mod1_degree is not None:value['bootstrap_mod1_degree']=self.mod1_degree
        if self.log_message_ratio is not None:value['bootstrap_log_message_ratio']=self.log_message_ratio
        if self.value_divisor:
            divisor=self.value_divisor
            value['bootstrap_input_value_divisor']=divisor
            value['bootstrap_output_value_multiplier']=divisor
            value['bootstrap_wrapper_contract']=f'{divisor}*NativeBoot(x/{divisor}); temporary copy Scale*={divisor} changes decoded value; integerMul{divisor} restores output value'
            value['bootstrap_normalization_log2']=self.bootstrap_normalization_log2 or 1
        return value


def dft_static_inventory(parameters):
    """Integer-index projection of native dft.go:544–665 allocation shapes.

    No matrix coefficients, keys, or ciphertexts are constructed. Native
    NewMatrixFromLiteral uses its literal LevelQ for every allocated diagonal;
    its local decreasing level controls scales rather than allocation levels.
    """
    import math
    n=1<<parameters['BootstrappingParameters']['LogN'];rows=[]
    for name in ('CoeffsToSlotsParameters','SlotsToCoeffsParameters'):
        d=parameters[name];log=d['LogSlots'];slots=1<<log
        encode=d['Type']==0;repack=d['Format']==2;reverse=d['BitReversed']
        depth=sum(d['Levels']);remaining=log;merge=[0]*depth
        for i in range(depth):
            group=math.ceil(remaining/(depth-i));merge[i if encode else depth-i-1]=group;remaining-=group
        def rotation(level):return 1<<(level-1) if encode!=reverse else 1<<(log-level)
        def advance(indices,level,period):
            step=rotation(level)&(period-1)
            return {k for i in indices for k in (i,(i+step)&(period-1),(i-step)&(period-1))}
        level=log;counts=[]
        for i,group in enumerate(merge):
            special=log<parameters['BootstrappingParameters']['LogN']-1 and not encode and i==0 and repack
            period=slots*2 if special else slots
            if special:indices=advance({0,slots},level,period)
            else:
                step=rotation(level);indices={0,step,slots-step}
            for offset in range(1,group):indices=advance(indices,level-offset,period)
            counts.append(len(indices));level-=group
        per_diagonal=n*(d['LevelQ']+1+d['LevelP']+1)*8
        rows.append(dict(transform=name,diagonals_per_matrix=counts,
            literal_level_q=d['LevelQ'],literal_level_p=d['LevelP'],
            coefficient_bytes_per_diagonal=per_diagonal,
            rns_coefficient_payload_bytes=sum(counts)*per_diagonal))
    return dict(matrices=rows,total_rns_coefficient_payload_bytes=sum(r['rns_coefficient_payload_bytes'] for r in rows),
        scope='Static uint64 RNS matrix coefficients only; excludes maps, encoder/DFT scratch, keys, evaluator buffers and allocator overhead',
        source='native dft/dft.go:544-665 index maps and :179-203 allocation; common/lintrans/lintrans.go:150-180')


def map_precision_plan(original,binding,*,bootstrap_cache=48):
    """Map only symbolic scales and the explicitly derived native target.

    Shapes, levels, kinds, components, identifiers, instruction order, attrs
    unrelated to target scales, input/output lists and bundle strings survive.
    Raw plaintext values and the frozen mathematical function stay unchanged.
    """
    mapped=copy.deepcopy(original);unit=binding.precision_bits;changes=[]
    def convert(value):
        if type(value) is not int or value not in (43,86):raise ValueError('Source symbolic scale must be43 or86')
        return value//43*unit
    for index,value in enumerate(mapped['values']):
        old=value['scale_log2'];value['scale_log2']=convert(old)
        changes.append(dict(path='values['+str(index)+'].scale_log2',source=old,derived=value['scale_log2']))
    for phase in ('initialization','execution','finalization'):
        for index,instruction in enumerate(mapped[phase]):
            attrs=instruction.get('attrs') or {}
            if 'target_scale_log2' in attrs:
                old=attrs['target_scale_log2'];attrs['target_scale_log2']=convert(old)
                changes.append(dict(path=phase+'['+str(index)+'].attrs.target_scale_log2',source=old,derived=attrs['target_scale_log2']))
    mapped['target']=dict(target_id='pt-agent-lattigo-native-adapter',adapter_variant=binding.variant(bootstrap_cache),
        precision_bits=unit,source_original_target=copy.deepcopy(original['target']),
        source_target_is_provenance_only=True,concrete_prime_binding_verified=False)
    return mapped,dict(source_scales=[43,86],derived_scales=[unit,unit*2],changes=changes,
        native_binding=binding.metadata(),source_original_target=copy.deepcopy(original['target']),
        raw_data_modified=False,frozen_program_modified=False,levels_modified=False,
        original_context_ids_preserved_as_opaque_identifiers=True,
        interpretation='Explicit independent native adapter variant; original compiler profile and raw RuntimePlan remain unchanged',
        mathematical_equations=dict(multiply='unit+unit=2unit',
            rescale='Actual native scale divided by the actual selected prime; symbolic2unit becomesunit',
            add='Same native integer-ratio/max-scale rule; scales50 have independently constructed Q primes',
            bootstrap=((f'Explicit{binding.value_divisor}*NativeBoot(x/{binding.value_divisor}) wrapper with inverse{binding.mod1_inv_degree} and message-ratio2' if binding.value_divisor else
                f'Explicit native inverse{binding.mod1_inv_degree} polynomial policy with message-ratio2' if binding.log_message_ratio is not None else
                f'Explicit native inverse{binding.mod1_inv_degree} polynomial policy' if binding.mod1_inv_degree is not None else
                'Explicit native degree63 Mod1 polynomial policy' if binding.mod1_degree is not None else 'Native default algorithm')
                +', external level1 to target5; native result dropped from8 to5')),
        security_acceptance=None,accuracy_acceptance=None)
