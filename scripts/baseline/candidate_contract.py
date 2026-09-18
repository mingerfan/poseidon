"""Provider-neutral data boundary. No model API, tracing, eval or exec here."""
import copy
import hashlib
import json
import math

from hecate_contract import CONTRACT_ROTATIONS, validate_function
from seal_artifact_gate import require
from spatial_ops import INPUT_SHAPES
from cipher_abi import has_zero_argument, physical_input_names
from native_function_rules import TASK as NATIVE_TASK, CONTRACT as NATIVE_CONTRACT, RULES as NATIVE_RULES
from native_function_rules import EXERCISE_TASK as NATIVE_EXERCISE_TASK, EXERCISE_RULES as NATIVE_EXERCISE_RULES
from native_function_rules import ARRAY_TASK as NATIVE_ARRAY_TASK, ARRAY_CONTRACT as NATIVE_ARRAY_CONTRACT, ARRAY_RULES as NATIVE_ARRAY_RULES
from native_function_rules import ARRAY_EXERCISE_TASK, ARRAY_EXERCISE_RULES
from native_function_rules import STAR_TASK, STAR_CONTRACT, STAR_RULES
from native_function_rules import ARITHMETIC_TASK, ARITHMETIC_CONTRACT, ARITHMETIC_RULES
from native_function_rules import LOOP_TASK, LOOP_CONTRACT, LOOP_RULES
from native_function_rules import AUGMENTED_TASK, AUGMENTED_CONTRACT, AUGMENTED_RULES
from native_function_rules import MUTATION_TASK, MUTATION_CONTRACT, MUTATION_RULES
from native_function_rules import STAR_EXERCISE_TASK, STAR_EXERCISE_RULES

MAX_BYTES = 131072
MAX_REPAIRS = 3
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["schema", "request_id", "hecate_source"],
    "properties": {"schema": {"type": "integer", "enum": [1]},
                   "request_id": {"type": "string"}, "hecate_source": {"type": "string"}},
}
RULES = """Generate one @hc.func(\"c\") def golden(x) Hecate function.
Only SSA single-name assignments and a final ciphertext or list return.
Allowed expressions: ciphertext + ciphertext/public; ciphertext * ciphertext/public;
ciphertext.rotate(1) or rotate(2). Positive rotation is left rotation.
Only use the supplied public constant names. No imports, calls to other functions,
attributes other than rotate, literals in arithmetic, control flow or mutation.
Input: period-4 repeated packed slots, encrypted; weights are public and fixed.
Return layout/selectors are fixed by the harness, not chosen by the candidate.
Packed dot product requires true cross-slot reduction. A ciphertext return list
means multiple ciphertexts, not slots of one ciphertext. No bootstrap is allowed.
Compiler owns scale/level/rescale/relinearization; do not emit runtime operations.
The program must implement the supplied model graph, not fit hidden test outputs.
Schema/AST acceptance is not semantic proof: compile, encrypt/evaluate/decrypt and
independent numerical comparison must pass. Reference, inputs, keys, tolerance,
security parameters and compiler settings are never writable by the generator.
"""



RULES_V1 = RULES.replace(
    "Allowed expressions: ciphertext + ciphertext/public; ciphertext * ciphertext/public;",
    "Allowed expressions: ciphertext + ciphertext/public; ciphertext * ciphertext/public;\n"
    "ciphertext - ciphertext/public; unary -ciphertext. Subtraction is ordered, not commutative;")
RULES_V2 = RULES_V1.replace("ciphertext.rotate(1) or rotate(2).",
                          "ciphertext.rotate(step), step must be one of -3,-2,-1,1,2,3.")
RULES_V3 = RULES_V2.replace('Generate one @hc.func("c") def golden(x) Hecate function.',
    'Generate one Hecate golden function. Use layout.inputs in its exact declared order.\n'
    'Each entry maps a user input name to its DSL parameter dsl_name and logical shape.\n'
    'The signature uses those dsl_name values; decorator is c,c for two encrypted inputs,\n'
    'c,c,c for three, or c,c,c,c for four. Every input is separately encrypted;\n'
    'never alias inputs or infer order from dictionary keys. Each has period-4 packing.\n'
    'Multidimensional logical inputs have C-row-major slot order; plaintext reference\n'
    'flattening is a view, not a ciphertext computation. Do not add Python reshape calls.')
RULES_V4 = RULES_V3.replace(
    'The signature uses those dsl_name values; decorator is c,c for two encrypted inputs,\n'
    'c,c,c for three, or c,c,c,c for four. Every input is separately encrypted;\n',
    'Use exactly one of these Python headers, selected by input count:\n'
    'Two inputs: @hc.func("c,c") followed by def golden(x, y):\n'
    'Three inputs: @hc.func("c,c,c") followed by def golden(x, y, z):\n'
    'Four inputs: @hc.func("c,c,c,c") followed by def golden(x, y, z, t):\n'
    'Put the decorator and def on separate lines; indent the function body.\n'
    'hc.func receives ONE comma-separated STRING argument, not separate arguments.\n'
    'Never use @c(...), @hc.func("c", "c"), @hc.func(c,c), a bare c,c, or omit the decorator.\n'
    'Function name is exactly golden, not the model id. Use DSL x/y/z/t parameters,\n'
    'not user input names. No parameter/return annotations, defaults, docstrings or type comments.\n'
    'Every input is separately encrypted;\n')
# v4 clarifies the generation protocol, not the accepted DSL semantics.
# Keep v3 immutable so historical requests/hashes remain verifiable.
TASK_RULES = {"hecate-function-synthesis-v0": ("hecate-function-v0", RULES),
              "hecate-function-synthesis-v1": ("hecate-function-v1", RULES_V1),
              "hecate-function-synthesis-v2": ("hecate-function-v2", RULES_V2),
              "hecate-function-synthesis-v3": ("hecate-function-v3", RULES_V3),
              "hecate-function-synthesis-v4": ("hecate-function-v3", RULES_V4)}

# An explicit new execution ABI, not a change to any historical task's rules.
RULES_V5 = ('Generate one Hecate golden function. Logical encrypted inputs are x, y, z, t '
    'in layout.inputs order, or just x when layout has input_shape. Append zero_ct as the final '
    'encrypted parameter. Use @hc.func with one comma-separated c for each physical parameter. '
    'Example for one logical input: @hc.func("c,c") followed by def golden(x, zero_ct):. '
    'There are 1..4 logical inputs plus exactly one zero_ct. zero_ct is freshly encrypted zero '
    'supplied by the trusted client, not plaintext, not a model input, and not a secret key. '
    'Use it for zero-weight rows or exact public-zero products; do not manufacture zero as x-x or x*0. '
    'Return selectors refer to actual ciphertext results, including zero results.\n' +
    RULES_V2[RULES_V2.index('Only SSA'):])
TASK_RULES['hecate-function-synthesis-v5'] = ('hecate-function-v4', RULES_V5)

# Arithmetic grammar expansion is opt-in; historical requests remain immutable.
RULES_V6 = '''Generate one Hecate golden function. Use logical inputs x/y/z/t in
layout.inputs order, or x for layout.input_shape. If layout.auxiliary_ciphertexts
declares zero_ct, append it as the last parameter; otherwise do not add it.
Use @hc.func with ONE comma-separated string containing c for each parameter.
Example: @hc.func("c,c") followed on the next line by def golden(x, y):.
Use straight-line single-name assignments and a final ciphertext or list return.
Ciphertext names may be rebound with =, +=, -=, *=. Evaluate the right side using
the old bindings, then bind the new Expr to the target. Existing aliases keep
their old value: a=x; x-=w leaves a unchanged. Public constants and zero_ct are
read-only bindings. zero_ct, if provided, is trusted client-encrypted zero.
Allowed arithmetic is +, -, * with at least one ciphertext operand, including
public+ciphertext, public-ciphertext, public*ciphertext; subtraction is ordered.
Unary -ciphertext and ciphertext.rotate(step) are allowed for steps -3,-2,-1,1,2,3.
Positive rotation is left rotation. Named public scalars/length-one arrays broadcast;
length-four arrays are period-four vectors. The harness binds them as Hecate Plain.
Only supplied public names; no arithmetic literals or public-only arithmetic.
No imports, arbitrary calls, indexing, containers other than the final return list,
control flow, attribute access except rotate, annotations or default arguments.
All encrypted inputs have repeated period-four packing in C-row-major logical order.
Follow the fixed output_ciphertexts and output_selectors; a return list contains
ciphertexts, not slots. Dot products require actual cross-slot reduction.
Do not manufacture zero with x-x or x*0. Use zero_ct only if declared by the harness.
Compiler owns scale/level/rescale/modswitch/relinearization. No bootstrap or runtime
operations. Do not alter reference, tests, weights, security parameters or tolerance.
Implement the supplied graph, not hidden outputs. Static acceptance is not proof;
real compile and encrypted execution must pass independent numerical comparison.
'''
TASK_RULES['hecate-function-synthesis-v6'] = ('hecate-function-v5', RULES_V6)

from chunked_input_abi import TASK as CHUNKED_TASK, validate_request_binding
CHUNKED_RULES = RULES_V6 + '''\nModel schema 4 describes ONE original logical input of 5..16 elements.
layout.model_input_binding is the authoritative client-side input mapping:
flatten the original input in C order, partition consecutive groups of four,
pad only the last group with zero, then repeat each group across CKKS slots.
layout.inputs names chunk0..chunk3 map to DSL parameters x/y/z/t, NOT independent
user model inputs. Evaluate the original model using all its logical elements.
For Linear, each chunk contributes its own four-element weighted reduction;
sum chunk contributions, then add the original bias ONCE. Padding columns must
not contribute. No Python indexing of ciphertext slots is available.
The fx_graph is the trusted physical decomposition; it is not the independent
plaintext reference. Public constants include block weights derived from the
original public model. Output layout remains 1..4 scalar-neuron ciphertexts.
'''
TASK_RULES[CHUNKED_TASK] = ('hecate-function-v5', CHUNKED_RULES)

from packed_input_abi import TASK as PACKED_TASK, CONTRACT as PACKED_CONTRACT
from packed_input_abi import NATIVE_TASK as PACKED_NATIVE_TASK, NATIVE_CONTRACT as PACKED_NATIVE_CONTRACT
from packed_input_abi import NATIVE_EXERCISE_TASK as PACKED_NATIVE_EXERCISE_TASK, TASKS as PACKED_TASKS, NATIVE_TASKS as PACKED_NATIVE_TASKS
from native_function_rules import PACKED_NATIVE_RULES
PACKED_RULES = '''Generate one @hc.func("c") def golden(x) Hecate function.
If layout.auxiliary_ciphertexts declares zero_ct, use @hc.func("c,c") and
def golden(x, zero_ct) instead. The trusted client freshly encrypts zero_ct.
Model schema5 describes ONE encrypted logical input, with public fixed weights.
layout.model_input_binding specifies its C-row-major flattening, logical length,
zero padding, and power-of-two slot_period P (4..256). The client repeats that
P-element vector across all16384 CKKS slots. Do not assume period4 or16.
Public constants are Hecate Plain scalar/length1 or lengthP periodic vectors.
Use only supplied constant names, no arithmetic literals or new constants.
Allowed: straight-line name assignments, rebinding, +=/-=/*=, +,-,* with at
least one ciphertext operand (public-left supported), unary -ciphertext,
and ciphertext.rotate(step). step is a positive power of two less than P;
positive rotation is left. No imports, arbitrary calls, indexing, loops,
attributes except rotate, public-only arithmetic, annotations or defaults.
Ciphertext aliases keep the old value on rebinding. zero_ct is read-only.
Linear needs real cross-slot reduction over the full period P with padded
weight columns zero. Repeatedly use sum += sum.rotate(1), then2,4,...P/2.
Public bias is added ONCE after reduction. Squaring happens at the original
graph location. Do not confuse input slots with output ciphertext list entries.
Follow fixed output_representation, output_ciphertexts, output_selectors and
output_shape. packed_prefix returns one ciphertext; scalar_neurons returns
one broadcast ciphertext per output, as a list when multiple outputs exist.
No repacking, bootstrap, scale/rescale/level manipulation or decrypt operations.
Compiler owns CKKS lowering. Do not manufacture zero with x-x or x*0; use
zero_ct when provided. No training, approximations or security changes.
Limits:1024 arithmetic/rotation operations,12288 AST nodes,65536 source bytes.
Use temporaries to avoid repeating expressions in the source AST.
Implement the full original model, never fit hidden test outputs. Parse/compile
success is not correctness: real encrypted execution must pass independent
reference comparison. Inputs, reference and tolerances are not generator inputs.
'''
TASK_RULES[PACKED_TASK]=(PACKED_CONTRACT,PACKED_RULES)
TASK_RULES[PACKED_NATIVE_TASK]=(PACKED_NATIVE_CONTRACT,PACKED_NATIVE_RULES)
TASK_RULES[PACKED_NATIVE_EXERCISE_TASK]=(PACKED_NATIVE_CONTRACT,PACKED_NATIVE_RULES+'''
This is a constrained construction exercise. Implement construction_exercise
using contributing, reachable computation, not unused helpers or cancelled terms.
Both finite public influence probes and real Hecate trace witnesses must pass,
in addition to original-model encrypted comparison. Do not alter the exercise.
Golden has ONLY encrypted input parameters: @hc.func("c") def golden(x), or
@hc.func("c,c") def golden(x,zero_ct) when declared. Public constants stay global;
do not add c0/c1 parameters to golden. Helpers may use noncolliding p parameters.
Rotation syntax is value.rotate(step), NEVER rotate(value,step) or hc.rotate(...).
''')

RULES_V7 = RULES_V6.replace(
    'Use straight-line single-name assignments and a final ciphertext or list return.',
    'Use a single golden function with a final ciphertext, list or tuple return.\n'
    'Public construction may use nested for loops over range or local lists/tuples,\n'
    'if/else and conditional expressions on public integer comparisons, len(container),\n'
    'integer +,-,*,//,%, local list/tuple literals, indexing with negative indices,\n'
    'exact-length unpacking, local list item assignment and list.append(value).\n'
    'Integer/bool literals are for public control only, never ciphertext operands.\n'
    'A local list holds separate values, not packed slots. Named public constants\n'
    'remain opaque scalar/vector operands, not indexable containers in this version.\n'
    'List aliases share item writes/append; rebinding a list name does not alter aliases.\n'
    'Ciphertext aliases keep their old value when a variable is rebound.\n'
    'Loops and branches build the graph before encryption and never inspect ciphertext data.\n'
    'Bounded expansion: 128 entries per container/range, 4096 evaluator steps,\n'
    '32 nesting depth, 256 emitted ciphertext operations. No recursive/cyclic containers.')
RULES_V7 = RULES_V7.replace(
    'No imports, arbitrary calls, indexing, containers other than the final return list,\n'
    'control flow, attribute access except rotate, annotations or default arguments.',
    'No imports, arbitrary calls, helper functions, comprehensions, while, break, continue,\n'
    'early returns, for-else, slicing, ciphertext indexing/conditions, annotations or defaults.\n'
    'Only range/len, ciphertext.rotate and local list.append calls are permitted.\n'
    'Augmented assignment requires a name, not an indexed list target.')
TASK_RULES['hecate-function-synthesis-v7'] = ('hecate-function-v6', RULES_V7)

RULES_V8 = RULES_V7.replace(
    'Use a single golden function with a final ciphertext, list or tuple return.',
    'Define one decorated golden function and at most 16 undecorated top-level helpers.\n'
    'Helpers have at most 16 positional parameters, no defaults, annotations or decorators.\n'
    'Helper calls bind arguments once left-to-right in a fresh lexical local frame.\n'
    'Free names refer to module public constants/helpers, never caller local variables.\n'
    'Parameters and locals may shadow public names without changing the constant registry.\n'
    'List arguments/results retain shared references; rebinding a parameter does not rebind the caller.\n'
    'Helpers may return cipher/plain values, public integers, local containers or declared helpers.\n'
    'Direct, composed and higher-order calls to declared helpers are allowed.\n'
    'Publicly controlled early returns unwind the current call, including from loops.\n'
    'Implicit/bare return is None and useful only for helper side effects; golden must return ciphertexts.\n'
    'Recursion is interpreted under 128 total helper calls and 16 call frames, with the existing step limit.\n'
    'This is graph construction, never Python execution or a runtime HEVM function call.')
RULES_V8 = RULES_V8.replace(
    'No imports, arbitrary calls, helper functions, comprehensions, while, break, continue,\n'
    'early returns, for-else, slicing, ciphertext indexing/conditions, annotations or defaults.',
    'No imports, external calls, nested definitions/closures, comprehensions, while, break, continue,\n'
    'for-else, slicing, ciphertext indexing/conditions, annotations or defaults.')
RULES_V8 = RULES_V8.replace('Only range/len, ciphertext.rotate and local list.append calls are permitted.',
    'Only declared construction helpers, range/len, ciphertext.rotate and local list.append calls are permitted.')
TASK_RULES['hecate-function-synthesis-v8'] = ('hecate-function-v7', RULES_V8)

RULES_V9 = RULES_V8.replace(
    'Define one decorated golden function and at most 16 undecorated top-level helpers.',
    'Define one decorated golden function and at most 16 undecorated helpers total, including nested definitions.').replace(
    'Free names refer to module public constants/helpers, never caller local variables.',
    'Free names resolve through lexical enclosing functions, then module constants/helpers; never dynamic caller locals.').replace(
    'nested definitions/closures, ', '') + '''
Nested functions capture shared lexical bindings, not snapshots. Rebinding an outer variable
is visible on later calls. Separate outer invocations create independent captured frames.
Loop-defined closures share that invocation's loop variable and observe its latest binding.
nonlocal may rebind an existing nearest enclosing function variable, never a module global.
Locals are determined statically; reading before initialization fails, without global fallback.
Captured lists preserve aliasing; captured ciphertext IDs remain immutable until rebinding.
At most 128 nested function instances; no lambda, defaults, keywords, global or external access.
'''
TASK_RULES['hecate-function-synthesis-v9'] = ('hecate-function-v8', RULES_V9)

RULES_V10 = RULES_V9.replace(
    'Helpers have at most 16 positional parameters, no defaults, annotations or decorators.',
    'Helpers have at most 16 parameters: positional-only (/), positional-or-keyword, keyword-only (*), '
    '*args and **kwargs are supported; no annotations or decorators.').replace(
    'for-else, slicing, ciphertext indexing/conditions, annotations or defaults.',
    'for-else, slicing, ciphertext indexing/conditions or annotations.').replace(
    'At most 128 nested function instances; no lambda, defaults, keywords, global or external access.',
    'At most 128 nested function instances; no lambda, global or external access.') + '''
Defaults are evaluated once when each definition executes, before its name is bound,
in its defining environment. Rebinding an outer variable does not change a saved default;
mutating a saved list/dict does. Repeated calls share defaults, separate factory definitions do not.
Module definitions execute in source order. Required missing parameters, duplicate assignments,
unexpected keywords and positional-only/keyword-only violations fail explicitly.
*args is a tuple; **kwargs is a fresh string-keyed dict whose values retain aliases.
Calls may expand bounded lists/tuples with * and string-keyed dicts with ** (128 items each).
Resolve the callable first, evaluate positional/star expressions in order, then keyword expressions
in order, then bind parameters. Do not silently reorder keyword side effects into parameter order.
String-keyed dict literals, indexing, item writes and len are permitted for construction;
dict unpacking literals, dict methods/iteration and general string computations are not yet supported.
Strings are bounded public keys, never ciphertext arithmetic literals. Builtins and rotate/append
retain their existing signatures; keyword calls are for declared construction helpers only.
'''
TASK_RULES['hecate-function-synthesis-v10'] = ('hecate-function-v9', RULES_V10)

RULES_V11 = RULES_V10.replace('external calls, comprehensions, while, break, continue,',
    'external calls, generator/set comprehensions, while, break, continue,').replace(
    'dict unpacking literals, dict methods/iteration and general string computations are not yet supported.',
    'dict unpacking literals, dict methods and general string computations are not yet supported.').replace(
    'Only declared construction helpers, range/len, ciphertext.rotate and local list.append calls are permitted.',
    'Only declared construction helpers, range/len/enumerate/zip/reversed/iter/next/list/tuple, '
    'ciphertext.rotate and local list.append calls are permitted.') + '''
List and string-keyed dict comprehensions support nested public for clauses and public if filters.
The first iterable is evaluated once in the enclosing scope; targets live in an implicit local scope
and do not leak. Later iterables and filters see earlier target bindings. Dict keys evaluate before
values; duplicate keys keep the last value. Each resulting container has at most 128 entries.
enumerate, zip and reversed return stateful lazy public iterators, not eager snapshots. Aliases share
their consumption state; zip stops at the shortest input, consuming inputs left-to-right.
iter returns a public iterator; next(iterator[, default]) consumes one item and errors on exhaustion
without a default. list/tuple materialize at most 128 items. Public dict loops follow key insertion order.
Existing step/resource limits apply to every iterator advance, including filtered/unused results.
Iterating ciphertext or opaque named constants and encrypted filter conditions is forbidden.
No generator expressions, yield, async iteration, sets, general strings or dict methods are enabled.
These builtins use positional calls only. The new builtins' names are reserved in this contract.
'''
TASK_RULES['hecate-function-synthesis-v11'] = ('hecate-function-v10', RULES_V11)

RULES_V12 = RULES_V11.replace('keyword calls are for declared construction helpers only.',
    'keyword calls are for declared helpers and sorted only.').replace('no lambda, global or external access.',
    'no global or external access.').replace('range/len/enumerate/zip/reversed/iter/next/list/tuple,',
    'range/len/enumerate/zip/reversed/iter/next/list/tuple/sorted,') + '''
Lambda expressions construct lexical function values under the same parameter/default rules as helpers.
All named/lambda declarations together are limited to 17, all nested/lambda instances to 128.
Lambda free variables are late-bound; defaults retain definition-time values. Lambda bodies are expressions.
Declared function values may be called directly, returned then called, or selected from public containers.
No external Python callable or arbitrary attribute access is permitted.
sorted(iterable, key=None, reverse=False) is available: materialize the input first, compute each key once,
then sort stably (equal keys preserve order even with reverse=True). key must be a declared function or None.
Keys must be public integers/booleans/strings or nested list/tuple keys made only of these public values.
Never sort ciphertext or opaque plaintext operands to infer a secret-dependent order. reverse must be public.
The sorted builtin accepts one positional iterable and only key/reverse keyword options; no other builtin
signature is changed. Sorting does not permute slots inside ciphertexts, only public construction objects.
'''
TASK_RULES['hecate-function-synthesis-v12'] = ('hecate-function-v11', RULES_V12)

RULES_V13 = RULES_V12.replace('for-else, slicing, ciphertext indexing/conditions or annotations.',
    'for-else, ciphertext indexing/conditions or annotations.').replace(
    'dict unpacking literals, dict methods and general string computations are not yet supported.',
    'dict unpacking literals and dict methods are not yet supported.').replace(
    'No generator expressions, yield, async iteration, sets, general strings or dict methods are enabled.',
    'No generator expressions, yield, async iteration, sets or dict methods are enabled.') + '''
Public list/tuple/string sequences support integer indexing and [start:stop:step] slicing.
Bounds are public integers/booleans or None, with Python clipping/negative-index semantics;
step cannot be zero. A slice is a shallow copy: nested mutable objects remain shared.
Same-type sequence + concatenates; sequence * public integer (either order) repeats shallow
references. Nonpositive counts produce empty sequences; no result may exceed 128 items/chars.
String len, iteration, reversed and list/tuple conversion are public construction operations.
List slice assignment evaluates RHS, then target/bounds, then materializes the replacement before
mutation. Contiguous slices may resize; non-unit step replacements must match selected length.
Named-list += extends IN PLACE and *= repeats IN PLACE, preserving aliases; ordinary +/* create
new sequences. Direct self-extension snapshots, but extending from an iterator observes appends.
Containers must remain acyclic and bounded. Tuple/string mutation and augmented item assignment
are rejected. Named constants remain opaque/read-only: these operations neither slice a weight
array nor access or permute ciphertext slots. No public float/NumPy/Torch array computation is added.
'''
TASK_RULES['hecate-function-synthesis-v13'] = ('hecate-function-v12', RULES_V13)

RULES_V14 = RULES_V13 + '''
PUBLIC NUMERIC EXTENSION (supersedes earlier opaque/named-only operand restrictions):
Finite public int/float literals and +,-,*,/,//,%,** plus unary +/- are supported.
float(value), int(value), pow(a,b) are bounded public conversions/arithmetic, not cipher operations.
Public magnitudes are at most 1048576; exponent magnitude at most 16. Reject zero division,
nonfinite/complex results and values outside resource bounds. Floats may be public conditions/keys.
np.array/asarray(value[, dtype="float64"]) construct checked real arrays from public numerical
lists/tuples/scalars/arrays only. No import, arbitrary NumPy calls, file or external capability.
Arrays have rank <=4 and <=128 elements; dtype is inferred int/float or explicit float64.
Named constants have the same float64 length-one/four array shape as actual tracing, including
scalar manifest values wrapped as length-one arrays. Index/slice/iterate/read shape from these
arrays; arithmetic makes new values. Arrays are read-only in this version, including through aliases.
Array +,-,*,/,//,%,** are elementwise with trailing-dimension broadcasting. Python list +/* retain
sequence behavior unless interacting with an array. reshape(shape) or reshape(dims...) and flatten()
are C-order only; inferred -1 dimension is checked. No implicit flatten or shape reinterpretation.
Derived scalar/one-dimensional length-one/four data can be used with cipher +,-,* on either side.
The normalizer emits a separately hashed derived-constants manifest and preserves original weights,
request, reference and profile. Encoded values remain bounded by 1024, total constants <=128.
Only public arithmetic may use division or powers: ciphertext / or ** is rejected, not silently rewritten.
No complex/boolean array dtype, masked/advanced indexing, array writes, ndarray comparisons,
reductions, matrix multiplication, np.polynomial, general Torch/NumPy or bootstrap is claimed.
'''
TASK_RULES['hecate-function-synthesis-v14'] = ('hecate-function-v13', RULES_V14)

RULES_V15 = RULES_V14 + '''
PUBLIC CONTROL EXTENSION (supersedes earlier for-else/control/string-key restrictions):
Public while, for-else, while-else, break, continue and pass are supported within existing expansion
budgets. Else executes on exhaustion/false condition, never after break or return. Break/continue
belong to the nearest lexically enclosing loop, never the caller of a helper. Unbounded loops fail.
and/or short-circuit and return the selected operand; not returns a public bool. Only operands
actually truth-tested must be public. None, public scalars/containers, functions and iterators have
Python truth semantics (iterators are true even when exhausted). Cipher and ndarray truth is rejected.
Immutable public None/bool/int/float/string/recursive-tuple dictionary keys are allowed, with Python
numeric key equality (1 == True == 1.0). Dict comprehensions and subscripts use the same rules.
Keyword ** mappings still require string keys. Mutable, cipher, array and function keys are rejected.
Public scalar/list/tuple/dict equality and scalar/lexicographic sequence ordering are supported.
Comparisons short-circuit left to right. in/not in supports dictionary keys, substring matching,
public lists/tuples/iterators; iterator membership consumes through the match or exhaustion.
Never compare ciphertext values, derive secret-dependent branches or access ciphertext slots.
No arbitrary Python execution, imports, I/O, generator/yield, dict methods or full GenPoly is claimed.
'''
TASK_RULES['hecate-function-synthesis-v15'] = ('hecate-function-v14', RULES_V15)

RULES_V16 = RULES_V15 + '''
PUBLIC STRING EXTENSION (supersedes earlier general-string exclusion for these methods):
Bounded public str supports direct strip/lstrip/rstrip, split/rsplit, partition/rpartition,
replace and join calls. strip(chars) removes a SET of characters, not a prefix or suffix.
split(None) collapses whitespace and discards empty boundaries; split(explicit_sep) retains
empty fields. Empty explicit separators are invalid. maxsplit and count follow Python 3.10.
Only split/rsplit accept sep/maxsplit keywords; all other method arguments are positional-only.
Method receiver is evaluated before arguments. Bounded *args/**kwargs use existing public call
binding rules. join consumes a public iterable and requires every element to be a string.
Results/containers are bounded to 128 characters/items. Calls have no file/network capability.
Use int/float after parsing numeric tokens; never interpret strings as candidate Python.
No format/eval/exec, encode/decode, arbitrary methods, bound-method values, cipher string
conversion or whole GenPoly implementation is claimed. Original weights/reference are unchanged.
'''
TASK_RULES['hecate-function-synthesis-v16'] = ('hecate-function-v15', RULES_V16)

RULES_V17 = RULES_V16 + '''
PUBLIC CHEBYSHEV EXTENSION (supersedes earlier np.polynomial exclusion only as below):
np.polynomial.Chebyshev(coef, domain=None, window=None, symbol="x") constructs bounded real
public Chebyshev series data. Explicit positional/keyword constructor arguments are supported.
Coefficients are a nonempty scalar/1D vector, at most 128 values; preserve trailing zeros on
construction. Default domain/window are [-1,1]. Symbol is a bounded identifier. Domain/window
must have two endpoints; arithmetic preserves metadata and rejects incompatible polynomials.
Public polynomial +,-,*,//,% and unary +/- use the project's existing NumPy Chebyshev
semantics. // and % are polynomial quotient and remainder, not elementwise coefficient division.
Integer powers 0..16 are supported; product/power output degree must be below 128.
.coef, .domain and .window expose read-only public arrays. No attribute or coefficient mutation.
np.double and np.float64 are accepted as dtype markers for the existing array constructors.
np.floor, np.ceil and np.log2 accept one public scalar/array argument; reject nonfinite
results and invalid domains. Original finite magnitude/resource bounds remain enforced.
No arbitrary NumPy calls, ciphertext numerical functions, direct Polynomial/cipher arithmetic,
polynomial object calls, fitting, roots, differentiation, integration or serialization are enabled.
The current upstream GenPoly leaf loop visits odd coefficient indices only. It is not a general
Chebyshev evaluator: do not route models with even/constant terms through it without a correct
explicit rewrite. Tree/length/decomposition compatibility must be validated, not assumed.
GenPoly may return a construction closure using these operations; the closure, not a polynomial
object, is then called on ciphertext. Only cipher +,-,*,negation and provisioned rotations lower.
'''
TASK_RULES['hecate-function-synthesis-v17'] = ('hecate-function-v16', RULES_V17)

RULES_V18 = RULES_V17 + '''
OBJECT STORAGE EXTENSION (supersedes earlier object-array exclusions only as follows):
np.empty(shape,dtype=object) creates None cells, not ciphertext zeros. np.full(shape,fill,
dtype=object) fills bounded object storage. np.array/asarray(value,dtype=object) construct
object arrays; np.object_ is also a permitted dtype marker. np.asarray(existing_object_array)
shares storage; np.array(existing_object_array) copies. The arrays store symbolic ciphertext/
plaintext expressions and public scalar data, NOT the encrypted slots inside one ciphertext.
Rank at most 4 and at most 128 elements; shape is public integer data. Integer/tuple/slice
indexing and assignment preserve NumPy broadcasting, view and overlap semantics. No boolean,
advanced, secret indexing or nested mutable objects in cells. Scalar-cell +=,-=,*= evaluate
target/index exactly once before RHS; whole-array/vectorized arithmetic is not enabled.
.shape/.size/.ndim, len and iteration are public metadata/construction operations. Iteration
observes writes and yields row views. reshape accepts positional C-order shape; flatten()
and copy() create new shallow storage. transpose() / transpose(axes) / .T follow NumPy.
np.concatenate((a,b),axis=0) copies object arrays; axis=None explicitly flattens for concatenation.
Empty() and hc.Empty() take no arguments. Empty()+Expr and Empty()-Expr return the Expr
unchanged: Empty is NOT numerical zero. Expr+Empty() and Expr-Empty() are invalid. Public
int/float/bool/list plus/minus Empty, in either order, converts public data to Plain. Empty
multiplication/negation and Empty/array mixed arithmetic are unsupported, not zero shortcuts.
Object-array cells may contain Expr, Empty, None or bounded public real scalars. None/Empty
cannot be returned as ciphertext. Return arrays must be 1D with exactly the declared number
of ciphertext objects; multidimensional returns are rejected, never implicitly flattened.
Other operations, reflection and arbitrary object methods stay forbidden. Do not invent
slot reshuffling from container reshaping. Existing packing, key, arithmetic, numerical and
security requirements are unchanged. No bootstrap/upscale or new runtime backend is enabled.
'''
TASK_RULES['hecate-function-synthesis-v18'] = ('hecate-function-v17', RULES_V18)

RULES_V19 = RULES_V18 + '''
Public mappings: dict() accepts zero/one dict or iterable-of-pairs argument and keywords.
Supported methods: copy/get/setdefault/pop/popitem/clear/update/keys/values/items.
copy is shallow; views are live. Keys must be bounded immutable public values.
Iterating while changing dictionary size fails. Cycles, secret keys/comparisons,
fromkeys, dict union and view set algebra remain forbidden.
If construction_exercise is present, additionally satisfy its frozen construction
requirements using output-relevant operations. Dead or unused padding is forbidden.
These exercises test a specified implementation form, unlike unrestricted synthesis.
'''
TASK_RULES['hecate-function-synthesis-v19'] = ('hecate-function-v18', RULES_V19)

RULES_V20 = RULES_V19 + '''
OBJECT ARITHMETIC EXTENSION supersedes the earlier whole-array arithmetic exclusion:
Object arrays support +,-,* with bounded public scalars/lists/numeric arrays or other
object arrays. Broadcasting aligns storage dimensions, not ciphertext slots. Result
storage is fresh; 0-D results are scalar. Object-array and slice +=,-=,*= mutate shared
storage in place without growing its shape; overlapping reads use pre-write values.
Evaluate the target once before RHS; a slice remains a live view during RHS effects.
An object array on the left may broadcast a single cipher Expr on the right. A bare
Expr or Empty on the left of an object array is NOT this dispatch and remains rejected.
Direct named/derived Plain operands require explicit public np.asarray conversion;
otherwise the Plain/object-array ambiguity is rejected. Cell arithmetic keeps the
existing Expr and Empty rules; Empty-x returns x, not -x. None arithmetic is invalid.
No object-array division, power, unary ufunc, matmul, reduction or advanced indexing is
introduced. Rank <=4, storage <=128 and expanded operation limits remain in force.
'''
TASK_RULES['hecate-function-synthesis-v20'] = ('hecate-function-v19', RULES_V20)

RULES_V21 = RULES_V20 + '''
PUBLIC SCALAR EXTRACTION EXTENSION (pinned NumPy 1.25.2):
float()/int() default to 0.0/0. Public bools, finite numbers and numeric strings
are accepted; int(string, base) supports explicit public integer bases. int(real)
truncates toward zero. Bounded numeric/object arrays convert only if size == 1.
Rank-zero conversions are ordinary; rank-positive singleton conversions match
NumPy 1.25.2's deprecated behavior and are recorded as legacy conversions.
.item() extracts the sole cell; .item(flat_index), .item((i,j)) and .item(i,j)
extract by bounded public integer index (negative indices allowed, booleans rejected).
np.array/asarray of original public constants with dtype=object preserves numeric
storage shape and numeric cells. An object-array item holding a cipher returns its
symbolic Expr, NOT a decrypted number. float/int(cipher), derived Plain Expr casts,
non-singleton array casts, None/Empty casts and non-finite values remain forbidden.
No file/network/process permissions, arbitrary coercion hooks or precision changes.
'''
TASK_RULES['hecate-function-synthesis-v21'] = ('hecate-function-v20', RULES_V21)

RULES_V22 = RULES_V21 + '''
OBJECT UNARY EXTENSION supersedes the object-array unary exclusion:
-array and np.negative(array) apply Hecate negation to each cipher cell and negate
public numeric cells. +array and np.positive(array) are allowed for public numeric
cells, NOT cipher/Plain Expr, because upstream Expr has no __pos__.
Each operator evaluates its operand once and returns fresh object storage; 0-D
object results are scalars/Expr. Input storage and aliases are not modified.
Known constructed Plain negation keeps Plain typing; float/int of its result
still fails. Empty and None cells fail. Booleans use Python object arithmetic.
The np.negative/positive entry requires one explicit object array; no out, where,
dtype, casting or other keywords. General numeric ufuncs are not introduced here.
No division/power/matmul/reduction or bootstrap expansion is introduced.
'''
TASK_RULES['hecate-function-synthesis-v22'] = ('hecate-function-v21', RULES_V22)
TASK_RULES[NATIVE_TASK] = (NATIVE_CONTRACT, NATIVE_RULES)
TASK_RULES[NATIVE_EXERCISE_TASK] = (NATIVE_CONTRACT, NATIVE_EXERCISE_RULES)
TASK_RULES[NATIVE_ARRAY_TASK] = (NATIVE_ARRAY_CONTRACT, NATIVE_ARRAY_RULES)
TASK_RULES[ARRAY_EXERCISE_TASK] = (NATIVE_ARRAY_CONTRACT, ARRAY_EXERCISE_RULES)
TASK_RULES[STAR_TASK] = (STAR_CONTRACT, STAR_RULES)
TASK_RULES[ARITHMETIC_TASK] = (ARITHMETIC_CONTRACT, ARITHMETIC_RULES)
TASK_RULES[LOOP_TASK] = (LOOP_CONTRACT, LOOP_RULES)
TASK_RULES[AUGMENTED_TASK] = (AUGMENTED_CONTRACT, AUGMENTED_RULES)
TASK_RULES[MUTATION_TASK] = (MUTATION_CONTRACT, MUTATION_RULES)
TASK_RULES[STAR_EXERCISE_TASK] = (STAR_CONTRACT, STAR_EXERCISE_RULES)


SEMANTIC_GUIDANCE = {
    'schema': 1,
    'authority': 'Clarifies the selected task rules; does not enable new syntax or operators.',
    'public_operands': 'A JSON scalar or length-one public list broadcasts to every logical slot. '
        'A length-four list is a period-four elementwise vector, not four ciphertexts. '
        'Only supplied named constants are usable; no literals, indexing or public-only arithmetic in the function.',
    'constant_origins': 'The public_constants registry contains lowered values, not necessarily the original model weights. '
        'Consult constant_origins and the model/FX graph: subtraction lowering can export a negative offset, '
        'and Linear exports individual rows or scalar coefficients. Never assume a supplied constant needs another negation.',
    'ssa_aliases': 'A fresh assignment such as a = x aliases that ciphertext value; it is not an independent encrypted input. '
        'Each temporary is defined once. Rebinding and augmented assignments are forbidden. '
        'Aliases must have ciphertext type, not public-constant type.',
    'layout': 'Each encrypted input has period-four repeated packing with C-row-major logical order. '
        'A broadcast public value does not change this packing. Linear outputs may be scalar-neuron ciphertexts; '
        'obey the fixed output_ciphertexts and output_selectors, never confuse ciphertext count with slot count.',
    'semantics_not_spelling': 'Preserve ordered subtraction and unary negation when available in the selected task. '
        'Equivalent expressions are allowed; no required spelling or coverage-padding operations. '
        'Numerical comparison with the immutable independent reference decides the tested correctness.',
    'fhe_boundary': 'Rotation is left for positive steps and uses only provisioned task steps. '
        'The compiler owns scale, level, rescale, modswitch and relinearization; the harness owns keys and encoding. '
        'No bootstrap, arbitrary helper, automatic activation approximation, or changed security/tolerance is allowed.',
}


SEMANTIC_GUIDANCE_V6 = dict(SEMANTIC_GUIDANCE, ssa_aliases=
    'Assignments bind ciphertext values. Rebinding and +=/-=/*= return a new Expr; '
    'existing aliases retain the old value. Public constants and zero_ct are read-only. '
    'Evaluate the complete right side before rebinding the target.')

PACKED_GUIDANCE = dict(SEMANTIC_GUIDANCE_V6,
    public_operands='A named scalar/length-one public constant broadcasts. A length-P public constant is a '
        'period-P vector, where P is layout.input_slot_period. It is not P ciphertexts. '
        'Only the supplied names are usable; no invented constants or public-only arithmetic.',
    layout='One encrypted logical input is C-row-major flattened, zero padded to layout.input_slot_period P, '
        'and periodically repeated across 16384 slots. P is one of 4,8,16,32,64,128,256, NOT always four. '
        'Linear sums over P slots using rotations 1,2,4,...P/2, including the zero-padded weight columns. '
        'Follow output_representation and output_selectors: packed_prefix and scalar_neurons are distinct.')

# Prompt semantics revision, not a new AST/ABI or a change to old requests.
PACKED_GUIDANCE_V2 = dict(PACKED_GUIDANCE, schema=2, revision='node-layout-v1',
    node_input_layout={
        'rule': 'Constants belonging to a graph node are expressed in that node input layout, not automatically the original model input layout.',
        'data_flow': 'Follow model.nodes inputs and the FX graph. A Conv/Linear after a permutation consumes the permuted value; its lowered row constants do not silently include earlier graph transformations.',
        'equivalence': 'Equivalent fused expressions are allowed only if they implement the complete original graph. Swapping plaintext/ciphertext multiplication order cannot repair a missing layout transformation.'},
    axis_permutation={
        'scope': 'Static permute/transpose changes C-row-major element order. Reshape alone only changes logical dimensions, not flat order.',
        'constant_coordinates': 'A constant origin ending in permutation_shift[d] is a DESTINATION-coordinate mask for the permutation node input. It is not a source-coordinate selector.',
        'positive_rotation': 'For period P, rotate(source,d)[j] = source[(j+d) mod P]. To read source position s at destination j use d=(s-j) mod P.',
        'mask_application': 'Each such mask selects destination slots AFTER the source has been rotated by d; sum these masked rotated values. Multiplying the unrotated source by the same mask then rotating generally implements a different map.',
        'allowed_steps': 'Only supplied positive power-of-two rotation steps are allowed. A composite displacement is the sum of these steps modulo P; do not use an unprovisioned negative step.',
        'scalar_layout': 'Permuting scalar-neuron layout reorders whole ciphertext references, not slots inside each ciphertext.'})

PACKED_NATIVE_GUIDANCE = dict(PACKED_GUIDANCE_V2, schema=3, revision='packed-native-v1',
    public_operands='Supplied Plain scalar/length1 or periodP constants and bounded finite numeric literals are public. Native c/p signatures must match actual Expr kinds.',
    construction='Native decorated helper calls, object arrays, starred positional calls, public range loops and scalar/array augmented operators follow the exact task rules. Array cells are whole Exprs, never packed slots. Scalar rebinding preserves old aliases; ndarray in-place operations update shared storage and views, not copied storage. No candidate Python execution or ciphertext control flow.')

SEMANTIC_GUIDANCE_V7 = dict(SEMANTIC_GUIDANCE_V6, public_construction=
    'Public loops, indices and conditions are expanded before Hecate tracing. '
    'Containers hold symbolic values, not ciphertext slots. No encrypted value may decide control flow. '
    'Mutable local lists preserve aliasing; ciphertext operations produce new values. '
    'The original source and checked straight-line expansion are both saved.',
    public_operands='Named scalar/length-one/length-four public constants keep their existing broadcasting. '
    'Local container indexing selects whole symbolic operands, not elements inside a named constant. '
    'Integer literals and integer arithmetic are for public construction only.')

SEMANTIC_GUIDANCE_V8 = dict(SEMANTIC_GUIDANCE_V7, function_composition=
    'Declared top-level helpers are interpreted as AST with fresh lexical frames, not executed as Python. '
    'Calls evaluate arguments left-to-right once, preserve shared containers, and isolate local rebindings. '
    'Returns unwind only their current call. Public bounded recursion and helper-valued parameters are allowed. '
    'Undefined locals never reuse values from an earlier invocation. No caller-local capture or external calls.',
    fhe_boundary='The compiler and harness retain scale/level, keys, encoding and security ownership. '
    'Declared helper calls only construct the graph; no bootstrap, external helper, automatic approximation, '
    'ciphertext-dependent branch or changed security/tolerance is allowed.')


SEMANTIC_GUIDANCE_V9 = dict(SEMANTIC_GUIDANCE_V8, function_composition=
    'Declared helpers and nested closures are interpreted as AST, never executed as Python. '
    'Each call has fresh locals and its lexical defining environment, not the dynamic caller. '
    'Free variables are late-bound shared cells; nonlocal writes the nearest enclosing function cell. '
    'Separate factory calls have separate cells. Uninitialized locals/cells fail. '
    'Only public control flow and bounded call/closure creation are supported.')


SEMANTIC_GUIDANCE_V10 = dict(SEMANTIC_GUIDANCE_V9, call_binding=
    'Defaults retain definition-time values/references, unlike late-bound closure cells. '
    'Omitted arguments reuse defaults; mutable defaults persist per function instance. '
    'Respect positional-only and keyword-only parameters; evaluate argument expressions before binding. '
    'Varargs/kwargs are bounded public containers of symbolic values, not ciphertext slot unpacking.')


SEMANTIC_GUIDANCE_V11 = dict(SEMANTIC_GUIDANCE_V10, public_iteration=
    'Comprehensions build bounded public containers in implicit lexical scopes, not packed tensor slots. '
    'Evaluate the outer iterable once before creating that scope; do not leak targets. '
    'Iterator aliases share state and observe permitted container mutations lazily. '
    'Reject encrypted iteration/filter decisions. Iteration counters are not output-coverage proof.')


SEMANTIC_GUIDANCE_V12 = dict(SEMANTIC_GUIDANCE_V11, function_literals=
    'Lambda values preserve lexical cells and definition-time defaults; only declared AST functions can be called. '
    'Stable sorted uses public keys computed once after consuming the input; it never compares ciphertext values.')

SEMANTIC_GUIDANCE_V13 = dict(SEMANTIC_GUIDANCE_V12, public_sequences=
    'Public slicing selects construction objects, not ciphertext slots or elements of opaque named constants. '
    'Slices/concatenations/repeats are shallow; nested aliases persist. Named-list +=/*= mutate shared lists. '
    'Slice writes preserve Python evaluation order and exact extended-slice length, subject to resource bounds.')

SEMANTIC_GUIDANCE_V14 = dict(SEMANTIC_GUIDANCE_V13, public_operands=
    'Original named constants are read-only float64 arrays of length one or four. Public arithmetic and '
    'checked np.array/asarray may derive new constants, emitted separately without changing the input registry. '
    'Broadcast is trailing-dimension array broadcast; lists still have Python concatenation/repetition semantics. '
    'Explicitly reshape/flatten to scalar or 1D length one/four before ciphertext use; no silent packing change.',
    numeric_boundary='Only cipher +,-,* are supported. Public division/powers do not create cipher division/powers. '
    'Reject nonfinite values and retain frozen encoding magnitude, security and numerical tolerance bounds.')


SEMANTIC_GUIDANCE_V15 = dict(SEMANTIC_GUIDANCE_V14,
    public_control='Public-only short-circuit truth, membership and loop control are evaluated while '
    'constructing the graph. Loop else is exhaustion, not a generic finally block. Public dictionary '
    'keys may be immutable tuples/numbers. No secret-dependent control flow or ciphertext equality.')

SEMANTIC_GUIDANCE_V16 = dict(SEMANTIC_GUIDANCE_V15,
    public_strings='Parse only public coefficient/tree text using bounded string methods. '
    'Preserve empty fields for explicit separators and character-set strip semantics. '
    'Parsing is construction-time, not encrypted string processing or code execution.')

SEMANTIC_GUIDANCE_V17 = dict(SEMANTIC_GUIDANCE_V16,
    public_polynomial='Chebyshev coefficients belong to the Chebyshev basis, not monomial powers. '
    'Quotient/remainder preserve the reconstruction a=q*b+r and basis/domain/window metadata. '
    'Read public coefficients then construct cipher arithmetic explicitly or via GenPoly closure. '
    'No ciphertext bootstrap, direct polynomial-object evaluation or generic NumPy is introduced.')

SEMANTIC_GUIDANCE_V18 = dict(SEMANTIC_GUIDANCE_V17,
    object_arrays='Object arrays hold symbolic expressions, not CKKS slots. Slices may alias; '
    'copies separate storage. Empty is a construction sentinel, not zero: Empty-x returns x '
    'but x-Empty fails. Preserve target evaluation order and explicitly verify packing.')

SEMANTIC_GUIDANCE_V19 = dict(SEMANTIC_GUIDANCE_V18,
    public_mappings='Bounded public dictionaries preserve insertion order, shallow copies and live views. '
        'They contain construction values, not encrypted dictionary keys or ciphertext slots.',
    semantics_not_spelling='For construction exercises obey the specified implementation form and '
        'use every required construct in output construction; no dead padding. Other expressions '
        'may vary. The independent encrypted numerical comparison is still mandatory.')


SEMANTIC_GUIDANCE_V20 = dict(SEMANTIC_GUIDANCE_V19,
    object_arithmetic='Array broadcasting constructs multiple ciphertext operations; '
        'it never changes packing within a ciphertext. Binary results have fresh storage, '
        'in-place updates preserve aliases, and overlapping slices read original values.')


SEMANTIC_GUIDANCE_V21 = dict(SEMANTIC_GUIDANCE_V20,
    scalar_conversion='Public construction-time extraction only. item on symbolic object cells '
        'preserves the Expr; float/int never decrypt. Array shapes describe storage, not CKKS slots. '
        'Rank-positive singleton casts are legacy NumPy 1.25.2 behavior, not a future-version guarantee.')

SEMANTIC_GUIDANCE_V22 = dict(SEMANTIC_GUIDANCE_V21,
    object_unary='Array negation creates elementwise expressions, not a slot permutation. '
        'Fresh storage must not alias input. Positive is undefined for Hecate Expr; '
        'do not replace it with identity or multiply by one.')


def valid_semantic_guidance(request):
    # Historical requests have no guidance field and remain verifiable unchanged.
    if 'semantic_guidance' not in request:
        return True
    value = request['semantic_guidance']
    expected = SEMANTIC_GUIDANCE_V6 if request.get('task') == 'hecate-function-synthesis-v6' else SEMANTIC_GUIDANCE
    if request.get('task') == CHUNKED_TASK:
        expected = SEMANTIC_GUIDANCE_V6
    if request.get('task') == PACKED_TASK:
        expected = PACKED_GUIDANCE
        if type(value) is dict and value.get('schema') == 2:
            model=request.get('model',{})
            if type(model) is not dict or not any(type(n) is dict and n.get('op') in ('permute','transpose')
                                                   for n in model.get('nodes',[])):
                return False
            expected = PACKED_GUIDANCE_V2
    if request.get('task') in PACKED_NATIVE_TASKS:
        expected = PACKED_NATIVE_GUIDANCE
    if request.get('task') == 'hecate-function-synthesis-v7':
        expected = SEMANTIC_GUIDANCE_V7
    if request.get('task') == 'hecate-function-synthesis-v8':
        expected = SEMANTIC_GUIDANCE_V8
    if request.get('task') == 'hecate-function-synthesis-v9':
        expected = SEMANTIC_GUIDANCE_V9
    if request.get('task') == 'hecate-function-synthesis-v10':
        expected = SEMANTIC_GUIDANCE_V10
    if request.get('task') == 'hecate-function-synthesis-v11':
        expected = SEMANTIC_GUIDANCE_V11
    if request.get('task') == 'hecate-function-synthesis-v12':
        expected = SEMANTIC_GUIDANCE_V12
    if request.get('task') == 'hecate-function-synthesis-v13':
        expected = SEMANTIC_GUIDANCE_V13
    if request.get('task') == 'hecate-function-synthesis-v14':
        expected = SEMANTIC_GUIDANCE_V14
    if request.get('task') == 'hecate-function-synthesis-v15':
        expected = SEMANTIC_GUIDANCE_V15
    if request.get('task') == 'hecate-function-synthesis-v16':
        expected = SEMANTIC_GUIDANCE_V16
    if request.get('task') == 'hecate-function-synthesis-v17':
        expected = SEMANTIC_GUIDANCE_V17
    if request.get('task') == 'hecate-function-synthesis-v18':
        expected = SEMANTIC_GUIDANCE_V18
    if request.get('task') == 'hecate-function-synthesis-v19':
        expected = SEMANTIC_GUIDANCE_V19
    if request.get('task') == 'hecate-function-synthesis-v20':
        expected = SEMANTIC_GUIDANCE_V20
    if request.get('task') == 'hecate-function-synthesis-v21':
        expected = SEMANTIC_GUIDANCE_V21
    if request.get('task') == 'hecate-function-synthesis-v22':
        expected = SEMANTIC_GUIDANCE_V22
    return type(value) is dict and type(value.get('schema')) is int and value == expected


def request_input_names(request):
    request_rotations(request)  # Verify the version/rules pair before interpreting layout.
    if request.get('task') in PACKED_TASKS:
        from packed_input_abi import validate_request
        validate_request(request)
        return physical_input_names(request['layout'])
    require('execution_abi' not in request['layout'] and request.get('model',{}).get('schema')!=5,
            'Variable period requires its versioned contract')
    if request.get('task') == CHUNKED_TASK:
        validate_request_binding(request)
    else:
        require('model_input_binding' not in request['layout'] and
                request.get('model', {}).get('schema') != 4, 'Chunked input requires its versioned contract')
    layout = request["layout"]
    contract = TASK_RULES[request['task']][0]
    zero = has_zero_argument(layout)
    extended = contract in ('hecate-function-v5', 'hecate-function-v6', 'hecate-function-v7', 'hecate-function-v8', 'hecate-function-v9', 'hecate-function-v10', 'hecate-function-v11', 'hecate-function-v12', 'hecate-function-v13', 'hecate-function-v14', 'hecate-function-v15', 'hecate-function-v16', 'hecate-function-v17', 'hecate-function-v18', 'hecate-function-v19', 'hecate-function-v20', 'hecate-function-v21')
    extended = extended or contract in (NATIVE_CONTRACT,NATIVE_ARRAY_CONTRACT,STAR_CONTRACT,ARITHMETIC_CONTRACT,LOOP_CONTRACT,AUGMENTED_CONTRACT,MUTATION_CONTRACT)
    require(extended or zero == (contract == 'hecate-function-v4'), 'Auxiliary zero requires its versioned ABI')
    if (contract == 'hecate-function-v4' or extended) and 'inputs' not in layout:
        require(type(layout.get('input_shape')) is list and all(type(n) is int for n in layout['input_shape']) and
                tuple(layout['input_shape']) in INPUT_SHAPES, 'Invalid logical input shape')
        return physical_input_names(layout)
    if not extended and contract not in ("hecate-function-v3", "hecate-function-v4"):
        require("inputs" not in layout, "Multi-input layout requires v3")
        return ("x",)
    specs = layout.get("inputs")
    require(type(specs) is list and 2 <= len(specs) <= 4, "Invalid multi-input layout")
    require(all(type(s) is dict and set(s) == {"name", "dsl_name", "shape"} and
                type(s["name"]) is str and type(s["dsl_name"]) is str and
                type(s["shape"]) is list and all(type(n) is int for n in s["shape"]) and
                tuple(s["shape"]) in INPUT_SHAPES for s in specs), "Invalid input layout entry")
    require(len({s["name"] for s in specs}) == len(specs), "Duplicate input name")
    names = tuple(s["dsl_name"] for s in specs)
    require(names == ("x", "y", "z", "t")[:len(specs)], "Noncanonical DSL input order")
    return physical_input_names(layout)


def request_rotations(request):
    task = request.get("task")
    require(type(task) is str and task in TASK_RULES and request.get("rules") == TASK_RULES[task][1],
            "Unknown rotation contract")
    if task in PACKED_TASKS:
        from packed_input_abi import validate_request,rotations
        return rotations(validate_request(request)['slot_period'])
    return CONTRACT_ROTATIONS[TASK_RULES[task][0]]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw):
    require(type(raw) is str and len(raw.encode()) <= MAX_BYTES, "JSON response size/type limit")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Non-finite JSON number")

    def real_number(token):
        number = float(token)
        require(math.isfinite(number), "Non-finite JSON number")
        return number

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=real_number)
    except (RecursionError, MemoryError) as error:
        raise ValueError("JSON resource limit") from error


def make_request(translation, descriptor, profile_hash, model_structure=None, *, extended_arithmetic=False,
                 public_construction=False, function_composition=False, closures=False, call_binding=False,
                 public_iteration=False, function_literals=False, public_sequences=False, public_numbers=False, public_control=False, public_strings=False, public_polynomial=False, object_arrays=False, public_mappings=False, construction_exercise=None, object_arithmetic=False, scalar_conversion=False, object_unary=False, native_functions=False, native_arrays=False, native_starred=False, native_array_arithmetic=False, native_public_loops=False, compiler_configuration=None, native_scalar_augmented=False, native_array_mutation=False):
    """Trusted prepared model -> public request; deliberately omit the rule answer.

    v0 fixes layout and a derived public-constant registry using the deterministic
    baseline. This is a constrained synthesis task, not independent layout choice.
    """
    packed_exercise=type(construction_exercise) is str and construction_exercise.startswith('pn-')
    require(not packed_exercise or descriptor.get('schema')==5 and native_array_mutation,
            'Packed-native exercises require schema5 and native-array-mutation')
    native_scalar_augmented = native_scalar_augmented or native_array_mutation
    native_public_loops = native_public_loops or native_scalar_augmented
    native_array_arithmetic = native_array_arithmetic or native_public_loops
    native_starred = native_starred or native_array_arithmetic
    require(not native_starred or construction_exercise is None or packed_exercise or
            (type(construction_exercise) is str and construction_exercise.startswith('ns-') and not native_array_arithmetic),
            'Frozen exercises cannot change starred-call contract')
    native_arrays = native_arrays or native_starred
    native_functions = native_functions or native_arrays
    native_prefix = 'pn-' if packed_exercise else 'ns-' if native_starred else 'na-'
    require(not native_arrays or construction_exercise is None or construction_exercise.startswith(native_prefix),
            'Frozen native exercises cannot change array contract')
    require(not (type(construction_exercise) is str and construction_exercise.startswith('ns-'))
            or native_starred, 'Native starred exercises require their explicit contract')
    require(not (type(construction_exercise) is str and construction_exercise.startswith('na-'))
            or native_arrays, 'Native array exercises require their explicit contract')
    require(not (type(construction_exercise) is str and construction_exercise.startswith('nf-'))
            or native_functions, 'Native function exercises require their explicit contract')
    task = "hecate-function-synthesis-v1" if descriptor.get("schema") == 2 else "hecate-function-synthesis-v0"
    if translation.get("static_check", {}).get("contract") == "hecate-function-v2":
        task = "hecate-function-synthesis-v2"
    if translation.get("static_check", {}).get("contract") == "hecate-function-v3":
        task = "hecate-function-synthesis-v4"
    if translation.get('static_check', {}).get('contract') == 'hecate-function-v4':
        task = 'hecate-function-synthesis-v5'
    if extended_arithmetic:
        task = 'hecate-function-synthesis-v6'
    if public_construction:
        task = 'hecate-function-synthesis-v7'
    if function_composition:
        task = 'hecate-function-synthesis-v8'
    if closures:
        task = 'hecate-function-synthesis-v9'
    if call_binding:
        task = 'hecate-function-synthesis-v10'
    if public_iteration:
        task = 'hecate-function-synthesis-v11'
    if function_literals:
        task = 'hecate-function-synthesis-v12'
    if public_sequences:
        task = 'hecate-function-synthesis-v13'
    if public_numbers:
        task = 'hecate-function-synthesis-v14'
    if public_control:
        task = 'hecate-function-synthesis-v15'
    if public_strings:
        task = 'hecate-function-synthesis-v16'
    if public_polynomial:
        task = 'hecate-function-synthesis-v17'
    if object_arrays:
        task = 'hecate-function-synthesis-v18'
    if public_mappings or construction_exercise is not None:
        task = 'hecate-function-synthesis-v19'
    if object_arithmetic:
        require(construction_exercise is None or construction_exercise.startswith('oa-'),
                'Frozen v19 exercises cannot change their contract')
        task = 'hecate-function-synthesis-v20'
    if scalar_conversion:
        require(construction_exercise is None or construction_exercise.startswith('sc-'),
                'Frozen older exercises cannot change their contract')
        task = 'hecate-function-synthesis-v21'
    if object_unary:
        require(construction_exercise is None or construction_exercise.startswith('ou-'),
                'Frozen earlier exercises cannot use object unary contract')
        task = 'hecate-function-synthesis-v22'
    if native_functions:
        require(not any((extended_arithmetic, public_construction, function_composition, closures, call_binding,
                public_iteration, function_literals, public_sequences, public_numbers, public_control, public_strings,
                public_polynomial, object_arrays, public_mappings, object_arithmetic, scalar_conversion, object_unary))
                and (construction_exercise is None or construction_exercise.startswith(native_prefix if native_arrays else 'nf-')),
                'Native function contract is separate; do not mix grammar flags')
        task = NATIVE_EXERCISE_TASK if construction_exercise is not None else NATIVE_TASK
        if native_arrays:
            task = ARRAY_EXERCISE_TASK if construction_exercise is not None else NATIVE_ARRAY_TASK
        if native_starred:
            task = STAR_EXERCISE_TASK if construction_exercise is not None else STAR_TASK
        if native_array_arithmetic:
            task = ARITHMETIC_TASK
        if native_public_loops:
            task = LOOP_TASK
        if native_scalar_augmented:
            task = AUGMENTED_TASK
        if native_array_mutation:
            task = MUTATION_TASK
    if descriptor.get('schema') == 4:
        require(task in ('hecate-function-synthesis-v4', 'hecate-function-synthesis-v5'),
                'Chunked input v1 cannot silently combine experimental grammar flags')
        task = CHUNKED_TASK
    if descriptor.get('schema')==5:
        require((task=='hecate-function-synthesis-v0' or task==MUTATION_TASK and native_array_mutation) and
                (construction_exercise is None or packed_exercise) and
                translation.get('static_check',{}).get('contract')==PACKED_CONTRACT,
                'Packed input cannot silently mix experimental grammar or legacy translation')
        task=PACKED_NATIVE_EXERCISE_TASK if packed_exercise else PACKED_NATIVE_TASK if native_array_mutation else PACKED_TASK
    body = dict(schema=1, task=task, model=descriptor,
                fx_graph=translation["fx_graph"], public_constants=translation["public_constants"],
                constant_origins=translation["constant_origins"], layout=translation["layout"],
                rules=TASK_RULES[task][1], response_schema=RESPONSE_SCHEMA,
                semantic_guidance=SEMANTIC_GUIDANCE if native_functions else SEMANTIC_GUIDANCE_V22 if object_unary else SEMANTIC_GUIDANCE_V21 if scalar_conversion else SEMANTIC_GUIDANCE_V20 if object_arithmetic else SEMANTIC_GUIDANCE_V19 if (public_mappings or construction_exercise is not None) else
                    SEMANTIC_GUIDANCE_V18 if object_arrays else
                    SEMANTIC_GUIDANCE_V17 if public_polynomial else
                    SEMANTIC_GUIDANCE_V16 if public_strings else
                    SEMANTIC_GUIDANCE_V15 if public_control else
                    SEMANTIC_GUIDANCE_V14 if public_numbers else
                    SEMANTIC_GUIDANCE_V13 if public_sequences else
                    SEMANTIC_GUIDANCE_V12 if function_literals else
                    SEMANTIC_GUIDANCE_V11 if public_iteration else
                    SEMANTIC_GUIDANCE_V10 if call_binding else
                    SEMANTIC_GUIDANCE_V9 if closures else
                    SEMANTIC_GUIDANCE_V8 if function_composition else
                    SEMANTIC_GUIDANCE_V7 if public_construction else
                    SEMANTIC_GUIDANCE_V6 if extended_arithmetic else SEMANTIC_GUIDANCE,
                compiler_profile_sha256=profile_hash,
                privacy={"input": "encrypted", "weights": "public"})
    if task == CHUNKED_TASK:
        body['semantic_guidance'] = SEMANTIC_GUIDANCE_V6
        validate_request_binding(body)
    if task in PACKED_TASKS:
        from packed_input_abi import validate_request
        body['semantic_guidance']=(PACKED_GUIDANCE_V2 if any(n.get('op') in ('permute','transpose')
            for n in descriptor.get('nodes',[])) else PACKED_GUIDANCE)
        validate_request(body)
        if task in PACKED_NATIVE_TASKS:body['semantic_guidance']=PACKED_NATIVE_GUIDANCE
    if model_structure is not None:
        body["model_structure"] = model_structure
    if compiler_configuration is not None:
        from compiler_configuration import validate_configuration
        body['compiler_configuration'] = validate_configuration(compiler_configuration, profile_hash)
    if construction_exercise is not None:
        from construction_exercises import exercise_spec
        body["construction_exercise"] = exercise_spec(construction_exercise)
    request = copy.deepcopy(body)
    request["request_id"] = hashlib.sha256(canonical(body)).hexdigest()
    require(len(canonical(request)) <= MAX_BYTES, "Request size limit")
    return request


def validate_candidate(candidate, request):
    from compiler_configuration import request_configuration
    request_configuration(request)
    require(type(candidate) is dict and set(candidate) == set(RESPONSE_SCHEMA["required"]),
            "Candidate must contain only schema, request_id, hecate_source")
    require(type(candidate["schema"]) is int and candidate["schema"] == 1, "Unsupported response schema")
    require(type(candidate["request_id"]) is str and candidate["request_id"] == request["request_id"],
            "Candidate belongs to another immutable request")
    task = request.get("task")
    require(type(task) is str and task in TASK_RULES and request.get("rules") == TASK_RULES[task][1],
            "Unknown or inconsistent request contract")
    require(valid_semantic_guidance(request), 'Unknown or changed semantic guidance')
    check = validate_function(candidate["hecate_source"], request["public_constants"],
                              request["layout"]["output_ciphertexts"], contract=TASK_RULES[task][0],
                              input_names=request_input_names(request),
                              slot_period=request['layout']['input_slot_period'] if task in PACKED_TASKS else 4)
    # Smaller operation bound for generated candidates than the generic AST cap.
    require(sum(check["operator_counts"].values()) <= (1024 if task in PACKED_TASKS else 256), "Candidate operation budget exceeded")
    if "construction_exercise" in request:
        from construction_exercises import validate_exercise_request, check_exercise
        validate_exercise_request(request)
        check["construction_exercise"] = check_exercise(candidate["hecate_source"], request)
    return check


class ReplayProvider:
    """Scripted responses, explicitly NOT model inference or feedback reasoning.

    Future adapters implement generate(request, feedback_history) -> raw JSON.
    No arbitrary command/provider plugin loading or credential discovery.
    """
    kind = "scripted_replay"
    agent_calls = 0

    def __init__(self, responses):
        require(type(responses) is list and 1 <= len(responses) <= MAX_REPAIRS + 1,
                "Replay must have 1..4 responses")
        require(all(type(x) is str and len(x.encode()) <= MAX_BYTES for x in responses),
                "Replay response size/type limit")
        self.responses = list(responses)
        self.index = 0

    def generate(self, request, feedback_history):
        if self.index >= len(self.responses):
            raise StopIteration("Replay exhausted; no generated repair available")
        response = self.responses[self.index]
        self.index += 1
        return response


def run_feedback_loop(request, provider, evaluate, record, max_repairs=MAX_REPAIRS):
    """Bounded orchestration; evaluator/record are trusted host capabilities.

    Host receives raw strings only. Generator receives copies of public data,
    never a compiler callback, filesystem path, key or reference object.
    """
    require(type(max_repairs) is int and 0 <= max_repairs <= MAX_REPAIRS, "Repair limit is 0..3")
    history = []
    terminal = "repair_budget_exhausted"
    for index in range(max_repairs + 1):
        try:
            raw = provider.generate(copy.deepcopy(request), copy.deepcopy(history))
        except StopIteration:
            terminal = "provider_exhausted"
            break
        except Exception:
            terminal = "provider_failed"
            break  # Do not echo provider errors which might include credentials.
        feedback = evaluate(raw, index)
        history.append(copy.deepcopy(feedback))
        record(index, raw, copy.deepcopy(feedback))
        if feedback["status"] == "passed":
            terminal = "passed"
            break
        if feedback.get("category") in ("infrastructure", "integrity"):
            terminal = feedback["category"] + "_failed"
            break
    result = dict(status=terminal, provider=provider.kind, agent_calls=provider.agent_calls,
                attempts=len(history), repairs_used=max(0, len(history)-1),
                first_attempt_passed=bool(history and history[0]["status"] == "passed"),
                final_passed=terminal == "passed", feedback_history=history,
                agent_success_rate=None,
                note=("Replay metrics are not Agent generation metrics" if provider.kind == "scripted_replay"
                      else "Provider metrics alone are not evidence of encrypted semantic correctness"))
    if callable(getattr(provider, "metrics", None)):
        result["provider_metrics"] = provider.metrics()
    return result
