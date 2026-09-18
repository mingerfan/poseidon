"""Explicit opt-in contract for native decorated calls; old contracts unchanged."""

# Independent composition contract; no changes to historical native prompts.
PACKED_NATIVE_RULES = '''Generate only decorated Hecate functions: golden and at most16 helpers.
Use exactly @hc.func("c,p,...") with one literal comma-separated string per function.
Every parameter is a positional c (ciphertext Expr) or p (Plain Expr). Golden uses
x, then zero_ct ONLY if layout.auxiliary_ciphertexts declares it; both are c.
Helpers can be declared later and called transitively, but no recursion or calls
to golden. At most16 parameters; no defaults, annotations, kwargs or varargs.

Model schema5 has one encrypted logical tensor of rank1..4 and <=256 elements.
The client flattens C-order, zero-pads to P=layout.input_slot_period, and repeats
that period across16384 CKKS slots. P is4..256, not necessarily4. Public constants
are read-only Hecate Plain scalar/length1 or lengthP vectors. Only finite real
literals of magnitude<=1024 and the supplied public names are available.
Cipher arithmetic is +,-,*, unary -, and rotate(step). Positive rotate reads
x[(j+step) mod P] at j. Allowed steps are ONLY positive powers of two less than P.
Compose steps for other displacements. Preserve the full graph and node layouts;
Linear needs cross-slot reduction, not elementwise multiplication. Do not omit
permutation before Conv/Linear. Use encrypted zero_ct for exact-zero results
when supplied; do not manufacture ciphertext zero as x-x or x*0.

Use assignments, flat list/tuple returns/unpacking, and integer indexing of
containers (not slots). np.array(data,dtype=object) supports rectangular nested
containers and scalar Expr data; outer storage rank<=4 and cells<=16. Each cell
is a WHOLE Expr, not a ciphertext slot. Allowed array reads: literal integer,
slice, tuple, Ellipsis and None/newaxis; methods: C-order reshape (one -1),
flatten(),copy(),transpose(),.T,item(). No dynamic/fancy/boolean indexing or
subscript writes. Golden returns exactly layout.output_ciphertexts cipher Exprs;
array returns flatten C-order. packed_prefix returns one ciphertext, whereas
scalar_neurons returns separate ciphertexts per logical element.
Helpers accept only scalar Expr arguments, not arrays. Starred positional calls
expand one list/tuple or array FIRST AXIS into scalar c/p arguments; no implicit
flattening. Empty storage expands to zero arguments. No keyword expansion.

Non-mutating array +,-,* require an array on the LEFT; right may be array or
scalar Expr. NumPy trailing-axis broadcasting applies to outer storage only;
each arithmetic pair needs a ciphertext. Unary minus needs ciphertext cells.
Zero-dimensional array arithmetic returns a scalar Expr, not an ndarray.
Name-target scalar +=,-=,*= rebind to new Exprs; existing aliases stay unchanged.
Name-target ndarray +=,-=,*= mutate shared object-array storage in place. Views
see changes; copies do not. The result shape cannot expand. Existing Expr cells
are not mutated; array entries are replaced. Plain/Plain operations are forbidden.

Public for i in range(...) loops are construction-time, never ciphertext-dependent.
range takes1..3 public integers from literals or earlier loop indices; integer
+,-,*,//,% and signs are folded before Expr construction. Range<=128 iterations,
total expanded AST<=4096 and nesting<=16. Index bindings are read-only and cannot
shadow parameters, constants, functions,hc,np,object,range or zero_ct. No break,
continue,for-else,while,comprehensions or early return. Each function ends in return.
Native bodies are traced as real Hecate functions and calls are compiler-inlined;
no eval/exec of generated Python. Local names cannot shadow constants/functions.
No imports, I/O, arbitrary NumPy methods, bootstrap, parameter changes or secret
access. Compiler owns levels/scales/rescale/relinearization; client owns keys.
Keep reference, test inputs, weights, security and tolerances immutable. Static
acceptance is not numerical proof; compile and real encrypted comparison must pass.
Source<=65536 bytes; total native expanded work<=4096, golden candidate work<=1024.
Return only the response-schema JSON. This contract composes native-v7 with the
periodic-packed ABI, not unrestricted Python or all upstream DSL semantics.
'''

TASK = 'hecate-native-function-synthesis-v1'
CONTRACT = 'hecate-native-functions-v1'
RULES = '''Generate Hecate Python function declarations only: golden plus up to 16 named
helpers, every function decorated exactly @hc.func("c,p,...") on its own line.
Each c means a ciphertext argument, each p means a Hecate Plain argument.
Use exactly the golden input names and order from the layout: x/y/z/t, with
zero_ct appended only when the layout declares it; all golden inputs are c.
Helpers may have zero arguments using @hc.func(""). All arguments are positional,
without defaults, annotations, kwargs or varargs. Helper names and local names
must not shadow supplied public constants, other functions or hc.
Use straight-line assignments, flat list/tuple construction, exact unpacking,
literal integer indexing of those containers, and a final return. Helpers may
return Expr, a flat list/tuple of Expr, or an empty list/tuple. Golden must return
exactly the declared number of ciphertexts, not Plain values. A list denotes
multiple ciphertexts, not the slots of one ciphertext.
Arithmetic: +,-,* with at least one ciphertext operand, unary -ciphertext,
and ciphertext.rotate(step) with step in -3,-2,-1,1,2,3. Positive rotation is left.
Use supplied public constant names or finite real literals of magnitude <=1024.
Supplied public constants have scalar/length-one or period-four layout and are
read-only. Helpers can accept them through p arguments; never pass cipher as p.
All logical inputs have repeated period-four packed slots, C-order for shapes.
Return selectors and packing are fixed by the harness. A Linear reduction needs
real rotations/sums; do not replace it with elementwise multiplication.
Forward and nested calls to declared helpers are allowed. No recursion, calls to
golden, imports, eval/exec, I/O, bootstrap, network, arbitrary attributes, control
flow, nested definitions, ordinary undecorated helpers, lambdas, array methods,
public-only arithmetic or augmented assignment in this initial native contract.
This is separate from the older Python construction contracts, not their union.
Helper bodies trace once, then their verified IR is statically inlined at calls.
Do not rely on Python side effects, mutable captured globals or callable values.
Compiler owns scale/level/rescale/relinearization. Never change reference, inputs,
weights, tolerance, security parameters or compiler settings. Implement the
supplied public model, not hidden test values. Return only the response-schema JSON.
'''

# Preserve v1 request text/hashes; exercises are an explicit new prompt version.
EXERCISE_TASK = 'hecate-native-function-synthesis-v2'
EXERCISE_RULES = RULES + '''When construction_exercise is present, also implement its
frozen construction requirement without changing the supplied model semantics.
Required calls must be reachable from golden, not unused declarations. Except
for the explicitly structural empty-return exercise, required return values and
specified parameters must affect the final output. This cohort tests constrained
implementation forms, not unrestricted synthesis. Never pad with dead calls.
'''

ARRAY_TASK = 'hecate-native-function-synthesis-v3'
ARRAY_CONTRACT = 'hecate-native-functions-v2'
ARRAY_RULES = RULES.replace('flat list/tuple construction',
    'list/tuple construction (nested only as object-array data)').replace(
    'or an empty list/tuple. Golden must return',
    'an empty list/tuple, or an object ndarray. Golden must return').replace(
    'lambdas, array methods,', 'lambdas, unlisted array methods,') + '''
This version additionally permits np.array(data, dtype=object), including an Expr
as zero-dimensional data or rectangular nested list/tuple data. Each array cell
is a separate Expr (cipher or Plain), NOT a packed slot. Golden return arrays are
flattened in C order into the declared ciphertext results; all final cells must
be ciphertexts. Maximum storage rank is four and maximum cell count is sixteen.
Array reads support literal integer/slice indices, tuple indices, Ellipsis and
None/newaxis. No boolean/fancy/dynamic indexing or cell writes. Allowed storage
methods are reshape with literal dimensions (one -1 allowed), flatten(), copy(),
transpose with literal axes (or no args), .T, and item with literal integer/tuple
indices (or no args for size one). Only C order; no keyword method arguments.
Array unpacking follows the first axis, not flattened storage. Zero-dimensional
arrays are not iterable; use item() or [()] to extract their single Expr.
Array-valued helper arguments, array arithmetic, mutation, np.empty/full/asarray,
and all other NumPy calls remain unsupported in this version. np and object are
reserved names. Storage reordering never rotates the slots inside a ciphertext.
'''

# New exercise prompt; preserve v3 request bytes for prior manual evidence.
ARRAY_EXERCISE_TASK = 'hecate-native-function-synthesis-v4'
ARRAY_EXERCISE_RULES = ARRAY_RULES + '''When construction_exercise is present, implement
all its required array forms in the reachable computation. Except the explicitly
structural empty-array exercise, operation results must contribute to golden.
For mixed returns both Plain and ciphertext cells must contribute. Dead calls,
unused storage operations and algebraically cancelled padding do not count.
This is finite construction coverage, not a request to alter model semantics.
'''

STAR_TASK = 'hecate-native-function-synthesis-v5'
STAR_CONTRACT = 'hecate-native-functions-v3'
STAR_RULES = ARRAY_RULES + '''This version additionally permits starred POSITIONAL
arguments to declared decorated helpers: helper(*items), helper(x,*items), and
multiple starred segments. Evaluate arguments left-to-right. Expand list/tuple
or object-array storage by exactly one iterable level, preserving its order.
Each expanded argument must be one Expr of the callee's declared c/p kind.
Array iteration is along its first axis, never implicit flattening: zero-dimensional
arrays cannot be unpacked, and matrix rows are arrays, not scalar Expr arguments.
Explicit .flatten() or .reshape(...) may create one-dimensional Expr storage.
Empty list/tuple or a one-dimensional empty array contributes zero arguments.
At most sixteen total expanded arguments. This is argument unpacking, not a new
array-valued native ABI. No keyword arguments, **kwargs, generator expressions,
starred function definitions, starred return/list construction or starred calls
to NumPy/storage methods/rotate are introduced by this version.
'''

ARITHMETIC_TASK = 'hecate-native-function-synthesis-v6'
ARITHMETIC_CONTRACT = 'hecate-native-functions-v4'
ARITHMETIC_RULES = STAR_RULES.replace('array-valued helper arguments, array arithmetic,',
    'array-valued helper arguments,').replace('Array-valued helper arguments, array arithmetic,',
    'Array-valued helper arguments,') + '''
This version additionally permits non-mutating object-array +, -, * and unary -.
The left binary operand must be an object ndarray; the right may be an object
ndarray or one scalar Expr. An Expr on the left with an array on the right is
NOT supported: Hecate resolves its operand before NumPy can broadcast it.
NumPy trailing-axis broadcasting applies to the OUTER array of Expr cells, not
the slots in a ciphertext. Every broadcast pair must contain a ciphertext;
Plain/Plain arithmetic and negating Plain cells remain forbidden. Result size
and rank are still at most sixteen cells and four dimensions. Unary - requires
all ciphertext cells. A zero-dimensional object arithmetic result is a scalar
Expr, not a zero-dimensional ndarray: do not index it or call .item() on it.
Empty-array arithmetic produces empty storage. Non-mutating operations preserve
input storage and aliases. No in-place operators, cell writes, division, power,
matrix multiplication, arbitrary NumPy ufuncs or new array-valued ABI are added.
'''

LOOP_TASK = 'hecate-native-function-synthesis-v7'
LOOP_CONTRACT = 'hecate-native-functions-v5'
LOOP_RULES = ARITHMETIC_RULES.replace('control\nflow, nested definitions',
    'unlisted control\nflow, nested definitions') + '''
This version additionally permits for i in range(...): inside decorated helpers
and golden. range takes 1..3 public integer arguments (no keywords). Bounds may
use literal integers and already bound induction indices, with public integer
+,-,*,//,% and unary signs. These pure integer expressions are folded before
Expr construction. No ciphertext or Plain parameter determines a range.
Loops are expanded during checked construction; native helper boundaries stay
intact. Each range has at most 128 elements; total iterations and expanded AST
are bounded by 4096. Integers have magnitude at most 1048576 and nesting <=16.
Induction names must not shadow parameters, constants, functions, range, hc, np,
object or zero_ct. They are read-only: no explicit assignment anywhere in that
function, and nested loops cannot reuse an active index. Sequential reuse is
allowed. Indices remain bound to their last value after a nonempty loop; an empty
loop leaves a preceding binding unchanged or leaves the index undefined.
Use ordinary accumulator assignment such as acc=acc+term. No augmented assignment,
break, continue, for-else, early return inside a loop, while, comprehensions or
ciphertext-dependent control flow is added. The final return remains mandatory.
'''

# Independent targeted-star contract; previous v1..v7 text remains frozen.
STAR_EXERCISE_TASK = 'hecate-native-function-synthesis-v8'
STAR_EXERCISE_RULES = STAR_RULES + '''When construction_exercise is present, implement
all specified starred argument forms in calls reachable from golden. For each
required nonempty call, EVERY positional argument (ordinary and expanded) must
affect golden's final output. Do not ignore arguments or pad with cancelled terms.
Empty expansion is explicitly trace-structural only. Preserve model semantics;
construction coverage is not a substitute for encrypted numerical comparison.
'''

# Scalar native augmented dispatch; previous contracts remain unchanged.
AUGMENTED_TASK = 'hecate-native-function-synthesis-v9'
AUGMENTED_CONTRACT = 'hecate-native-functions-v6'
AUGMENTED_RULES = LOOP_RULES + '''
Override the preceding prohibition only for scalar Expr name-target augmented
assignment: name += expr, name -= expr, name *= expr. The name must already be
bound to one c/p Expr, the RHS must also be one c/p Expr, and at least one operand
must be ciphertext. Numeric literals are resolved as Plain. Result is ciphertext.
Evaluate the existing LHS first, then the RHS once, preserving subtraction order.
Use actual Hecate augmented operators: they return a new Expr and rebind the name;
other names and array cells holding the original Expr are NOT mutated. A Plain
local can become ciphertext; public constant registry names remain read-only.
Loops and native calls follow the preceding contract, including read-only indices.
Object arrays/list/tuple are not scalar Exprs. Array augmented assignment and
subscript/attribute writes remain rejected pending alias-aware mutation support.
No division, power, encrypted branching, Empty accumulator or implicit bootstrap.
'''

MUTATION_TASK = 'hecate-native-function-synthesis-v10'
MUTATION_CONTRACT = 'hecate-native-functions-v7'
MUTATION_RULES = AUGMENTED_RULES + '''
This version additionally permits name +=/-=/*= RHS when name refers to an object
ndarray. This overrides the preceding array-augmented prohibition only. The RHS
may be one scalar Expr or a compatible object ndarray. NumPy trailing-axis
broadcasting must fit the EXISTING target shape, never resize it. Each broadcast
cell pair must include ciphertext. All resulting target cells become ciphertext.
The target ndarray identity and shape are preserved, even for a 0-D ndarray.
Ordinary aliases and basic slice/transpose/reshape views observe changed cells.
Scalar Exprs previously extracted from cells remain unchanged. copy() and flatten()
allocate separate storage; reshape may share or copy according to NumPy strides.
The checker tracks these aliases and c/p transitions before permitting execution.
np.array(data,dtype=object) allocates fresh storage. Each native helper call returns
fresh reconstructed array storage; changing one call result does not change a
different call's result. Overlapping operands use actual NumPy inplace behavior.
Empty array updates have structural meaning only, no numerical output cells.
Subscript/attribute writes, list mutation, resized arrays, kwargs, arbitrary
NumPy calls and new array-valued helper parameters remain unsupported.
'''
