ŤŤ
OĎN
.
Abs
x"T
y"T"
Ttype:

2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
E
AssignAddVariableOp
resource
value"dtype"
dtypetype
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ŕ
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ŕ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

j
	DecodeRaw	
bytes
output"out_type"#
out_typetype:
2	
"
little_endianbool(
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Ž
FIFOQueueV2

handle"!
component_types
list(type)(0"
shapeslist(shape)
 ("
capacityint˙˙˙˙˙˙˙˙˙"
	containerstring "
shared_namestring 
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
A
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
Ž
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
ş
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
2
L2Loss
t"T
output"T"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
$

LogicalAnd
x

y

z


MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 

ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
8
Pow
x"T
y"T
z"T"
Ttype:
2
	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
B
QueueCloseV2

handle"#
cancel_pending_enqueuesbool( 

QueueDequeueManyV2

handle
n

components2component_types"!
component_types
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
}
QueueEnqueueManyV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
y
QueueEnqueueV2

handle

components2Tcomponents"
Tcomponents
list(type)(0"

timeout_msint˙˙˙˙˙˙˙˙˙
&
QueueSizeV2

handle
size
Y
RandomShuffle

value"T
output"T"
seedint "
seed2int "	
Ttype
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
)
Rank

input"T

output"	
Ttype
@
ReadVariableOp
resource
value"dtype"
dtypetype
J
ReaderReadV2
reader_handle
queue_handle
key	
value
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
7

Reciprocal
x"T
y"T"
Ttype:
2
	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
ŕ
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
t
	ScatterNd
indices"Tindices
updates"T
shape"Tindices
output"T"	
Ttype"
Tindicestype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
1
Sign
x"T
y"T"
Ttype:
2
	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ŕ
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
|
TFRecordReaderV2
reader_handle"
	containerstring "
shared_namestring "
compression_typestring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
Á
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized

E
Where

input"T	
index	"%
Ttype0
:
2	
"train"serve*2.10.12v2.10.0-76-gfdfc646704c8˙

input_producer/ConstConst*
_output_shapes
:*
dtype0*K
valueBB@B6C:\Users\User\Desktop\MTCNN\tmp/data/pnet\all.tfrecord
U
input_producer/SizeConst*
_output_shapes
: *
dtype0*
value	B :
Z
input_producer/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 
q
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0*
_output_shapes
: 

input_producer/Assert/ConstConst*
_output_shapes
: *
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor

#input_producer/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor
o
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*

T
2
}
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0*
_output_shapes
:
k
input_producer/RandomShuffleRandomShuffleinput_producer/Identity*
T0*
_output_shapes
:
~
input_producerFIFOQueueV2"/device:CPU:**
_output_shapes
: *
capacity *
component_types
2*
shapes
: 

)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/RandomShuffle*
Tcomponents
2
C
#input_producer/input_producer_CloseQueueCloseV2input_producer
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
Y
"input_producer/input_producer_SizeQueueSizeV2input_producer*
_output_shapes
: 
o
input_producer/CastCast"input_producer/input_producer_Size*

DstT0*

SrcT0*
_output_shapes
: 
Y
input_producer/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   =
e
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0*
_output_shapes
: 

'input_producer/fraction_of_32_full/tagsConst*
_output_shapes
: *
dtype0*3
value*B( B"input_producer/fraction_of_32_full

"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0*
_output_shapes
: 
<
TFRecordReaderV2TFRecordReaderV2*
_output_shapes
: 
X
ReaderReadV2ReaderReadV2TFRecordReaderV2input_producer*
_output_shapes
: : 
h
%ParseSingleExample/ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
j
'ParseSingleExample/ParseExample/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
j
'ParseSingleExample/ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
j
'ParseSingleExample/ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
w
4ParseSingleExample/ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
}
:ParseSingleExample/ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB 
ˇ
9ParseSingleExample/ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*J
valueAB?Bimage/encodedBimage/labelBimage/landmarkB	image/roi
}
:ParseSingleExample/ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 

.ParseSingleExample/ParseExample/ParseExampleV2ParseExampleV2ReaderReadV2:14ParseSingleExample/ParseExample/ParseExampleV2/names:ParseSingleExample/ParseExample/ParseExampleV2/sparse_keys9ParseSingleExample/ParseExample/ParseExampleV2/dense_keys:ParseSingleExample/ParseExample/ParseExampleV2/ragged_keys%ParseSingleExample/ParseExample/Const'ParseSingleExample/ParseExample/Const_1'ParseSingleExample/ParseExample/Const_2'ParseSingleExample/ParseExample/Const_3*
Tdense
2	*$
_output_shapes
: : :
:*"
dense_shapes
: : :
:*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 
{
	DecodeRaw	DecodeRaw.ParseSingleExample/ParseExample/ParseExampleV2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
out_type0
b
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         
Y
ReshapeReshape	DecodeRawReshape/shape*
T0*"
_output_shapes
:
Q
CastCastReshape*

DstT0*

SrcT0*"
_output_shapes
:
J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ˙B
D
subSubCastsub/y*
T0*"
_output_shapes
:
N
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C
O
truedivRealDivsub	truediv/y*
T0*"
_output_shapes
:
p
Cast_1Cast0ParseSingleExample/ParseExample/ParseExampleV2:1*

DstT0*

SrcT0	*
_output_shapes
: 
M
batch/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z

batch/fifo_queueFIFOQueueV2"/device:CPU:**
_output_shapes
: *
capacity*
component_types
2*(
shapes
:: ::

Ę
batch/fifo_queue_enqueueQueueEnqueueV2batch/fifo_queuetruedivCast_10ParseSingleExample/ParseExample/ParseExampleV2:30ParseSingleExample/ParseExample/ParseExampleV2:2*
Tcomponents
2
8
batch/fifo_queue_CloseQueueCloseV2batch/fifo_queue
Y
batch/fifo_queue_Close_1QueueCloseV2batch/fifo_queue*
cancel_pending_enqueues(
N
batch/fifo_queue_SizeQueueSizeV2batch/fifo_queue*
_output_shapes
: 
Y

batch/CastCastbatch/fifo_queue_Size*

DstT0*

SrcT0*
_output_shapes
: 
P
batch/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŤŞ*;
J
	batch/mulMul
batch/Castbatch/mul/y*
T0*
_output_shapes
: 
z
batch/fraction_of_384_full/tagsConst*
_output_shapes
: *
dtype0*+
value"B  Bbatch/fraction_of_384_full
x
batch/fraction_of_384_fullScalarSummarybatch/fraction_of_384_full/tags	batch/mul*
T0*
_output_shapes
: 
J
batch/nConst*
_output_shapes
: *
dtype0*
value
B :

batchQueueDequeueManyV2batch/fifo_queuebatch/n*D
_output_shapes2
0:::	:	
*
component_types
2
Z
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
T
	Reshape_1Reshapebatch:1Reshape_1/shape*
T0*
_output_shapes	
:
`
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"     
X
	Reshape_2Reshapebatch:2Reshape_2/shape*
T0*
_output_shapes
:	
`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"  
   
X
	Reshape_3Reshapebatch:3Reshape_3/shape*
T0*
_output_shapes
:	

n
input_imagePlaceholder*'
_output_shapes
:*
dtype0*
shape:
P
labelPlaceholder*
_output_shapes	
:*
dtype0*
shape:
^
bbox_targetPlaceholder*
_output_shapes
:	*
dtype0*
shape:	
b
landmark_targetPlaceholder*
_output_shapes
:	
*
dtype0*
shape:	

Š
.conv1/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv1/weights*
_output_shapes
:*
dtype0*%
valueB"         
   

,conv1/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv1/weights*
_output_shapes
: *
dtype0*
valueB
 *íăgž

,conv1/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv1/weights*
_output_shapes
: *
dtype0*
valueB
 *íăg>
×
6conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv1/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
:
*
dtype0
Ň
,conv1/weights/Initializer/random_uniform/subSub,conv1/weights/Initializer/random_uniform/max,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*
_output_shapes
: 
ě
,conv1/weights/Initializer/random_uniform/mulMul6conv1/weights/Initializer/random_uniform/RandomUniform,conv1/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
:

ŕ
(conv1/weights/Initializer/random_uniformAddV2,conv1/weights/Initializer/random_uniform/mul,conv1/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv1/weights*&
_output_shapes
:

 
conv1/weightsVarHandleOp* 
_class
loc:@conv1/weights*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1/weights
k
.conv1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/weights*
_output_shapes
: 
n
conv1/weights/AssignAssignVariableOpconv1/weights(conv1/weights/Initializer/random_uniform*
dtype0
w
!conv1/weights/Read/ReadVariableOpReadVariableOpconv1/weights*&
_output_shapes
:
*
dtype0
r
-conv1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:

=conv1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpconv1/weights*&
_output_shapes
:
*
dtype0

.conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss=conv1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
Ž
'conv1/kernel/Regularizer/l2_regularizerMul-conv1/kernel/Regularizer/l2_regularizer/scale.conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

conv1/biases/Initializer/zerosConst*
_class
loc:@conv1/biases*
_output_shapes
:
*
dtype0*
valueB
*    

conv1/biasesVarHandleOp*
_class
loc:@conv1/biases*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1/biases
i
-conv1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/biases*
_output_shapes
: 
b
conv1/biases/AssignAssignVariableOpconv1/biasesconv1/biases/Initializer/zeros*
dtype0
i
 conv1/biases/Read/ReadVariableOpReadVariableOpconv1/biases*
_output_shapes
:
*
dtype0
q
conv1/Conv2D/ReadVariableOpReadVariableOpconv1/weights*&
_output_shapes
:
*
dtype0

conv1/Conv2DConv2Dinput_imageconv1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:


*
paddingVALID*
strides

e
conv1/BiasAdd/ReadVariableOpReadVariableOpconv1/biases*
_output_shapes
:
*
dtype0
v
conv1/BiasAddBiasAddconv1/Conv2Dconv1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:




conv1/alphas/Initializer/ConstConst*
_class
loc:@conv1/alphas*
_output_shapes
:
*
dtype0*
valueB
*  >

conv1/alphasVarHandleOp*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1/alphas
i
-conv1/alphas/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/alphas*
_output_shapes
: 
b
conv1/alphas/AssignAssignVariableOpconv1/alphasconv1/alphas/Initializer/Const*
dtype0
i
 conv1/alphas/Read/ReadVariableOpReadVariableOpconv1/alphas*
_output_shapes
:
*
dtype0
S

conv1/ReluReluconv1/BiasAdd*
T0*'
_output_shapes
:



Q
	conv1/AbsAbsconv1/BiasAdd*
T0*'
_output_shapes
:



\
	conv1/subSubconv1/BiasAdd	conv1/Abs*
T0*'
_output_shapes
:



]
conv1/ReadVariableOpReadVariableOpconv1/alphas*
_output_shapes
:
*
dtype0
c
	conv1/mulMulconv1/ReadVariableOp	conv1/sub*
T0*'
_output_shapes
:



R
conv1/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
^
conv1/mul_1Mul	conv1/mulconv1/mul_1/y*
T0*'
_output_shapes
:



]
	conv1/addAddV2
conv1/Reluconv1/mul_1*
T0*'
_output_shapes
:



Y
StopGradientStopGradient	conv1/add*
T0*'
_output_shapes
:



k
)conv1/add/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 

pool1/MaxPoolMaxPool	conv1/add*'
_output_shapes
:
*
ksize
*
paddingSAME*
strides

_
StopGradient_1StopGradientpool1/MaxPool*
T0*'
_output_shapes
:

o
-pool1/MaxPool/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
Š
.conv2/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv2/weights*
_output_shapes
:*
dtype0*%
valueB"      
      

,conv2/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
valueB
 *˘ř#ž

,conv2/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
valueB
 *˘ř#>
×
6conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv2/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:
*
dtype0
Ň
,conv2/weights/Initializer/random_uniform/subSub,conv2/weights/Initializer/random_uniform/max,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*
_output_shapes
: 
ě
,conv2/weights/Initializer/random_uniform/mulMul6conv2/weights/Initializer/random_uniform/RandomUniform,conv2/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:

ŕ
(conv2/weights/Initializer/random_uniformAddV2,conv2/weights/Initializer/random_uniform/mul,conv2/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:

 
conv2/weightsVarHandleOp* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2/weights
k
.conv2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/weights*
_output_shapes
: 
n
conv2/weights/AssignAssignVariableOpconv2/weights(conv2/weights/Initializer/random_uniform*
dtype0
w
!conv2/weights/Read/ReadVariableOpReadVariableOpconv2/weights*&
_output_shapes
:
*
dtype0
r
-conv2/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:

=conv2/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpconv2/weights*&
_output_shapes
:
*
dtype0

.conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss=conv2/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
Ž
'conv2/kernel/Regularizer/l2_regularizerMul-conv2/kernel/Regularizer/l2_regularizer/scale.conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

conv2/biases/Initializer/zerosConst*
_class
loc:@conv2/biases*
_output_shapes
:*
dtype0*
valueB*    

conv2/biasesVarHandleOp*
_class
loc:@conv2/biases*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2/biases
i
-conv2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/biases*
_output_shapes
: 
b
conv2/biases/AssignAssignVariableOpconv2/biasesconv2/biases/Initializer/zeros*
dtype0
i
 conv2/biases/Read/ReadVariableOpReadVariableOpconv2/biases*
_output_shapes
:*
dtype0
q
conv2/Conv2D/ReadVariableOpReadVariableOpconv2/weights*&
_output_shapes
:
*
dtype0

conv2/Conv2DConv2Dpool1/MaxPoolconv2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:*
paddingVALID*
strides

e
conv2/BiasAdd/ReadVariableOpReadVariableOpconv2/biases*
_output_shapes
:*
dtype0
v
conv2/BiasAddBiasAddconv2/Conv2Dconv2/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:

conv2/alphas/Initializer/ConstConst*
_class
loc:@conv2/alphas*
_output_shapes
:*
dtype0*
valueB*  >

conv2/alphasVarHandleOp*
_class
loc:@conv2/alphas*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2/alphas
i
-conv2/alphas/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/alphas*
_output_shapes
: 
b
conv2/alphas/AssignAssignVariableOpconv2/alphasconv2/alphas/Initializer/Const*
dtype0
i
 conv2/alphas/Read/ReadVariableOpReadVariableOpconv2/alphas*
_output_shapes
:*
dtype0
S

conv2/ReluReluconv2/BiasAdd*
T0*'
_output_shapes
:
Q
	conv2/AbsAbsconv2/BiasAdd*
T0*'
_output_shapes
:
\
	conv2/subSubconv2/BiasAdd	conv2/Abs*
T0*'
_output_shapes
:
]
conv2/ReadVariableOpReadVariableOpconv2/alphas*
_output_shapes
:*
dtype0
c
	conv2/mulMulconv2/ReadVariableOp	conv2/sub*
T0*'
_output_shapes
:
R
conv2/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
^
conv2/mul_1Mul	conv2/mulconv2/mul_1/y*
T0*'
_output_shapes
:
]
	conv2/addAddV2
conv2/Reluconv2/mul_1*
T0*'
_output_shapes
:
[
StopGradient_2StopGradient	conv2/add*
T0*'
_output_shapes
:
k
)conv2/add/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
Š
.conv3/weights/Initializer/random_uniform/shapeConst* 
_class
loc:@conv3/weights*
_output_shapes
:*
dtype0*%
valueB"             

,conv3/weights/Initializer/random_uniform/minConst* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
valueB
 *ď[ń˝

,conv3/weights/Initializer/random_uniform/maxConst* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
valueB
 *ď[ń=
×
6conv3/weights/Initializer/random_uniform/RandomUniformRandomUniform.conv3/weights/Initializer/random_uniform/shape*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: *
dtype0
Ň
,conv3/weights/Initializer/random_uniform/subSub,conv3/weights/Initializer/random_uniform/max,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*
_output_shapes
: 
ě
,conv3/weights/Initializer/random_uniform/mulMul6conv3/weights/Initializer/random_uniform/RandomUniform,conv3/weights/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: 
ŕ
(conv3/weights/Initializer/random_uniformAddV2,conv3/weights/Initializer/random_uniform/mul,conv3/weights/Initializer/random_uniform/min*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: 
 
conv3/weightsVarHandleOp* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3/weights
k
.conv3/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/weights*
_output_shapes
: 
n
conv3/weights/AssignAssignVariableOpconv3/weights(conv3/weights/Initializer/random_uniform*
dtype0
w
!conv3/weights/Read/ReadVariableOpReadVariableOpconv3/weights*&
_output_shapes
: *
dtype0
r
-conv3/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:

=conv3/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpconv3/weights*&
_output_shapes
: *
dtype0

.conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss=conv3/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
Ž
'conv3/kernel/Regularizer/l2_regularizerMul-conv3/kernel/Regularizer/l2_regularizer/scale.conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

conv3/biases/Initializer/zerosConst*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0*
valueB *    

conv3/biasesVarHandleOp*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3/biases
i
-conv3/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/biases*
_output_shapes
: 
b
conv3/biases/AssignAssignVariableOpconv3/biasesconv3/biases/Initializer/zeros*
dtype0
i
 conv3/biases/Read/ReadVariableOpReadVariableOpconv3/biases*
_output_shapes
: *
dtype0
q
conv3/Conv2D/ReadVariableOpReadVariableOpconv3/weights*&
_output_shapes
: *
dtype0

conv3/Conv2DConv2D	conv2/addconv3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
: *
paddingVALID*
strides

e
conv3/BiasAdd/ReadVariableOpReadVariableOpconv3/biases*
_output_shapes
: *
dtype0
v
conv3/BiasAddBiasAddconv3/Conv2Dconv3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
: 

conv3/alphas/Initializer/ConstConst*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0*
valueB *  >

conv3/alphasVarHandleOp*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3/alphas
i
-conv3/alphas/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/alphas*
_output_shapes
: 
b
conv3/alphas/AssignAssignVariableOpconv3/alphasconv3/alphas/Initializer/Const*
dtype0
i
 conv3/alphas/Read/ReadVariableOpReadVariableOpconv3/alphas*
_output_shapes
: *
dtype0
S

conv3/ReluReluconv3/BiasAdd*
T0*'
_output_shapes
: 
Q
	conv3/AbsAbsconv3/BiasAdd*
T0*'
_output_shapes
: 
\
	conv3/subSubconv3/BiasAdd	conv3/Abs*
T0*'
_output_shapes
: 
]
conv3/ReadVariableOpReadVariableOpconv3/alphas*
_output_shapes
: *
dtype0
c
	conv3/mulMulconv3/ReadVariableOp	conv3/sub*
T0*'
_output_shapes
: 
R
conv3/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
^
conv3/mul_1Mul	conv3/mulconv3/mul_1/y*
T0*'
_output_shapes
: 
]
	conv3/addAddV2
conv3/Reluconv3/mul_1*
T0*'
_output_shapes
: 
[
StopGradient_3StopGradient	conv3/add*
T0*'
_output_shapes
: 
k
)conv3/add/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
­
0conv4_1/weights/Initializer/random_uniform/shapeConst*"
_class
loc:@conv4_1/weights*
_output_shapes
:*
dtype0*%
valueB"             

.conv4_1/weights/Initializer/random_uniform/minConst*"
_class
loc:@conv4_1/weights*
_output_shapes
: *
dtype0*
valueB
 *A×ž

.conv4_1/weights/Initializer/random_uniform/maxConst*"
_class
loc:@conv4_1/weights*
_output_shapes
: *
dtype0*
valueB
 *A×>
Ý
8conv4_1/weights/Initializer/random_uniform/RandomUniformRandomUniform0conv4_1/weights/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv4_1/weights*&
_output_shapes
: *
dtype0
Ú
.conv4_1/weights/Initializer/random_uniform/subSub.conv4_1/weights/Initializer/random_uniform/max.conv4_1/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@conv4_1/weights*
_output_shapes
: 
ô
.conv4_1/weights/Initializer/random_uniform/mulMul8conv4_1/weights/Initializer/random_uniform/RandomUniform.conv4_1/weights/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv4_1/weights*&
_output_shapes
: 
č
*conv4_1/weights/Initializer/random_uniformAddV2.conv4_1/weights/Initializer/random_uniform/mul.conv4_1/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@conv4_1/weights*&
_output_shapes
: 
Ś
conv4_1/weightsVarHandleOp*"
_class
loc:@conv4_1/weights*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv4_1/weights
o
0conv4_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_1/weights*
_output_shapes
: 
t
conv4_1/weights/AssignAssignVariableOpconv4_1/weights*conv4_1/weights/Initializer/random_uniform*
dtype0
{
#conv4_1/weights/Read/ReadVariableOpReadVariableOpconv4_1/weights*&
_output_shapes
: *
dtype0
t
/conv4_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:

?conv4_1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpconv4_1/weights*&
_output_shapes
: *
dtype0

0conv4_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss?conv4_1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
´
)conv4_1/kernel/Regularizer/l2_regularizerMul/conv4_1/kernel/Regularizer/l2_regularizer/scale0conv4_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

 conv4_1/biases/Initializer/zerosConst*!
_class
loc:@conv4_1/biases*
_output_shapes
:*
dtype0*
valueB*    

conv4_1/biasesVarHandleOp*!
_class
loc:@conv4_1/biases*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv4_1/biases
m
/conv4_1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_1/biases*
_output_shapes
: 
h
conv4_1/biases/AssignAssignVariableOpconv4_1/biases conv4_1/biases/Initializer/zeros*
dtype0
m
"conv4_1/biases/Read/ReadVariableOpReadVariableOpconv4_1/biases*
_output_shapes
:*
dtype0
u
conv4_1/Conv2D/ReadVariableOpReadVariableOpconv4_1/weights*&
_output_shapes
: *
dtype0

conv4_1/Conv2DConv2D	conv3/addconv4_1/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:*
paddingVALID*
strides

i
conv4_1/BiasAdd/ReadVariableOpReadVariableOpconv4_1/biases*
_output_shapes
:*
dtype0
|
conv4_1/BiasAddBiasAddconv4_1/Conv2Dconv4_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:
]
conv4_1/SoftmaxSoftmaxconv4_1/BiasAdd*
T0*'
_output_shapes
:
a
StopGradient_4StopGradientconv4_1/Softmax*
T0*'
_output_shapes
:
q
/conv4_1/Softmax/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
­
0conv4_2/weights/Initializer/random_uniform/shapeConst*"
_class
loc:@conv4_2/weights*
_output_shapes
:*
dtype0*%
valueB"             

.conv4_2/weights/Initializer/random_uniform/minConst*"
_class
loc:@conv4_2/weights*
_output_shapes
: *
dtype0*
valueB
 *ěŃž

.conv4_2/weights/Initializer/random_uniform/maxConst*"
_class
loc:@conv4_2/weights*
_output_shapes
: *
dtype0*
valueB
 *ěŃ>
Ý
8conv4_2/weights/Initializer/random_uniform/RandomUniformRandomUniform0conv4_2/weights/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv4_2/weights*&
_output_shapes
: *
dtype0
Ú
.conv4_2/weights/Initializer/random_uniform/subSub.conv4_2/weights/Initializer/random_uniform/max.conv4_2/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@conv4_2/weights*
_output_shapes
: 
ô
.conv4_2/weights/Initializer/random_uniform/mulMul8conv4_2/weights/Initializer/random_uniform/RandomUniform.conv4_2/weights/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv4_2/weights*&
_output_shapes
: 
č
*conv4_2/weights/Initializer/random_uniformAddV2.conv4_2/weights/Initializer/random_uniform/mul.conv4_2/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@conv4_2/weights*&
_output_shapes
: 
Ś
conv4_2/weightsVarHandleOp*"
_class
loc:@conv4_2/weights*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv4_2/weights
o
0conv4_2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_2/weights*
_output_shapes
: 
t
conv4_2/weights/AssignAssignVariableOpconv4_2/weights*conv4_2/weights/Initializer/random_uniform*
dtype0
{
#conv4_2/weights/Read/ReadVariableOpReadVariableOpconv4_2/weights*&
_output_shapes
: *
dtype0
t
/conv4_2/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:

?conv4_2/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpconv4_2/weights*&
_output_shapes
: *
dtype0

0conv4_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss?conv4_2/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
´
)conv4_2/kernel/Regularizer/l2_regularizerMul/conv4_2/kernel/Regularizer/l2_regularizer/scale0conv4_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

 conv4_2/biases/Initializer/zerosConst*!
_class
loc:@conv4_2/biases*
_output_shapes
:*
dtype0*
valueB*    

conv4_2/biasesVarHandleOp*!
_class
loc:@conv4_2/biases*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv4_2/biases
m
/conv4_2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_2/biases*
_output_shapes
: 
h
conv4_2/biases/AssignAssignVariableOpconv4_2/biases conv4_2/biases/Initializer/zeros*
dtype0
m
"conv4_2/biases/Read/ReadVariableOpReadVariableOpconv4_2/biases*
_output_shapes
:*
dtype0
u
conv4_2/Conv2D/ReadVariableOpReadVariableOpconv4_2/weights*&
_output_shapes
: *
dtype0

conv4_2/Conv2DConv2D	conv3/addconv4_2/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:*
paddingVALID*
strides

i
conv4_2/BiasAdd/ReadVariableOpReadVariableOpconv4_2/biases*
_output_shapes
:*
dtype0
|
conv4_2/BiasAddBiasAddconv4_2/Conv2Dconv4_2/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:
a
StopGradient_5StopGradientconv4_2/BiasAdd*
T0*'
_output_shapes
:
q
/conv4_2/BiasAdd/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
­
0conv4_3/weights/Initializer/random_uniform/shapeConst*"
_class
loc:@conv4_3/weights*
_output_shapes
:*
dtype0*%
valueB"          
   

.conv4_3/weights/Initializer/random_uniform/minConst*"
_class
loc:@conv4_3/weights*
_output_shapes
: *
dtype0*
valueB
 *Áž

.conv4_3/weights/Initializer/random_uniform/maxConst*"
_class
loc:@conv4_3/weights*
_output_shapes
: *
dtype0*
valueB
 *Á>
Ý
8conv4_3/weights/Initializer/random_uniform/RandomUniformRandomUniform0conv4_3/weights/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 
*
dtype0
Ú
.conv4_3/weights/Initializer/random_uniform/subSub.conv4_3/weights/Initializer/random_uniform/max.conv4_3/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@conv4_3/weights*
_output_shapes
: 
ô
.conv4_3/weights/Initializer/random_uniform/mulMul8conv4_3/weights/Initializer/random_uniform/RandomUniform.conv4_3/weights/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 

č
*conv4_3/weights/Initializer/random_uniformAddV2.conv4_3/weights/Initializer/random_uniform/mul.conv4_3/weights/Initializer/random_uniform/min*
T0*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 

Ś
conv4_3/weightsVarHandleOp*"
_class
loc:@conv4_3/weights*
_output_shapes
: *
dtype0*
shape: 
* 
shared_nameconv4_3/weights
o
0conv4_3/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_3/weights*
_output_shapes
: 
t
conv4_3/weights/AssignAssignVariableOpconv4_3/weights*conv4_3/weights/Initializer/random_uniform*
dtype0
{
#conv4_3/weights/Read/ReadVariableOpReadVariableOpconv4_3/weights*&
_output_shapes
: 
*
dtype0
t
/conv4_3/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:

?conv4_3/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpReadVariableOpconv4_3/weights*&
_output_shapes
: 
*
dtype0

0conv4_3/kernel/Regularizer/l2_regularizer/L2LossL2Loss?conv4_3/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOp*
T0*
_output_shapes
: 
´
)conv4_3/kernel/Regularizer/l2_regularizerMul/conv4_3/kernel/Regularizer/l2_regularizer/scale0conv4_3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 

 conv4_3/biases/Initializer/zerosConst*!
_class
loc:@conv4_3/biases*
_output_shapes
:
*
dtype0*
valueB
*    

conv4_3/biasesVarHandleOp*!
_class
loc:@conv4_3/biases*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv4_3/biases
m
/conv4_3/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_3/biases*
_output_shapes
: 
h
conv4_3/biases/AssignAssignVariableOpconv4_3/biases conv4_3/biases/Initializer/zeros*
dtype0
m
"conv4_3/biases/Read/ReadVariableOpReadVariableOpconv4_3/biases*
_output_shapes
:
*
dtype0
u
conv4_3/Conv2D/ReadVariableOpReadVariableOpconv4_3/weights*&
_output_shapes
: 
*
dtype0

conv4_3/Conv2DConv2D	conv3/addconv4_3/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:
*
paddingVALID*
strides

i
conv4_3/BiasAdd/ReadVariableOpReadVariableOpconv4_3/biases*
_output_shapes
:
*
dtype0
|
conv4_3/BiasAddBiasAddconv4_3/Conv2Dconv4_3/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:

a
StopGradient_6StopGradientconv4_3/BiasAdd*
T0*'
_output_shapes
:

q
/conv4_3/BiasAdd/activations/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
f
cls_probSqueezeconv4_1/Softmax*
T0*
_output_shapes
:	*
squeeze_dims

Y

zeros_likeConst*
_output_shapes	
:*
dtype0*
valueB*    
K
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
A
LessLesslabelLess/y*
T0*
_output_shapes	
:
S
SelectV2SelectV2Less
zeros_likelabel*
T0*
_output_shapes	
:
G
SizeConst*
_output_shapes
: *
dtype0*
value
B :
\
Reshape_4/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
^
Reshape_4/shapePackSizeReshape_4/shape/1*
N*
T0*
_output_shapes
:
Y
	Reshape_4Reshapecls_probReshape_4/shape*
T0*
_output_shapes
:	
M
Cast_2CastSelectV2*

DstT0*

SrcT0*
_output_shapes	
:
L
	ToInt32/xConst*
_output_shapes
: *
dtype0*
value
B :
M
range/startConst*
_output_shapes
: *
dtype0*
value	B : 
M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
P
rangeRangerange/start	ToInt32/xrange/delta*
_output_shapes	
:
G
mul/yConst*
_output_shapes
: *
dtype0*
value	B :
>
mulMulrangemul/y*
T0*
_output_shapes	
:
?
addAddV2mulCast_2*
T0*
_output_shapes	
:
O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 

GatherV2GatherV2	Reshape_4addGatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:	
B
SqueezeSqueezeGatherV2*
T0*
_output_shapes	
:
L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *˙ćŰ.
F
add_1AddV2Squeezeadd_1/y*
T0*
_output_shapes	
:
7
LogLogadd_1*
T0*
_output_shapes	
:
5
NegNegLog*
T0*
_output_shapes	
:
[
zeros_like_1Const*
_output_shapes	
:*
dtype0*
valueB*    
j
ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
i
	ones_likeFillones_like/Shape/shape_as_tensorones_like/Const*
T0*
_output_shapes	
:
I
Less_1Lesslabelzeros_like_1*
T0*
_output_shapes	
:
]

SelectV2_1SelectV2Less_1zeros_like_1	ones_like*
T0*
_output_shapes	
:
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
>
SumSum
SelectV2_1Const*
T0*
_output_shapes
: 
L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *333?
;
mul_1MulSummul_1/y*
T0*
_output_shapes
: 
E
Cast_3Castmul_1*

DstT0*

SrcT0*
_output_shapes
: 
C
mul_2MulNeg
SelectV2_1*
T0*
_output_shapes	
:
\
TopKV2TopKV2mul_2Cast_3*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
>
MeanMeanTopKV2Const_1*
T0*
_output_shapes
: 
g
	bbox_predSqueezeconv4_2/BiasAdd*
T0*
_output_shapes
:	*
squeeze_dims

[
zeros_like_2Const*
_output_shapes	
:*
dtype0*
valueB*    
l
!ones_like_1/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
o
ones_like_1Fill!ones_like_1/Shape/shape_as_tensorones_like_1/Const*
T0*
_output_shapes	
:
7
AbsAbslabel*
T0*
_output_shapes	
:
L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
EqualEqualAbsEqual/y*
T0*
_output_shapes	
:
^

SelectV2_2SelectV2Equalones_like_1zeros_like_2*
T0*
_output_shapes	
:
N
sub_1Sub	bbox_predbbox_target*
T0*
_output_shapes
:	
A
SquareSquaresub_1*
T0*
_output_shapes
:	
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
S
Sum_1SumSquareSum_1/reduction_indices*
T0*
_output_shapes	
:
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
B
Sum_2Sum
SelectV2_2Const_2*
T0*
_output_shapes
: 
E
Cast_4CastSum_2*

DstT0*

SrcT0*
_output_shapes
: 
E
mul_3MulSum_1
SelectV2_2*
T0*
_output_shapes	
:
^
TopKV2_1TopKV2mul_3Cast_4*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 


GatherV2_1GatherV2mul_3
TopKV2_1:1GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
D
Mean_1Mean
GatherV2_1Const_3*
T0*
_output_shapes
: 
k
landmark_predSqueezeconv4_3/BiasAdd*
T0*
_output_shapes
:	
*
squeeze_dims

l
!ones_like_2/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
V
ones_like_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
o
ones_like_2Fill!ones_like_2/Shape/shape_as_tensorones_like_2/Const*
T0*
_output_shapes	
:
[
zeros_like_3Const*
_output_shapes	
:*
dtype0*
valueB*    
N
	Equal_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ŕ
H
Equal_1Equallabel	Equal_1/y*
T0*
_output_shapes	
:
`

SelectV2_3SelectV2Equal_1ones_like_2zeros_like_3*
T0*
_output_shapes	
:
V
sub_2Sublandmark_predlandmark_target*
T0*
_output_shapes
:	

C
Square_1Squaresub_2*
T0*
_output_shapes
:	

Y
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
U
Sum_3SumSquare_1Sum_3/reduction_indices*
T0*
_output_shapes	
:
Q
Const_4Const*
_output_shapes
:*
dtype0*
valueB: 
B
Sum_4Sum
SelectV2_3Const_4*
T0*
_output_shapes
: 
E
Cast_5CastSum_4*

DstT0*

SrcT0*
_output_shapes
: 
E
mul_4MulSum_3
SelectV2_3*
T0*
_output_shapes	
:
^
TopKV2_2TopKV2mul_4Cast_5*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 


GatherV2_2GatherV2mul_4
TopKV2_2:1GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
Const_5Const*
_output_shapes
:*
dtype0*
valueB: 
D
Mean_2Mean
GatherV2_2Const_5*
T0*
_output_shapes
: 
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
R
ArgMaxArgMaxcls_probArgMax/dimension*
T0*
_output_shapes	
:
J
Cast_6Castlabel*

DstT0	*

SrcT0*
_output_shapes	
:
P
GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
Z
GreaterEqualGreaterEqualCast_6GreaterEqual/y*
T0	*
_output_shapes	
:
E
WhereWhereGreaterEqual*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>
	Squeeze_1SqueezeWhere*
T0	*
_output_shapes
:
Q
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 


GatherV2_3GatherV2Cast_6	Squeeze_1GatherV2_3/axis*
Taxis0*
Tindices0	*
Tparams0	*
_output_shapes
:
Q
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 


GatherV2_4GatherV2ArgMax	Squeeze_1GatherV2_4/axis*
Taxis0*
Tindices0	*
Tparams0	*
_output_shapes
:
K
Equal_2Equal
GatherV2_3
GatherV2_4*
T0	*
_output_shapes
:
I
Cast_7CastEqual_2*

DstT0*

SrcT0
*
_output_shapes
:
5
RankRankCast_7*
T0*
_output_shapes
: 
O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
Y
range_1Rangerange_1/startRankrange_1/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
@
Mean_3MeanCast_7range_1*
T0*
_output_shapes
: 
˛
AddNAddN'conv1/kernel/Regularizer/l2_regularizer'conv2/kernel/Regularizer/l2_regularizer'conv3/kernel/Regularizer/l2_regularizer)conv4_1/kernel/Regularizer/l2_regularizer)conv4_2/kernel/Regularizer/l2_regularizer)conv4_3/kernel/Regularizer/l2_regularizer*
N*
T0*
_output_shapes
: 
L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
<
mul_5Mulmul_5/xMean*
T0*
_output_shapes
: 
L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
>
mul_6Mulmul_6/xMean_1*
T0*
_output_shapes
: 
=
add_2AddV2mul_5mul_6*
T0*
_output_shapes
: 
L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
>
mul_7Mulmul_7/xMean_2*
T0*
_output_shapes
: 
=
add_3AddV2add_2mul_7*
T0*
_output_shapes
: 
<
add_4AddV2add_3AddN*
T0*
_output_shapes
: 

"Variable/Initializer/initial_valueConst*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
value	B : 

VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
^
Variable/AssignAssignVariableOpVariable"Variable/Initializer/initial_value*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
K
Const_6Const*
_output_shapes
: *
dtype0*
valueB	 :Ú
K
Const_7Const*
_output_shapes
: *
dtype0*
valueB	 :üĚ
K
Const_8Const*
_output_shapes
: *
dtype0*
valueB	 :ÖŰ
L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *
×#<
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *o:
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *ˇŃ8
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'7
O
ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
a
 PiecewiseConstant/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
t
PiecewiseConstant/LessEqual	LessEqual PiecewiseConstant/ReadVariableOpConst_6*
T0*
_output_shapes
: 
p
PiecewiseConstant/GreaterGreater PiecewiseConstant/ReadVariableOpConst_8*
T0*
_output_shapes
: 
r
PiecewiseConstant/Greater_1Greater PiecewiseConstant/ReadVariableOpConst_6*
T0*
_output_shapes
: 
v
PiecewiseConstant/LessEqual_1	LessEqual PiecewiseConstant/ReadVariableOpConst_7*
T0*
_output_shapes
: 
w
PiecewiseConstant/and
LogicalAndPiecewiseConstant/Greater_1PiecewiseConstant/LessEqual_1*
_output_shapes
: 
r
PiecewiseConstant/Greater_2Greater PiecewiseConstant/ReadVariableOpConst_7*
T0*
_output_shapes
: 
v
PiecewiseConstant/LessEqual_2	LessEqual PiecewiseConstant/ReadVariableOpConst_8*
T0*
_output_shapes
: 
y
PiecewiseConstant/and_1
LogicalAndPiecewiseConstant/Greater_2PiecewiseConstant/LessEqual_2*
_output_shapes
: 
ź
PiecewiseConstant/case/preds_cPackPiecewiseConstant/LessEqualPiecewiseConstant/GreaterPiecewiseConstant/andPiecewiseConstant/and_1*
N*
T0
*
_output_shapes
:
w
PiecewiseConstant/case/CastCastPiecewiseConstant/case/preds_c*

DstT0*

SrcT0
*
_output_shapes
:
f
PiecewiseConstant/case/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

%PiecewiseConstant/case/num_true_condsSumPiecewiseConstant/case/CastPiecewiseConstant/case/Const*
T0*
_output_shapes
: 
e
#PiecewiseConstant/case/n_true_condsConst*
_output_shapes
: *
dtype0*
value	B :

 PiecewiseConstant/case/LessEqual	LessEqual%PiecewiseConstant/case/num_true_conds#PiecewiseConstant/case/n_true_conds*
T0*
_output_shapes
: 

#PiecewiseConstant/case/Assert/ConstConst*
_output_shapes
: *
dtype0*Ë
valueÁBž BˇInput error: exclusive=True: more than 1 conditions (PiecewiseConstant/LessEqual:0, PiecewiseConstant/Greater:0, PiecewiseConstant/and:0, PiecewiseConstant/and_1:0) evaluated as True:
ż
)PiecewiseConstant/case/Assert/AssertGuardIf PiecewiseConstant/case/LessEqual PiecewiseConstant/case/LessEqualPiecewiseConstant/case/preds_c*
Tcond0
*
Tin
2

*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *F
else_branch7R5
3PiecewiseConstant_case_Assert_AssertGuard_false_389*
output_shapes
: *E
then_branch6R4
2PiecewiseConstant_case_Assert_AssertGuard_true_388

2PiecewiseConstant/case/Assert/AssertGuard/IdentityIdentity)PiecewiseConstant/case/Assert/AssertGuard*
T0
*
_output_shapes
: 

PiecewiseConstant/case/condStatelessIfPiecewiseConstant/LessEqualConst_9PiecewiseConstant/GreaterConst_12PiecewiseConstant/andConst_10PiecewiseConstant/and_1Const_113^PiecewiseConstant/case/Assert/AssertGuard/Identity*
Tcond0
*
Tin
	2


*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *8
else_branch)R'
%PiecewiseConstant_case_cond_false_402*
output_shapes
: *7
then_branch(R&
$PiecewiseConstant_case_cond_true_401
n
$PiecewiseConstant/case/cond/IdentityIdentityPiecewiseConstant/case/cond*
T0*
_output_shapes
: 
k
&ExponentialDecay/initial_learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *o:
Z
ExponentialDecay/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :N
f
ExponentialDecay/CastCastExponentialDecay/Cast/x*

DstT0*

SrcT0*
_output_shapes
: 
^
ExponentialDecay/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
g
&ExponentialDecay/Cast_2/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
w
ExponentialDecay/Cast_2Cast&ExponentialDecay/Cast_2/ReadVariableOp*

DstT0*

SrcT0*
_output_shapes
: 
t
ExponentialDecay/truedivRealDivExponentialDecay/Cast_2ExponentialDecay/Cast*
T0*
_output_shapes
: 
Z
ExponentialDecay/FloorFloorExponentialDecay/truediv*
T0*
_output_shapes
: 
o
ExponentialDecay/PowPowExponentialDecay/Cast_1/xExponentialDecay/Floor*
T0*
_output_shapes
: 
v
ExponentialDecayMul&ExponentialDecay/initial_learning_rateExponentialDecay/Pow*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
^
gradients/grad_ys_0/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
h
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
T0*
_output_shapes
: 
C
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/grad_ys_0
ż
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/grad_ys_0&^gradients/add_4_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Á
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/grad_ys_0&^gradients/add_4_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
]
%gradients/add_3_grad/tuple/group_depsNoOp.^gradients/add_4_grad/tuple/control_dependency
Ů
-gradients/add_3_grad/tuple/control_dependencyIdentity-gradients/add_4_grad/tuple/control_dependency&^gradients/add_3_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
/gradients/add_3_grad/tuple/control_dependency_1Identity-gradients/add_4_grad/tuple/control_dependency&^gradients/add_3_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
^
$gradients/AddN_grad/tuple/group_depsNoOp0^gradients/add_4_grad/tuple/control_dependency_1
Ů
,gradients/AddN_grad/tuple/control_dependencyIdentity/gradients/add_4_grad/tuple/control_dependency_1%^gradients/AddN_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
.gradients/AddN_grad/tuple/control_dependency_1Identity/gradients/add_4_grad/tuple/control_dependency_1%^gradients/AddN_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
.gradients/AddN_grad/tuple/control_dependency_2Identity/gradients/add_4_grad/tuple/control_dependency_1%^gradients/AddN_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
.gradients/AddN_grad/tuple/control_dependency_3Identity/gradients/add_4_grad/tuple/control_dependency_1%^gradients/AddN_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
.gradients/AddN_grad/tuple/control_dependency_4Identity/gradients/add_4_grad/tuple/control_dependency_1%^gradients/AddN_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
.gradients/AddN_grad/tuple/control_dependency_5Identity/gradients/add_4_grad/tuple/control_dependency_1%^gradients/AddN_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
]
%gradients/add_2_grad/tuple/group_depsNoOp.^gradients/add_3_grad/tuple/control_dependency
Ů
-gradients/add_2_grad/tuple/control_dependencyIdentity-gradients/add_3_grad/tuple/control_dependency&^gradients/add_2_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
Ű
/gradients/add_2_grad/tuple/control_dependency_1Identity-gradients/add_3_grad/tuple/control_dependency&^gradients/add_2_grad/tuple/group_deps*
T0*&
_class
loc:@gradients/grad_ys_0*
_output_shapes
: 
y
gradients/mul_7_grad/MulMul/gradients/add_3_grad/tuple/control_dependency_1Mean_2*
T0*
_output_shapes
: 
|
gradients/mul_7_grad/Mul_1Mul/gradients/add_3_grad/tuple/control_dependency_1mul_7/x*
T0*
_output_shapes
: 
e
%gradients/mul_7_grad/tuple/group_depsNoOp^gradients/mul_7_grad/Mul^gradients/mul_7_grad/Mul_1
É
-gradients/mul_7_grad/tuple/control_dependencyIdentitygradients/mul_7_grad/Mul&^gradients/mul_7_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_7_grad/Mul*
_output_shapes
: 
Ď
/gradients/mul_7_grad/tuple/control_dependency_1Identitygradients/mul_7_grad/Mul_1&^gradients/mul_7_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_7_grad/Mul_1*
_output_shapes
: 
Ŕ
:gradients/conv1/kernel/Regularizer/l2_regularizer_grad/MulMul,gradients/AddN_grad/tuple/control_dependency.conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Á
<gradients/conv1/kernel/Regularizer/l2_regularizer_grad/Mul_1Mul,gradients/AddN_grad/tuple/control_dependency-conv1/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
Ë
Ggradients/conv1/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp;^gradients/conv1/kernel/Regularizer/l2_regularizer_grad/Mul=^gradients/conv1/kernel/Regularizer/l2_regularizer_grad/Mul_1
Ń
Ogradients/conv1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity:gradients/conv1/kernel/Regularizer/l2_regularizer_grad/MulH^gradients/conv1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv1/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
×
Qgradients/conv1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity<gradients/conv1/kernel/Regularizer/l2_regularizer_grad/Mul_1H^gradients/conv1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/conv1/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
Â
:gradients/conv2/kernel/Regularizer/l2_regularizer_grad/MulMul.gradients/AddN_grad/tuple/control_dependency_1.conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Ă
<gradients/conv2/kernel/Regularizer/l2_regularizer_grad/Mul_1Mul.gradients/AddN_grad/tuple/control_dependency_1-conv2/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
Ë
Ggradients/conv2/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp;^gradients/conv2/kernel/Regularizer/l2_regularizer_grad/Mul=^gradients/conv2/kernel/Regularizer/l2_regularizer_grad/Mul_1
Ń
Ogradients/conv2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity:gradients/conv2/kernel/Regularizer/l2_regularizer_grad/MulH^gradients/conv2/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv2/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
×
Qgradients/conv2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity<gradients/conv2/kernel/Regularizer/l2_regularizer_grad/Mul_1H^gradients/conv2/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/conv2/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
Â
:gradients/conv3/kernel/Regularizer/l2_regularizer_grad/MulMul.gradients/AddN_grad/tuple/control_dependency_2.conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Ă
<gradients/conv3/kernel/Regularizer/l2_regularizer_grad/Mul_1Mul.gradients/AddN_grad/tuple/control_dependency_2-conv3/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
Ë
Ggradients/conv3/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp;^gradients/conv3/kernel/Regularizer/l2_regularizer_grad/Mul=^gradients/conv3/kernel/Regularizer/l2_regularizer_grad/Mul_1
Ń
Ogradients/conv3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity:gradients/conv3/kernel/Regularizer/l2_regularizer_grad/MulH^gradients/conv3/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/conv3/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
×
Qgradients/conv3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity<gradients/conv3/kernel/Regularizer/l2_regularizer_grad/Mul_1H^gradients/conv3/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/conv3/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
Ć
<gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/MulMul.gradients/AddN_grad/tuple/control_dependency_30conv4_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Ç
>gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/Mul_1Mul.gradients/AddN_grad/tuple/control_dependency_3/conv4_1/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
Ń
Igradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp=^gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/Mul?^gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/Mul_1
Ů
Qgradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity<gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/MulJ^gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
ß
Sgradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity>gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/Mul_1J^gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
Ć
<gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/MulMul.gradients/AddN_grad/tuple/control_dependency_40conv4_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Ç
>gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/Mul_1Mul.gradients/AddN_grad/tuple/control_dependency_4/conv4_2/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
Ń
Igradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp=^gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/Mul?^gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/Mul_1
Ů
Qgradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity<gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/MulJ^gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
ß
Sgradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity>gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/Mul_1J^gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
Ć
<gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/MulMul.gradients/AddN_grad/tuple/control_dependency_50conv4_3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
Ç
>gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/Mul_1Mul.gradients/AddN_grad/tuple/control_dependency_5/conv4_3/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
Ń
Igradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOp=^gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/Mul?^gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/Mul_1
Ů
Qgradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentity<gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/MulJ^gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
ß
Sgradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1Identity>gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/Mul_1J^gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
u
gradients/mul_5_grad/MulMul-gradients/add_2_grad/tuple/control_dependencyMean*
T0*
_output_shapes
: 
z
gradients/mul_5_grad/Mul_1Mul-gradients/add_2_grad/tuple/control_dependencymul_5/x*
T0*
_output_shapes
: 
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
É
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_5_grad/Mul*
_output_shapes
: 
Ď
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
_output_shapes
: 
y
gradients/mul_6_grad/MulMul/gradients/add_2_grad/tuple/control_dependency_1Mean_1*
T0*
_output_shapes
: 
|
gradients/mul_6_grad/Mul_1Mul/gradients/add_2_grad/tuple/control_dependency_1mul_6/x*
T0*
_output_shapes
: 
e
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Mul^gradients/mul_6_grad/Mul_1
É
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Mul&^gradients/mul_6_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_6_grad/Mul*
_output_shapes
: 
Ď
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Mul_1&^gradients/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_6_grad/Mul_1*
_output_shapes
: 
m
#gradients/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ł
gradients/Mean_2_grad/ReshapeReshape/gradients/mul_7_grad/tuple/control_dependency_1#gradients/Mean_2_grad/Reshape/shape*
T0*
_output_shapes
:
U
gradients/Mean_2_grad/ShapeShape
GatherV2_2*
T0*
_output_shapes
:

gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
gradients/Mean_2_grad/Shape_1Shape
GatherV2_2*
T0*
_output_shapes
:
`
gradients/Mean_2_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
e
gradients/Mean_2_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
T0*
_output_shapes
: 
g
gradients/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
T0*
_output_shapes
: 
a
gradients/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
r
gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Agradients/conv1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMul=conv1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpQgradients/conv1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:


Agradients/conv2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMul=conv2/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpQgradients/conv2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:


Agradients/conv3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMul=conv3/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpQgradients/conv3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: 

Cgradients/conv4_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMul?conv4_1/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpSgradients/conv4_1/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: 

Cgradients/conv4_2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMul?conv4_2/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpSgradients/conv4_2/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: 

Cgradients/conv4_3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMul?conv4_3/kernel/Regularizer/l2_regularizer/L2Loss/ReadVariableOpSgradients/conv4_3/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: 

k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:

gradients/Mean_grad/ReshapeReshape/gradients/mul_5_grad/tuple/control_dependency_1!gradients/Mean_grad/Reshape/shape*
T0*
_output_shapes
:
O
gradients/Mean_grad/ShapeShapeTopKV2*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Q
gradients/Mean_grad/Shape_1ShapeTopKV2*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
y
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
}
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
#gradients/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ł
gradients/Mean_1_grad/ReshapeReshape/gradients/mul_6_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:
U
gradients/Mean_1_grad/ShapeShape
GatherV2_1*
T0*
_output_shapes
:

gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
W
gradients/Mean_1_grad/Shape_1Shape
GatherV2_1*
T0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
e
gradients/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
T0*
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
T0*
_output_shapes
: 
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
r
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 

gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/GatherV2_2_grad/ShapeConst*
_class

loc:@mul_4*
_output_shapes
:*
dtype0	*
valueB	R

gradients/GatherV2_2_grad/CastCastgradients/GatherV2_2_grad/Shape*

DstT0*

SrcT0	*
_class

loc:@mul_4*
_output_shapes
:
S
gradients/GatherV2_2_grad/SizeSize
TopKV2_2:1*
T0*
_output_shapes
: 
j
(gradients/GatherV2_2_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Ą
$gradients/GatherV2_2_grad/ExpandDims
ExpandDimsgradients/GatherV2_2_grad/Size(gradients/GatherV2_2_grad/ExpandDims/dim*
T0*
_output_shapes
:
w
-gradients/GatherV2_2_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
y
/gradients/GatherV2_2_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
y
/gradients/GatherV2_2_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
¨
'gradients/GatherV2_2_grad/strided_sliceStridedSlicegradients/GatherV2_2_grad/Cast-gradients/GatherV2_2_grad/strided_slice/stack/gradients/GatherV2_2_grad/strided_slice/stack_1/gradients/GatherV2_2_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
end_mask
g
%gradients/GatherV2_2_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Đ
 gradients/GatherV2_2_grad/concatConcatV2$gradients/GatherV2_2_grad/ExpandDims'gradients/GatherV2_2_grad/strided_slice%gradients/GatherV2_2_grad/concat/axis*
N*
T0*
_output_shapes
:

!gradients/GatherV2_2_grad/ReshapeReshapegradients/Mean_2_grad/truediv gradients/GatherV2_2_grad/concat*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/GatherV2_2_grad/Reshape_1Reshape
TopKV2_2:1$gradients/GatherV2_2_grad/ExpandDims*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
gradients/TopKV2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
U
gradients/TopKV2_grad/Shape_1ShapeTopKV2:1*
T0*
_output_shapes
:
u
gradients/TopKV2_grad/CastCastgradients/TopKV2_grad/Shape_1*

DstT0	*

SrcT0*
_output_shapes
:
\
gradients/TopKV2_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
]
gradients/TopKV2_grad/sub/yConst*
_output_shapes
: *
dtype0*
value	B :
z
gradients/TopKV2_grad/subSubgradients/TopKV2_grad/Sizegradients/TopKV2_grad/sub/y*
T0*
_output_shapes
: 
e
#gradients/TopKV2_grad/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ę
gradients/TopKV2_grad/GatherV2GatherV2gradients/TopKV2_grad/Castgradients/TopKV2_grad/sub#gradients/TopKV2_grad/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
h
gradients/TopKV2_grad/stack/0Const*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙

gradients/TopKV2_grad/stackPackgradients/TopKV2_grad/stack/0gradients/TopKV2_grad/GatherV2*
N*
T0	*
_output_shapes
:

gradients/TopKV2_grad/ReshapeReshapeTopKV2:1gradients/TopKV2_grad/stack*
T0*
Tshape0	*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
u
gradients/TopKV2_grad/Cast_1Castgradients/TopKV2_grad/Shape*

DstT0	*

SrcT0*
_output_shapes
:
^
gradients/TopKV2_grad/Size_1Const*
_output_shapes
: *
dtype0*
value	B :
_
gradients/TopKV2_grad/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/TopKV2_grad/sub_1Subgradients/TopKV2_grad/Size_1gradients/TopKV2_grad/sub_1/y*
T0*
_output_shapes
: 
g
%gradients/TopKV2_grad/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Ň
 gradients/TopKV2_grad/GatherV2_1GatherV2gradients/TopKV2_grad/Cast_1gradients/TopKV2_grad/sub_1%gradients/TopKV2_grad/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
j
gradients/TopKV2_grad/Shape_2Shapegradients/TopKV2_grad/Reshape*
T0*
_output_shapes
:
s
)gradients/TopKV2_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
u
+gradients/TopKV2_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
u
+gradients/TopKV2_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

#gradients/TopKV2_grad/strided_sliceStridedSlicegradients/TopKV2_grad/Shape_2)gradients/TopKV2_grad/strided_slice/stack+gradients/TopKV2_grad/strided_slice/stack_1+gradients/TopKV2_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
y
gradients/TopKV2_grad/Cast_2Cast#gradients/TopKV2_grad/strided_slice*

DstT0	*

SrcT0*
_output_shapes
: 

gradients/TopKV2_grad/mulMulgradients/TopKV2_grad/Cast_2 gradients/TopKV2_grad/GatherV2_1*
T0	*
_output_shapes
: 
c
!gradients/TopKV2_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
{
 gradients/TopKV2_grad/range/CastCast!gradients/TopKV2_grad/range/start*

DstT0	*

SrcT0*
_output_shapes
: 
´
gradients/TopKV2_grad/rangeRange gradients/TopKV2_grad/range/Castgradients/TopKV2_grad/mul gradients/TopKV2_grad/GatherV2_1*

Tidx0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
$gradients/TopKV2_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ł
 gradients/TopKV2_grad/ExpandDims
ExpandDimsgradients/TopKV2_grad/range$gradients/TopKV2_grad/ExpandDims/dim*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/TopKV2_grad/Cast_3Cast gradients/TopKV2_grad/ExpandDims*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/TopKV2_grad/addAddV2gradients/TopKV2_grad/Reshapegradients/TopKV2_grad/Cast_3*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
x
%gradients/TopKV2_grad/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

gradients/TopKV2_grad/Reshape_1Reshapegradients/TopKV2_grad/add%gradients/TopKV2_grad/Reshape_1/shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
&gradients/TopKV2_grad/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ť
"gradients/TopKV2_grad/ExpandDims_1
ExpandDimsgradients/TopKV2_grad/Reshape_1&gradients/TopKV2_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
%gradients/TopKV2_grad/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

gradients/TopKV2_grad/Reshape_2Reshapegradients/Mean_grad/truediv%gradients/TopKV2_grad/Reshape_2/shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
gradients/TopKV2_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
}
gradients/TopKV2_grad/ProdProdgradients/TopKV2_grad/Shapegradients/TopKV2_grad/Const*
T0*
_output_shapes
: 
w
%gradients/TopKV2_grad/ScatterNd/shapePackgradients/TopKV2_grad/Prod*
N*
T0*
_output_shapes
:
Î
gradients/TopKV2_grad/ScatterNd	ScatterNd"gradients/TopKV2_grad/ExpandDims_1gradients/TopKV2_grad/Reshape_2%gradients/TopKV2_grad/ScatterNd/shape*
T0*
Tindices0*
_output_shapes	
:

gradients/TopKV2_grad/Reshape_3Reshapegradients/TopKV2_grad/ScatterNdgradients/TopKV2_grad/Shape*
T0*
_output_shapes	
:
]
gradients/TopKV2_grad/zerosConst*
_output_shapes
: *
dtype0*
value	B : 
n
&gradients/TopKV2_grad/tuple/group_depsNoOp ^gradients/TopKV2_grad/Reshape_3^gradients/TopKV2_grad/zeros
Ţ
.gradients/TopKV2_grad/tuple/control_dependencyIdentitygradients/TopKV2_grad/Reshape_3'^gradients/TopKV2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/TopKV2_grad/Reshape_3*
_output_shapes	
:
Ó
0gradients/TopKV2_grad/tuple/control_dependency_1Identitygradients/TopKV2_grad/zeros'^gradients/TopKV2_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/TopKV2_grad/zeros*
_output_shapes
: 

gradients/GatherV2_1_grad/ShapeConst*
_class

loc:@mul_3*
_output_shapes
:*
dtype0	*
valueB	R

gradients/GatherV2_1_grad/CastCastgradients/GatherV2_1_grad/Shape*

DstT0*

SrcT0	*
_class

loc:@mul_3*
_output_shapes
:
S
gradients/GatherV2_1_grad/SizeSize
TopKV2_1:1*
T0*
_output_shapes
: 
j
(gradients/GatherV2_1_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Ą
$gradients/GatherV2_1_grad/ExpandDims
ExpandDimsgradients/GatherV2_1_grad/Size(gradients/GatherV2_1_grad/ExpandDims/dim*
T0*
_output_shapes
:
w
-gradients/GatherV2_1_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
y
/gradients/GatherV2_1_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
y
/gradients/GatherV2_1_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
¨
'gradients/GatherV2_1_grad/strided_sliceStridedSlicegradients/GatherV2_1_grad/Cast-gradients/GatherV2_1_grad/strided_slice/stack/gradients/GatherV2_1_grad/strided_slice/stack_1/gradients/GatherV2_1_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
end_mask
g
%gradients/GatherV2_1_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Đ
 gradients/GatherV2_1_grad/concatConcatV2$gradients/GatherV2_1_grad/ExpandDims'gradients/GatherV2_1_grad/strided_slice%gradients/GatherV2_1_grad/concat/axis*
N*
T0*
_output_shapes
:

!gradients/GatherV2_1_grad/ReshapeReshapegradients/Mean_1_grad/truediv gradients/GatherV2_1_grad/concat*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

#gradients/GatherV2_1_grad/Reshape_1Reshape
TopKV2_1:1$gradients/GatherV2_1_grad/ExpandDims*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/mul_2_grad/MulMul.gradients/TopKV2_grad/tuple/control_dependency
SelectV2_1*
T0*
_output_shapes	
:
|
gradients/mul_2_grad/Mul_1Mul.gradients/TopKV2_grad/tuple/control_dependencyNeg*
T0*
_output_shapes	
:
e
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Mul^gradients/mul_2_grad/Mul_1
Î
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Mul&^gradients/mul_2_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_2_grad/Mul*
_output_shapes	
:
Ô
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Mul_1&^gradients/mul_2_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_2_grad/Mul_1*
_output_shapes	
:
e
gradients/mul_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
g
gradients/mul_4_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
ą
*gradients/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_4_grad/Shapegradients/mul_4_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
v
,gradients/mul_4_grad/Mul/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
x
.gradients/mul_4_grad/Mul/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
x
.gradients/mul_4_grad/Mul/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ş
&gradients/mul_4_grad/Mul/strided_sliceStridedSlicegradients/GatherV2_2_grad/Cast,gradients/mul_4_grad/Mul/strided_slice/stack.gradients/mul_4_grad/Mul/strided_slice/stack_1.gradients/mul_4_grad/Mul/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ö
gradients/mul_4_grad/Mul/xUnsortedSegmentSum!gradients/GatherV2_2_grad/Reshape#gradients/GatherV2_2_grad/Reshape_1&gradients/mul_4_grad/Mul/strided_slice*
T0*
Tindices0*
_output_shapes	
:
m
gradients/mul_4_grad/MulMulgradients/mul_4_grad/Mul/x
SelectV2_3*
T0*
_output_shapes	
:

gradients/mul_4_grad/SumSumgradients/mul_4_grad/Mul*gradients/mul_4_grad/BroadcastGradientArgs*
T0*
_output_shapes	
:

gradients/mul_4_grad/ReshapeReshapegradients/mul_4_grad/Sumgradients/mul_4_grad/Shape*
T0*
_output_shapes	
:
x
.gradients/mul_4_grad/Mul_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
z
0gradients/mul_4_grad/Mul_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
z
0gradients/mul_4_grad/Mul_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
˛
(gradients/mul_4_grad/Mul_1/strided_sliceStridedSlicegradients/GatherV2_2_grad/Cast.gradients/mul_4_grad/Mul_1/strided_slice/stack0gradients/mul_4_grad/Mul_1/strided_slice/stack_10gradients/mul_4_grad/Mul_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ú
gradients/mul_4_grad/Mul_1/yUnsortedSegmentSum!gradients/GatherV2_2_grad/Reshape#gradients/GatherV2_2_grad/Reshape_1(gradients/mul_4_grad/Mul_1/strided_slice*
T0*
Tindices0*
_output_shapes	
:
l
gradients/mul_4_grad/Mul_1MulSum_3gradients/mul_4_grad/Mul_1/y*
T0*
_output_shapes	
:

gradients/mul_4_grad/Sum_1Sumgradients/mul_4_grad/Mul_1,gradients/mul_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes	
:

gradients/mul_4_grad/Reshape_1Reshapegradients/mul_4_grad/Sum_1gradients/mul_4_grad/Shape_1*
T0*
_output_shapes	
:
m
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Reshape^gradients/mul_4_grad/Reshape_1
Ö
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Reshape&^gradients/mul_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_4_grad/Reshape*
_output_shapes	
:
Ü
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Reshape_1&^gradients/mul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_4_grad/Reshape_1*
_output_shapes	
:
r
gradients/Neg_grad/NegNeg-gradients/mul_2_grad/tuple/control_dependency*
T0*
_output_shapes	
:
e
gradients/mul_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
g
gradients/mul_3_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
ą
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
v
,gradients/mul_3_grad/Mul/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
x
.gradients/mul_3_grad/Mul/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
x
.gradients/mul_3_grad/Mul/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ş
&gradients/mul_3_grad/Mul/strided_sliceStridedSlicegradients/GatherV2_1_grad/Cast,gradients/mul_3_grad/Mul/strided_slice/stack.gradients/mul_3_grad/Mul/strided_slice/stack_1.gradients/mul_3_grad/Mul/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ö
gradients/mul_3_grad/Mul/xUnsortedSegmentSum!gradients/GatherV2_1_grad/Reshape#gradients/GatherV2_1_grad/Reshape_1&gradients/mul_3_grad/Mul/strided_slice*
T0*
Tindices0*
_output_shapes	
:
m
gradients/mul_3_grad/MulMulgradients/mul_3_grad/Mul/x
SelectV2_2*
T0*
_output_shapes	
:

gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes	
:

gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*
_output_shapes	
:
x
.gradients/mul_3_grad/Mul_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
z
0gradients/mul_3_grad/Mul_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
z
0gradients/mul_3_grad/Mul_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
˛
(gradients/mul_3_grad/Mul_1/strided_sliceStridedSlicegradients/GatherV2_1_grad/Cast.gradients/mul_3_grad/Mul_1/strided_slice/stack0gradients/mul_3_grad/Mul_1/strided_slice/stack_10gradients/mul_3_grad/Mul_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ú
gradients/mul_3_grad/Mul_1/yUnsortedSegmentSum!gradients/GatherV2_1_grad/Reshape#gradients/GatherV2_1_grad/Reshape_1(gradients/mul_3_grad/Mul_1/strided_slice*
T0*
Tindices0*
_output_shapes	
:
l
gradients/mul_3_grad/Mul_1MulSum_1gradients/mul_3_grad/Mul_1/y*
T0*
_output_shapes	
:

gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes	
:

gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
_output_shapes	
:
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
Ö
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*
_output_shapes	
:
Ü
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*
_output_shapes	
:
o
gradients/Sum_3_grad/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"     
`
gradients/Sum_3_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_3_grad/MaximumMaximumgradients/Sum_3_grad/Maximum/xgradients/Sum_3_grad/Maximum/y*
T0*
_output_shapes
:
p
gradients/Sum_3_grad/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"  
   

gradients/Sum_3_grad/floordivFloorDivgradients/Sum_3_grad/floordiv/xgradients/Sum_3_grad/Maximum*
T0*
_output_shapes
:
s
"gradients/Sum_3_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     
¤
gradients/Sum_3_grad/ReshapeReshape-gradients/mul_4_grad/tuple/control_dependency"gradients/Sum_3_grad/Reshape/shape*
T0*
_output_shapes
:	
t
#gradients/Sum_3_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"   
   

gradients/Sum_3_grad/TileTilegradients/Sum_3_grad/Reshape#gradients/Sum_3_grad/Tile/multiples*
T0*
_output_shapes
:	

q
gradients/Log_grad/Reciprocal
Reciprocaladd_1^gradients/Neg_grad/Neg*
T0*
_output_shapes	
:
z
gradients/Log_grad/mulMulgradients/Neg_grad/Neggradients/Log_grad/Reciprocal*
T0*
_output_shapes	
:
o
gradients/Sum_1_grad/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"     
`
gradients/Sum_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_1_grad/MaximumMaximumgradients/Sum_1_grad/Maximum/xgradients/Sum_1_grad/Maximum/y*
T0*
_output_shapes
:
p
gradients/Sum_1_grad/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"     

gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/floordiv/xgradients/Sum_1_grad/Maximum*
T0*
_output_shapes
:
s
"gradients/Sum_1_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"     
¤
gradients/Sum_1_grad/ReshapeReshape-gradients/mul_3_grad/tuple/control_dependency"gradients/Sum_1_grad/Reshape/shape*
T0*
_output_shapes
:	
t
#gradients/Sum_1_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      

gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshape#gradients/Sum_1_grad/Tile/multiples*
T0*
_output_shapes
:	
~
gradients/Square_1_grad/ConstConst^gradients/Sum_3_grad/Tile*
_output_shapes
: *
dtype0*
valueB
 *   @
r
gradients/Square_1_grad/MulMulsub_2gradients/Square_1_grad/Const*
T0*
_output_shapes
:	


gradients/Square_1_grad/Mul_1Mulgradients/Sum_3_grad/Tilegradients/Square_1_grad/Mul*
T0*
_output_shapes
:	

x
-gradients/add_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB:
p
-gradients/add_1_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
Ő
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/add_1_grad/BroadcastGradientArgs/s0-gradients/add_1_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
t
*gradients/add_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 

gradients/add_1_grad/SumSumgradients/Log_grad/mul*gradients/add_1_grad/Sum/reduction_indices*
T0*
_output_shapes
: 
e
"gradients/add_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
g
$gradients/add_1_grad/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sum$gradients/add_1_grad/Reshape/shape_1*
T0*
_output_shapes
: 
e
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/Log_grad/mul^gradients/add_1_grad/Reshape
Ę
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/Log_grad/mul&^gradients/add_1_grad/tuple/group_deps*
T0*)
_class
loc:@gradients/Log_grad/mul*
_output_shapes	
:
Ó
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
_output_shapes
: 
|
gradients/Square_grad/ConstConst^gradients/Sum_1_grad/Tile*
_output_shapes
: *
dtype0*
valueB
 *   @
n
gradients/Square_grad/MulMulsub_1gradients/Square_grad/Const*
T0*
_output_shapes
:	

gradients/Square_grad/Mul_1Mulgradients/Sum_1_grad/Tilegradients/Square_grad/Mul*
T0*
_output_shapes
:	
h
gradients/sub_2_grad/NegNeggradients/Square_1_grad/Mul_1*
T0*
_output_shapes
:	

h
%gradients/sub_2_grad/tuple/group_depsNoOp^gradients/Square_1_grad/Mul_1^gradients/sub_2_grad/Neg
Ü
-gradients/sub_2_grad/tuple/control_dependencyIdentitygradients/Square_1_grad/Mul_1&^gradients/sub_2_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/Square_1_grad/Mul_1*
_output_shapes
:	

Ô
/gradients/sub_2_grad/tuple/control_dependency_1Identitygradients/sub_2_grad/Neg&^gradients/sub_2_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_2_grad/Neg*
_output_shapes
:	

m
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     
 
gradients/Squeeze_grad/ReshapeReshape-gradients/add_1_grad/tuple/control_dependencygradients/Squeeze_grad/Shape*
T0*
_output_shapes
:	
f
gradients/sub_1_grad/NegNeggradients/Square_grad/Mul_1*
T0*
_output_shapes
:	
f
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/Square_grad/Mul_1^gradients/sub_1_grad/Neg
Ř
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/Square_grad/Mul_1&^gradients/sub_1_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/Square_grad/Mul_1*
_output_shapes
:	
Ô
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Neg&^gradients/sub_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_1_grad/Neg*
_output_shapes
:	
{
"gradients/landmark_pred_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"        
   
´
$gradients/landmark_pred_grad/ReshapeReshape-gradients/sub_2_grad/tuple/control_dependency"gradients/landmark_pred_grad/Shape*
T0*'
_output_shapes
:


gradients/GatherV2_grad/ShapeConst*
_class
loc:@Reshape_4*
_output_shapes
:*
dtype0	*%
valueB	"              

gradients/GatherV2_grad/CastCastgradients/GatherV2_grad/Shape*

DstT0*

SrcT0	*
_class
loc:@Reshape_4*
_output_shapes
:
_
gradients/GatherV2_grad/SizeConst*
_output_shapes
: *
dtype0*
value
B :
h
&gradients/GatherV2_grad/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

"gradients/GatherV2_grad/ExpandDims
ExpandDimsgradients/GatherV2_grad/Size&gradients/GatherV2_grad/ExpandDims/dim*
T0*
_output_shapes
:
u
+gradients/GatherV2_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
w
-gradients/GatherV2_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
w
-gradients/GatherV2_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
%gradients/GatherV2_grad/strided_sliceStridedSlicegradients/GatherV2_grad/Cast+gradients/GatherV2_grad/strided_slice/stack-gradients/GatherV2_grad/strided_slice/stack_1-gradients/GatherV2_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*
end_mask
e
#gradients/GatherV2_grad/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Č
gradients/GatherV2_grad/concatConcatV2"gradients/GatherV2_grad/ExpandDims%gradients/GatherV2_grad/strided_slice#gradients/GatherV2_grad/concat/axis*
N*
T0*
_output_shapes
:

gradients/GatherV2_grad/ReshapeReshapegradients/Squeeze_grad/Reshapegradients/GatherV2_grad/concat*
T0*
_output_shapes
:	
{
!gradients/GatherV2_grad/Reshape_1Reshapeadd"gradients/GatherV2_grad/ExpandDims*
T0*
_output_shapes	
:
w
gradients/bbox_pred_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"           
Ź
 gradients/bbox_pred_grad/ReshapeReshape-gradients/sub_1_grad/tuple/control_dependencygradients/bbox_pred_grad/Shape*
T0*'
_output_shapes
:

*gradients/conv4_3/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/landmark_pred_grad/Reshape*
T0*
_output_shapes
:


/gradients/conv4_3/BiasAdd_grad/tuple/group_depsNoOp+^gradients/conv4_3/BiasAdd_grad/BiasAddGrad%^gradients/landmark_pred_grad/Reshape

7gradients/conv4_3/BiasAdd_grad/tuple/control_dependencyIdentity$gradients/landmark_pred_grad/Reshape0^gradients/conv4_3/BiasAdd_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/landmark_pred_grad/Reshape*'
_output_shapes
:


9gradients/conv4_3/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv4_3/BiasAdd_grad/BiasAddGrad0^gradients/conv4_3/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv4_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

v
,gradients/Reshape_4_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
x
.gradients/Reshape_4_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
x
.gradients/Reshape_4_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
¨
&gradients/Reshape_4_grad/strided_sliceStridedSlicegradients/GatherV2_grad/Cast,gradients/Reshape_4_grad/strided_slice/stack.gradients/Reshape_4_grad/strided_slice/stack_1.gradients/Reshape_4_grad/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ç
+gradients/Reshape_4_grad/UnsortedSegmentSumUnsortedSegmentSumgradients/GatherV2_grad/Reshape!gradients/GatherV2_grad/Reshape_1&gradients/Reshape_4_grad/strided_slice*
T0*
Tindices0*
_output_shapes
:	
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     
˘
 gradients/Reshape_4_grad/ReshapeReshape+gradients/Reshape_4_grad/UnsortedSegmentSumgradients/Reshape_4_grad/Shape*
T0*
_output_shapes
:	

*gradients/conv4_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/bbox_pred_grad/Reshape*
T0*
_output_shapes
:

/gradients/conv4_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients/bbox_pred_grad/Reshape+^gradients/conv4_2/BiasAdd_grad/BiasAddGrad
ţ
7gradients/conv4_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients/bbox_pred_grad/Reshape0^gradients/conv4_2/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/bbox_pred_grad/Reshape*'
_output_shapes
:

9gradients/conv4_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv4_2/BiasAdd_grad/BiasAddGrad0^gradients/conv4_2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv4_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

$gradients/conv4_3/Conv2D_grad/ShapeNShapeN	conv3/addconv4_3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ą
1gradients/conv4_3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/conv4_3/Conv2D_grad/ShapeNconv4_3/Conv2D/ReadVariableOp7gradients/conv4_3/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
: *
paddingVALID*
strides


2gradients/conv4_3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	conv3/add&gradients/conv4_3/Conv2D_grad/ShapeN:17gradients/conv4_3/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
: 
*
paddingVALID*
strides


.gradients/conv4_3/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv4_3/Conv2D_grad/Conv2DBackpropFilter2^gradients/conv4_3/Conv2D_grad/Conv2DBackpropInput

6gradients/conv4_3/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv4_3/Conv2D_grad/Conv2DBackpropInput/^gradients/conv4_3/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv4_3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
: 
Ą
8gradients/conv4_3/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv4_3/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv4_3/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv4_3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 

v
gradients/cls_prob_grad/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"           

gradients/cls_prob_grad/ReshapeReshape gradients/Reshape_4_grad/Reshapegradients/cls_prob_grad/Shape*
T0*'
_output_shapes
:

$gradients/conv4_2/Conv2D_grad/ShapeNShapeN	conv3/addconv4_2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ą
1gradients/conv4_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/conv4_2/Conv2D_grad/ShapeNconv4_2/Conv2D/ReadVariableOp7gradients/conv4_2/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
: *
paddingVALID*
strides


2gradients/conv4_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	conv3/add&gradients/conv4_2/Conv2D_grad/ShapeN:17gradients/conv4_2/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
: *
paddingVALID*
strides


.gradients/conv4_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv4_2/Conv2D_grad/Conv2DBackpropFilter2^gradients/conv4_2/Conv2D_grad/Conv2DBackpropInput

6gradients/conv4_2/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv4_2/Conv2D_grad/Conv2DBackpropInput/^gradients/conv4_2/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv4_2/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
: 
Ą
8gradients/conv4_2/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv4_2/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv4_2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv4_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 

"gradients/conv4_1/Softmax_grad/mulMulgradients/cls_prob_grad/Reshapeconv4_1/Softmax*
T0*'
_output_shapes
:

4gradients/conv4_1/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Ć
"gradients/conv4_1/Softmax_grad/SumSum"gradients/conv4_1/Softmax_grad/mul4gradients/conv4_1/Softmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:*
	keep_dims(
 
"gradients/conv4_1/Softmax_grad/subSubgradients/cls_prob_grad/Reshape"gradients/conv4_1/Softmax_grad/Sum*
T0*'
_output_shapes
:

$gradients/conv4_1/Softmax_grad/mul_1Mul"gradients/conv4_1/Softmax_grad/subconv4_1/Softmax*
T0*'
_output_shapes
:
§
gradients/AddNAddNCgradients/conv4_3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul8gradients/conv4_3/Conv2D_grad/tuple/control_dependency_1*
N*
T0*V
_classL
JHloc:@gradients/conv4_3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
: 


*gradients/conv4_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/conv4_1/Softmax_grad/mul_1*
T0*
_output_shapes
:

/gradients/conv4_1/BiasAdd_grad/tuple/group_depsNoOp+^gradients/conv4_1/BiasAdd_grad/BiasAddGrad%^gradients/conv4_1/Softmax_grad/mul_1

7gradients/conv4_1/BiasAdd_grad/tuple/control_dependencyIdentity$gradients/conv4_1/Softmax_grad/mul_10^gradients/conv4_1/BiasAdd_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/conv4_1/Softmax_grad/mul_1*'
_output_shapes
:

9gradients/conv4_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/conv4_1/BiasAdd_grad/BiasAddGrad0^gradients/conv4_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/conv4_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Š
gradients/AddN_1AddNCgradients/conv4_2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul8gradients/conv4_2/Conv2D_grad/tuple/control_dependency_1*
N*
T0*V
_classL
JHloc:@gradients/conv4_2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
: 

$gradients/conv4_1/Conv2D_grad/ShapeNShapeN	conv3/addconv4_1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::
Ą
1gradients/conv4_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/conv4_1/Conv2D_grad/ShapeNconv4_1/Conv2D/ReadVariableOp7gradients/conv4_1/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
: *
paddingVALID*
strides


2gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	conv3/add&gradients/conv4_1/Conv2D_grad/ShapeN:17gradients/conv4_1/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
: *
paddingVALID*
strides


.gradients/conv4_1/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilter2^gradients/conv4_1/Conv2D_grad/Conv2DBackpropInput

6gradients/conv4_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/conv4_1/Conv2D_grad/Conv2DBackpropInput/^gradients/conv4_1/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv4_1/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
: 
Ą
8gradients/conv4_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/conv4_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/conv4_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Á
gradients/AddN_2AddN6gradients/conv4_3/Conv2D_grad/tuple/control_dependency6gradients/conv4_2/Conv2D_grad/tuple/control_dependency6gradients/conv4_1/Conv2D_grad/tuple/control_dependency*
N*
T0*D
_class:
86loc:@gradients/conv4_3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
: 
D
)gradients/conv3/add_grad/tuple/group_depsNoOp^gradients/AddN_2
ó
1gradients/conv3/add_grad/tuple/control_dependencyIdentitygradients/AddN_2*^gradients/conv3/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv4_3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
: 
ő
3gradients/conv3/add_grad/tuple/control_dependency_1Identitygradients/AddN_2*^gradients/conv3/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/conv4_3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
: 

"gradients/conv3/Relu_grad/ReluGradReluGrad1gradients/conv3/add_grad/tuple/control_dependency
conv3/Relu*
T0*'
_output_shapes
: 

3gradients/conv3/mul_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"            
v
3gradients/conv3/mul_1_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
ç
0gradients/conv3/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/conv3/mul_1_grad/BroadcastGradientArgs/s03gradients/conv3/mul_1_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/conv3/mul_1_grad/MulMul3gradients/conv3/add_grad/tuple/control_dependency_1conv3/mul_1/y*
T0*'
_output_shapes
: 

 gradients/conv3/mul_1_grad/Mul_1Mul	conv3/mul3gradients/conv3/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
: 

0gradients/conv3/mul_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             

gradients/conv3/mul_1_grad/SumSum gradients/conv3/mul_1_grad/Mul_10gradients/conv3/mul_1_grad/Sum/reduction_indices*
T0*
_output_shapes
: 
k
(gradients/conv3/mul_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
m
*gradients/conv3/mul_1_grad/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 

"gradients/conv3/mul_1_grad/ReshapeReshapegradients/conv3/mul_1_grad/Sum*gradients/conv3/mul_1_grad/Reshape/shape_1*
T0*
_output_shapes
: 
y
+gradients/conv3/mul_1_grad/tuple/group_depsNoOp^gradients/conv3/mul_1_grad/Mul#^gradients/conv3/mul_1_grad/Reshape
ň
3gradients/conv3/mul_1_grad/tuple/control_dependencyIdentitygradients/conv3/mul_1_grad/Mul,^gradients/conv3/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv3/mul_1_grad/Mul*'
_output_shapes
: 
ë
5gradients/conv3/mul_1_grad/tuple/control_dependency_1Identity"gradients/conv3/mul_1_grad/Reshape,^gradients/conv3/mul_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv3/mul_1_grad/Reshape*
_output_shapes
: 
Š
gradients/AddN_3AddNCgradients/conv4_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul8gradients/conv4_1/Conv2D_grad/tuple/control_dependency_1*
N*
T0*V
_classL
JHloc:@gradients/conv4_1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
: 
{
1gradients/conv3/mul_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB: 

1gradients/conv3/mul_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"            
á
.gradients/conv3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/conv3/mul_grad/BroadcastGradientArgs/s01gradients/conv3/mul_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/conv3/mul_grad/MulMul3gradients/conv3/mul_1_grad/tuple/control_dependency	conv3/sub*
T0*'
_output_shapes
: 

.gradients/conv3/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          

gradients/conv3/mul_grad/SumSumgradients/conv3/mul_grad/Mul.gradients/conv3/mul_grad/Sum/reduction_indices*
T0*
_output_shapes
: 
p
&gradients/conv3/mul_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 

 gradients/conv3/mul_grad/ReshapeReshapegradients/conv3/mul_grad/Sum&gradients/conv3/mul_grad/Reshape/shape*
T0*
_output_shapes
: 
˘
gradients/conv3/mul_grad/Mul_1Mulconv3/ReadVariableOp3gradients/conv3/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
: 
u
)gradients/conv3/mul_grad/tuple/group_depsNoOp^gradients/conv3/mul_grad/Mul_1!^gradients/conv3/mul_grad/Reshape
ĺ
1gradients/conv3/mul_grad/tuple/control_dependencyIdentity gradients/conv3/mul_grad/Reshape*^gradients/conv3/mul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/conv3/mul_grad/Reshape*
_output_shapes
: 
đ
3gradients/conv3/mul_grad/tuple/control_dependency_1Identitygradients/conv3/mul_grad/Mul_1*^gradients/conv3/mul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv3/mul_grad/Mul_1*'
_output_shapes
: 

gradients/conv3/sub_grad/NegNeg3gradients/conv3/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
: 

)gradients/conv3/sub_grad/tuple/group_depsNoOp4^gradients/conv3/mul_grad/tuple/control_dependency_1^gradients/conv3/sub_grad/Neg

1gradients/conv3/sub_grad/tuple/control_dependencyIdentity3gradients/conv3/mul_grad/tuple/control_dependency_1*^gradients/conv3/sub_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv3/mul_grad/Mul_1*'
_output_shapes
: 
ě
3gradients/conv3/sub_grad/tuple/control_dependency_1Identitygradients/conv3/sub_grad/Neg*^gradients/conv3/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/conv3/sub_grad/Neg*'
_output_shapes
: 
f
gradients/conv3/Abs_grad/SignSignconv3/BiasAdd*
T0*'
_output_shapes
: 
Š
gradients/conv3/Abs_grad/mulMul3gradients/conv3/sub_grad/tuple/control_dependency_1gradients/conv3/Abs_grad/Sign*
T0*'
_output_shapes
: 
˙
gradients/AddN_4AddN"gradients/conv3/Relu_grad/ReluGrad1gradients/conv3/sub_grad/tuple/control_dependencygradients/conv3/Abs_grad/mul*
N*
T0*5
_class+
)'loc:@gradients/conv3/Relu_grad/ReluGrad*'
_output_shapes
: 
n
(gradients/conv3/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_4*
T0*
_output_shapes
: 
s
-gradients/conv3/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_4)^gradients/conv3/BiasAdd_grad/BiasAddGrad
ě
5gradients/conv3/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_4.^gradients/conv3/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv3/Relu_grad/ReluGrad*'
_output_shapes
: 
˙
7gradients/conv3/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv3/BiasAdd_grad/BiasAddGrad.^gradients/conv3/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

"gradients/conv3/Conv2D_grad/ShapeNShapeN	conv2/addconv3/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::

/gradients/conv3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv3/Conv2D_grad/ShapeNconv3/Conv2D/ReadVariableOp5gradients/conv3/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
:*
paddingVALID*
strides


0gradients/conv3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	conv2/add$gradients/conv3/Conv2D_grad/ShapeN:15gradients/conv3/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
: *
paddingVALID*
strides


,gradients/conv3/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv3/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv3/Conv2D_grad/Conv2DBackpropInput

4gradients/conv3/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv3/Conv2D_grad/Conv2DBackpropInput-^gradients/conv3/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:

6gradients/conv3/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv3/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv3/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv3/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
h
)gradients/conv2/add_grad/tuple/group_depsNoOp5^gradients/conv3/Conv2D_grad/tuple/control_dependency

1gradients/conv2/add_grad/tuple/control_dependencyIdentity4gradients/conv3/Conv2D_grad/tuple/control_dependency*^gradients/conv2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:

3gradients/conv2/add_grad/tuple/control_dependency_1Identity4gradients/conv3/Conv2D_grad/tuple/control_dependency*^gradients/conv2/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv3/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:

"gradients/conv2/Relu_grad/ReluGradReluGrad1gradients/conv2/add_grad/tuple/control_dependency
conv2/Relu*
T0*'
_output_shapes
:

3gradients/conv2/mul_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"           
v
3gradients/conv2/mul_1_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
ç
0gradients/conv2/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/conv2/mul_1_grad/BroadcastGradientArgs/s03gradients/conv2/mul_1_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/conv2/mul_1_grad/MulMul3gradients/conv2/add_grad/tuple/control_dependency_1conv2/mul_1/y*
T0*'
_output_shapes
:

 gradients/conv2/mul_1_grad/Mul_1Mul	conv2/mul3gradients/conv2/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:

0gradients/conv2/mul_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             

gradients/conv2/mul_1_grad/SumSum gradients/conv2/mul_1_grad/Mul_10gradients/conv2/mul_1_grad/Sum/reduction_indices*
T0*
_output_shapes
: 
k
(gradients/conv2/mul_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
m
*gradients/conv2/mul_1_grad/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 

"gradients/conv2/mul_1_grad/ReshapeReshapegradients/conv2/mul_1_grad/Sum*gradients/conv2/mul_1_grad/Reshape/shape_1*
T0*
_output_shapes
: 
y
+gradients/conv2/mul_1_grad/tuple/group_depsNoOp^gradients/conv2/mul_1_grad/Mul#^gradients/conv2/mul_1_grad/Reshape
ň
3gradients/conv2/mul_1_grad/tuple/control_dependencyIdentitygradients/conv2/mul_1_grad/Mul,^gradients/conv2/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv2/mul_1_grad/Mul*'
_output_shapes
:
ë
5gradients/conv2/mul_1_grad/tuple/control_dependency_1Identity"gradients/conv2/mul_1_grad/Reshape,^gradients/conv2/mul_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv2/mul_1_grad/Reshape*
_output_shapes
: 
Ł
gradients/AddN_5AddNAgradients/conv3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul6gradients/conv3/Conv2D_grad/tuple/control_dependency_1*
N*
T0*T
_classJ
HFloc:@gradients/conv3/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
: 
{
1gradients/conv2/mul_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB:

1gradients/conv2/mul_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"           
á
.gradients/conv2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/conv2/mul_grad/BroadcastGradientArgs/s01gradients/conv2/mul_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/conv2/mul_grad/MulMul3gradients/conv2/mul_1_grad/tuple/control_dependency	conv2/sub*
T0*'
_output_shapes
:

.gradients/conv2/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          

gradients/conv2/mul_grad/SumSumgradients/conv2/mul_grad/Mul.gradients/conv2/mul_grad/Sum/reduction_indices*
T0*
_output_shapes
:
p
&gradients/conv2/mul_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:

 gradients/conv2/mul_grad/ReshapeReshapegradients/conv2/mul_grad/Sum&gradients/conv2/mul_grad/Reshape/shape*
T0*
_output_shapes
:
˘
gradients/conv2/mul_grad/Mul_1Mulconv2/ReadVariableOp3gradients/conv2/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:
u
)gradients/conv2/mul_grad/tuple/group_depsNoOp^gradients/conv2/mul_grad/Mul_1!^gradients/conv2/mul_grad/Reshape
ĺ
1gradients/conv2/mul_grad/tuple/control_dependencyIdentity gradients/conv2/mul_grad/Reshape*^gradients/conv2/mul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/conv2/mul_grad/Reshape*
_output_shapes
:
đ
3gradients/conv2/mul_grad/tuple/control_dependency_1Identitygradients/conv2/mul_grad/Mul_1*^gradients/conv2/mul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv2/mul_grad/Mul_1*'
_output_shapes
:

gradients/conv2/sub_grad/NegNeg3gradients/conv2/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:

)gradients/conv2/sub_grad/tuple/group_depsNoOp4^gradients/conv2/mul_grad/tuple/control_dependency_1^gradients/conv2/sub_grad/Neg

1gradients/conv2/sub_grad/tuple/control_dependencyIdentity3gradients/conv2/mul_grad/tuple/control_dependency_1*^gradients/conv2/sub_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv2/mul_grad/Mul_1*'
_output_shapes
:
ě
3gradients/conv2/sub_grad/tuple/control_dependency_1Identitygradients/conv2/sub_grad/Neg*^gradients/conv2/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/conv2/sub_grad/Neg*'
_output_shapes
:
f
gradients/conv2/Abs_grad/SignSignconv2/BiasAdd*
T0*'
_output_shapes
:
Š
gradients/conv2/Abs_grad/mulMul3gradients/conv2/sub_grad/tuple/control_dependency_1gradients/conv2/Abs_grad/Sign*
T0*'
_output_shapes
:
˙
gradients/AddN_6AddN"gradients/conv2/Relu_grad/ReluGrad1gradients/conv2/sub_grad/tuple/control_dependencygradients/conv2/Abs_grad/mul*
N*
T0*5
_class+
)'loc:@gradients/conv2/Relu_grad/ReluGrad*'
_output_shapes
:
n
(gradients/conv2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_6*
T0*
_output_shapes
:
s
-gradients/conv2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_6)^gradients/conv2/BiasAdd_grad/BiasAddGrad
ě
5gradients/conv2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_6.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv2/Relu_grad/ReluGrad*'
_output_shapes
:
˙
7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv2/BiasAdd_grad/BiasAddGrad.^gradients/conv2/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

"gradients/conv2/Conv2D_grad/ShapeNShapeNpool1/MaxPoolconv2/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::

/gradients/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv2/Conv2D_grad/ShapeNconv2/Conv2D/ReadVariableOp5gradients/conv2/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
:
*
paddingVALID*
strides


0gradients/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterpool1/MaxPool$gradients/conv2/Conv2D_grad/ShapeN:15gradients/conv2/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
:
*
paddingVALID*
strides


,gradients/conv2/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2/Conv2D_grad/Conv2DBackpropInput

4gradients/conv2/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv2/Conv2D_grad/Conv2DBackpropInput-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:


6gradients/conv2/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv2/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv2/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

ë
(gradients/pool1/MaxPool_grad/MaxPoolGradMaxPoolGrad	conv1/addpool1/MaxPool4gradients/conv2/Conv2D_grad/tuple/control_dependency*'
_output_shapes
:


*
ksize
*
paddingSAME*
strides

\
)gradients/conv1/add_grad/tuple/group_depsNoOp)^gradients/pool1/MaxPool_grad/MaxPoolGrad

1gradients/conv1/add_grad/tuple/control_dependencyIdentity(gradients/pool1/MaxPool_grad/MaxPoolGrad*^gradients/conv1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/pool1/MaxPool_grad/MaxPoolGrad*'
_output_shapes
:




3gradients/conv1/add_grad/tuple/control_dependency_1Identity(gradients/pool1/MaxPool_grad/MaxPoolGrad*^gradients/conv1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/pool1/MaxPool_grad/MaxPoolGrad*'
_output_shapes
:



Ł
gradients/AddN_7AddNAgradients/conv2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul6gradients/conv2/Conv2D_grad/tuple/control_dependency_1*
N*
T0*T
_classJ
HFloc:@gradients/conv2/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
:


"gradients/conv1/Relu_grad/ReluGradReluGrad1gradients/conv1/add_grad/tuple/control_dependency
conv1/Relu*
T0*'
_output_shapes
:




3gradients/conv1/mul_1_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*%
valueB"  
   
   
   
v
3gradients/conv1/mul_1_grad/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 
ç
0gradients/conv1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs3gradients/conv1/mul_1_grad/BroadcastGradientArgs/s03gradients/conv1/mul_1_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/conv1/mul_1_grad/MulMul3gradients/conv1/add_grad/tuple/control_dependency_1conv1/mul_1/y*
T0*'
_output_shapes
:




 gradients/conv1/mul_1_grad/Mul_1Mul	conv1/mul3gradients/conv1/add_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:




0gradients/conv1/mul_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             

gradients/conv1/mul_1_grad/SumSum gradients/conv1/mul_1_grad/Mul_10gradients/conv1/mul_1_grad/Sum/reduction_indices*
T0*
_output_shapes
: 
k
(gradients/conv1/mul_1_grad/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 
m
*gradients/conv1/mul_1_grad/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 

"gradients/conv1/mul_1_grad/ReshapeReshapegradients/conv1/mul_1_grad/Sum*gradients/conv1/mul_1_grad/Reshape/shape_1*
T0*
_output_shapes
: 
y
+gradients/conv1/mul_1_grad/tuple/group_depsNoOp^gradients/conv1/mul_1_grad/Mul#^gradients/conv1/mul_1_grad/Reshape
ň
3gradients/conv1/mul_1_grad/tuple/control_dependencyIdentitygradients/conv1/mul_1_grad/Mul,^gradients/conv1/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv1/mul_1_grad/Mul*'
_output_shapes
:



ë
5gradients/conv1/mul_1_grad/tuple/control_dependency_1Identity"gradients/conv1/mul_1_grad/Reshape,^gradients/conv1/mul_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv1/mul_1_grad/Reshape*
_output_shapes
: 
{
1gradients/conv1/mul_grad/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB:


1gradients/conv1/mul_grad/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*%
valueB"  
   
   
   
á
.gradients/conv1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs1gradients/conv1/mul_grad/BroadcastGradientArgs/s01gradients/conv1/mul_grad/BroadcastGradientArgs/s1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

gradients/conv1/mul_grad/MulMul3gradients/conv1/mul_1_grad/tuple/control_dependency	conv1/sub*
T0*'
_output_shapes
:




.gradients/conv1/mul_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          

gradients/conv1/mul_grad/SumSumgradients/conv1/mul_grad/Mul.gradients/conv1/mul_grad/Sum/reduction_indices*
T0*
_output_shapes
:

p
&gradients/conv1/mul_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:


 gradients/conv1/mul_grad/ReshapeReshapegradients/conv1/mul_grad/Sum&gradients/conv1/mul_grad/Reshape/shape*
T0*
_output_shapes
:

˘
gradients/conv1/mul_grad/Mul_1Mulconv1/ReadVariableOp3gradients/conv1/mul_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:



u
)gradients/conv1/mul_grad/tuple/group_depsNoOp^gradients/conv1/mul_grad/Mul_1!^gradients/conv1/mul_grad/Reshape
ĺ
1gradients/conv1/mul_grad/tuple/control_dependencyIdentity gradients/conv1/mul_grad/Reshape*^gradients/conv1/mul_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/conv1/mul_grad/Reshape*
_output_shapes
:

đ
3gradients/conv1/mul_grad/tuple/control_dependency_1Identitygradients/conv1/mul_grad/Mul_1*^gradients/conv1/mul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv1/mul_grad/Mul_1*'
_output_shapes
:




gradients/conv1/sub_grad/NegNeg3gradients/conv1/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:




)gradients/conv1/sub_grad/tuple/group_depsNoOp4^gradients/conv1/mul_grad/tuple/control_dependency_1^gradients/conv1/sub_grad/Neg

1gradients/conv1/sub_grad/tuple/control_dependencyIdentity3gradients/conv1/mul_grad/tuple/control_dependency_1*^gradients/conv1/sub_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/conv1/mul_grad/Mul_1*'
_output_shapes
:



ě
3gradients/conv1/sub_grad/tuple/control_dependency_1Identitygradients/conv1/sub_grad/Neg*^gradients/conv1/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/conv1/sub_grad/Neg*'
_output_shapes
:



f
gradients/conv1/Abs_grad/SignSignconv1/BiasAdd*
T0*'
_output_shapes
:



Š
gradients/conv1/Abs_grad/mulMul3gradients/conv1/sub_grad/tuple/control_dependency_1gradients/conv1/Abs_grad/Sign*
T0*'
_output_shapes
:



˙
gradients/AddN_8AddN"gradients/conv1/Relu_grad/ReluGrad1gradients/conv1/sub_grad/tuple/control_dependencygradients/conv1/Abs_grad/mul*
N*
T0*5
_class+
)'loc:@gradients/conv1/Relu_grad/ReluGrad*'
_output_shapes
:



n
(gradients/conv1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
_output_shapes
:

s
-gradients/conv1/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8)^gradients/conv1/BiasAdd_grad/BiasAddGrad
ě
5gradients/conv1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/conv1/Relu_grad/ReluGrad*'
_output_shapes
:



˙
7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/conv1/BiasAdd_grad/BiasAddGrad.^gradients/conv1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/conv1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:


"gradients/conv1/Conv2D_grad/ShapeNShapeNinput_imageconv1/Conv2D/ReadVariableOp*
N*
T0* 
_output_shapes
::

/gradients/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput"gradients/conv1/Conv2D_grad/ShapeNconv1/Conv2D/ReadVariableOp5gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*'
_output_shapes
:*
paddingVALID*
strides


0gradients/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput_image$gradients/conv1/Conv2D_grad/ShapeN:15gradients/conv1/BiasAdd_grad/tuple/control_dependency*
T0*&
_output_shapes
:
*
paddingVALID*
strides


,gradients/conv1/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv1/Conv2D_grad/Conv2DBackpropInput

4gradients/conv1/Conv2D_grad/tuple/control_dependencyIdentity/gradients/conv1/Conv2D_grad/Conv2DBackpropInput-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropInput*'
_output_shapes
:

6gradients/conv1/Conv2D_grad/tuple/control_dependency_1Identity0gradients/conv1/Conv2D_grad/Conv2DBackpropFilter-^gradients/conv1/Conv2D_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/conv1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

Ł
gradients/AddN_9AddNAgradients/conv1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul6gradients/conv1/Conv2D_grad/tuple/control_dependency_1*
N*
T0*T
_classJ
HFloc:@gradients/conv1/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*&
_output_shapes
:


%beta1_power/Initializer/initial_valueConst*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
valueB
 *fff?

beta1_powerVarHandleOp*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta1_power

,beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta1_power*
_class
loc:@conv1/alphas*
_output_shapes
: 
g
beta1_power/AssignAssignVariableOpbeta1_power%beta1_power/Initializer/initial_value*
dtype0

beta1_power/Read/ReadVariableOpReadVariableOpbeta1_power*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0

%beta2_power/Initializer/initial_valueConst*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
valueB
 *wž?

beta2_powerVarHandleOp*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta2_power

,beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta2_power*
_class
loc:@conv1/alphas*
_output_shapes
: 
g
beta2_power/AssignAssignVariableOpbeta2_power%beta2_power/Initializer/initial_value*
dtype0

beta2_power/Read/ReadVariableOpReadVariableOpbeta2_power*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0
Ť
$conv1/weights/Adam/Initializer/zerosConst* 
_class
loc:@conv1/weights*&
_output_shapes
:
*
dtype0*%
valueB
*    
Ş
conv1/weights/AdamVarHandleOp* 
_class
loc:@conv1/weights*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameconv1/weights/Adam

3conv1/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/weights/Adam* 
_class
loc:@conv1/weights*
_output_shapes
: 
t
conv1/weights/Adam/AssignAssignVariableOpconv1/weights/Adam$conv1/weights/Adam/Initializer/zeros*
dtype0
Ł
&conv1/weights/Adam/Read/ReadVariableOpReadVariableOpconv1/weights/Adam* 
_class
loc:@conv1/weights*&
_output_shapes
:
*
dtype0
­
&conv1/weights/Adam_1/Initializer/zerosConst* 
_class
loc:@conv1/weights*&
_output_shapes
:
*
dtype0*%
valueB
*    
Ž
conv1/weights/Adam_1VarHandleOp* 
_class
loc:@conv1/weights*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameconv1/weights/Adam_1

5conv1/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/weights/Adam_1* 
_class
loc:@conv1/weights*
_output_shapes
: 
z
conv1/weights/Adam_1/AssignAssignVariableOpconv1/weights/Adam_1&conv1/weights/Adam_1/Initializer/zeros*
dtype0
§
(conv1/weights/Adam_1/Read/ReadVariableOpReadVariableOpconv1/weights/Adam_1* 
_class
loc:@conv1/weights*&
_output_shapes
:
*
dtype0

#conv1/biases/Adam/Initializer/zerosConst*
_class
loc:@conv1/biases*
_output_shapes
:
*
dtype0*
valueB
*    

conv1/biases/AdamVarHandleOp*
_class
loc:@conv1/biases*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameconv1/biases/Adam

2conv1/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/biases/Adam*
_class
loc:@conv1/biases*
_output_shapes
: 
q
conv1/biases/Adam/AssignAssignVariableOpconv1/biases/Adam#conv1/biases/Adam/Initializer/zeros*
dtype0

%conv1/biases/Adam/Read/ReadVariableOpReadVariableOpconv1/biases/Adam*
_class
loc:@conv1/biases*
_output_shapes
:
*
dtype0

%conv1/biases/Adam_1/Initializer/zerosConst*
_class
loc:@conv1/biases*
_output_shapes
:
*
dtype0*
valueB
*    

conv1/biases/Adam_1VarHandleOp*
_class
loc:@conv1/biases*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameconv1/biases/Adam_1

4conv1/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/biases/Adam_1*
_class
loc:@conv1/biases*
_output_shapes
: 
w
conv1/biases/Adam_1/AssignAssignVariableOpconv1/biases/Adam_1%conv1/biases/Adam_1/Initializer/zeros*
dtype0

'conv1/biases/Adam_1/Read/ReadVariableOpReadVariableOpconv1/biases/Adam_1*
_class
loc:@conv1/biases*
_output_shapes
:
*
dtype0

#conv1/alphas/Adam/Initializer/zerosConst*
_class
loc:@conv1/alphas*
_output_shapes
:
*
dtype0*
valueB
*    

conv1/alphas/AdamVarHandleOp*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
shape:
*"
shared_nameconv1/alphas/Adam

2conv1/alphas/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/alphas/Adam*
_class
loc:@conv1/alphas*
_output_shapes
: 
q
conv1/alphas/Adam/AssignAssignVariableOpconv1/alphas/Adam#conv1/alphas/Adam/Initializer/zeros*
dtype0

%conv1/alphas/Adam/Read/ReadVariableOpReadVariableOpconv1/alphas/Adam*
_class
loc:@conv1/alphas*
_output_shapes
:
*
dtype0

%conv1/alphas/Adam_1/Initializer/zerosConst*
_class
loc:@conv1/alphas*
_output_shapes
:
*
dtype0*
valueB
*    

conv1/alphas/Adam_1VarHandleOp*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameconv1/alphas/Adam_1

4conv1/alphas/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv1/alphas/Adam_1*
_class
loc:@conv1/alphas*
_output_shapes
: 
w
conv1/alphas/Adam_1/AssignAssignVariableOpconv1/alphas/Adam_1%conv1/alphas/Adam_1/Initializer/zeros*
dtype0

'conv1/alphas/Adam_1/Read/ReadVariableOpReadVariableOpconv1/alphas/Adam_1*
_class
loc:@conv1/alphas*
_output_shapes
:
*
dtype0
Ż
4conv2/weights/Adam/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv2/weights*
_output_shapes
:*
dtype0*%
valueB"      
      

*conv2/weights/Adam/Initializer/zeros/ConstConst* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
valueB
 *    
á
$conv2/weights/Adam/Initializer/zerosFill4conv2/weights/Adam/Initializer/zeros/shape_as_tensor*conv2/weights/Adam/Initializer/zeros/Const*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:

Ş
conv2/weights/AdamVarHandleOp* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameconv2/weights/Adam

3conv2/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/weights/Adam* 
_class
loc:@conv2/weights*
_output_shapes
: 
t
conv2/weights/Adam/AssignAssignVariableOpconv2/weights/Adam$conv2/weights/Adam/Initializer/zeros*
dtype0
Ł
&conv2/weights/Adam/Read/ReadVariableOpReadVariableOpconv2/weights/Adam* 
_class
loc:@conv2/weights*&
_output_shapes
:
*
dtype0
ą
6conv2/weights/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv2/weights*
_output_shapes
:*
dtype0*%
valueB"      
      

,conv2/weights/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
valueB
 *    
ç
&conv2/weights/Adam_1/Initializer/zerosFill6conv2/weights/Adam_1/Initializer/zeros/shape_as_tensor,conv2/weights/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@conv2/weights*&
_output_shapes
:

Ž
conv2/weights/Adam_1VarHandleOp* 
_class
loc:@conv2/weights*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameconv2/weights/Adam_1

5conv2/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/weights/Adam_1* 
_class
loc:@conv2/weights*
_output_shapes
: 
z
conv2/weights/Adam_1/AssignAssignVariableOpconv2/weights/Adam_1&conv2/weights/Adam_1/Initializer/zeros*
dtype0
§
(conv2/weights/Adam_1/Read/ReadVariableOpReadVariableOpconv2/weights/Adam_1* 
_class
loc:@conv2/weights*&
_output_shapes
:
*
dtype0

#conv2/biases/Adam/Initializer/zerosConst*
_class
loc:@conv2/biases*
_output_shapes
:*
dtype0*
valueB*    

conv2/biases/AdamVarHandleOp*
_class
loc:@conv2/biases*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2/biases/Adam

2conv2/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/biases/Adam*
_class
loc:@conv2/biases*
_output_shapes
: 
q
conv2/biases/Adam/AssignAssignVariableOpconv2/biases/Adam#conv2/biases/Adam/Initializer/zeros*
dtype0

%conv2/biases/Adam/Read/ReadVariableOpReadVariableOpconv2/biases/Adam*
_class
loc:@conv2/biases*
_output_shapes
:*
dtype0

%conv2/biases/Adam_1/Initializer/zerosConst*
_class
loc:@conv2/biases*
_output_shapes
:*
dtype0*
valueB*    

conv2/biases/Adam_1VarHandleOp*
_class
loc:@conv2/biases*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv2/biases/Adam_1

4conv2/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/biases/Adam_1*
_class
loc:@conv2/biases*
_output_shapes
: 
w
conv2/biases/Adam_1/AssignAssignVariableOpconv2/biases/Adam_1%conv2/biases/Adam_1/Initializer/zeros*
dtype0

'conv2/biases/Adam_1/Read/ReadVariableOpReadVariableOpconv2/biases/Adam_1*
_class
loc:@conv2/biases*
_output_shapes
:*
dtype0

#conv2/alphas/Adam/Initializer/zerosConst*
_class
loc:@conv2/alphas*
_output_shapes
:*
dtype0*
valueB*    

conv2/alphas/AdamVarHandleOp*
_class
loc:@conv2/alphas*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2/alphas/Adam

2conv2/alphas/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/alphas/Adam*
_class
loc:@conv2/alphas*
_output_shapes
: 
q
conv2/alphas/Adam/AssignAssignVariableOpconv2/alphas/Adam#conv2/alphas/Adam/Initializer/zeros*
dtype0

%conv2/alphas/Adam/Read/ReadVariableOpReadVariableOpconv2/alphas/Adam*
_class
loc:@conv2/alphas*
_output_shapes
:*
dtype0

%conv2/alphas/Adam_1/Initializer/zerosConst*
_class
loc:@conv2/alphas*
_output_shapes
:*
dtype0*
valueB*    

conv2/alphas/Adam_1VarHandleOp*
_class
loc:@conv2/alphas*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv2/alphas/Adam_1

4conv2/alphas/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2/alphas/Adam_1*
_class
loc:@conv2/alphas*
_output_shapes
: 
w
conv2/alphas/Adam_1/AssignAssignVariableOpconv2/alphas/Adam_1%conv2/alphas/Adam_1/Initializer/zeros*
dtype0

'conv2/alphas/Adam_1/Read/ReadVariableOpReadVariableOpconv2/alphas/Adam_1*
_class
loc:@conv2/alphas*
_output_shapes
:*
dtype0
Ż
4conv3/weights/Adam/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv3/weights*
_output_shapes
:*
dtype0*%
valueB"             

*conv3/weights/Adam/Initializer/zeros/ConstConst* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
valueB
 *    
á
$conv3/weights/Adam/Initializer/zerosFill4conv3/weights/Adam/Initializer/zeros/shape_as_tensor*conv3/weights/Adam/Initializer/zeros/Const*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: 
Ş
conv3/weights/AdamVarHandleOp* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv3/weights/Adam

3conv3/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/weights/Adam* 
_class
loc:@conv3/weights*
_output_shapes
: 
t
conv3/weights/Adam/AssignAssignVariableOpconv3/weights/Adam$conv3/weights/Adam/Initializer/zeros*
dtype0
Ł
&conv3/weights/Adam/Read/ReadVariableOpReadVariableOpconv3/weights/Adam* 
_class
loc:@conv3/weights*&
_output_shapes
: *
dtype0
ą
6conv3/weights/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@conv3/weights*
_output_shapes
:*
dtype0*%
valueB"             

,conv3/weights/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
valueB
 *    
ç
&conv3/weights/Adam_1/Initializer/zerosFill6conv3/weights/Adam_1/Initializer/zeros/shape_as_tensor,conv3/weights/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@conv3/weights*&
_output_shapes
: 
Ž
conv3/weights/Adam_1VarHandleOp* 
_class
loc:@conv3/weights*
_output_shapes
: *
dtype0*
shape: *%
shared_nameconv3/weights/Adam_1

5conv3/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/weights/Adam_1* 
_class
loc:@conv3/weights*
_output_shapes
: 
z
conv3/weights/Adam_1/AssignAssignVariableOpconv3/weights/Adam_1&conv3/weights/Adam_1/Initializer/zeros*
dtype0
§
(conv3/weights/Adam_1/Read/ReadVariableOpReadVariableOpconv3/weights/Adam_1* 
_class
loc:@conv3/weights*&
_output_shapes
: *
dtype0

#conv3/biases/Adam/Initializer/zerosConst*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0*
valueB *    

conv3/biases/AdamVarHandleOp*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv3/biases/Adam

2conv3/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/biases/Adam*
_class
loc:@conv3/biases*
_output_shapes
: 
q
conv3/biases/Adam/AssignAssignVariableOpconv3/biases/Adam#conv3/biases/Adam/Initializer/zeros*
dtype0

%conv3/biases/Adam/Read/ReadVariableOpReadVariableOpconv3/biases/Adam*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0

%conv3/biases/Adam_1/Initializer/zerosConst*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0*
valueB *    

conv3/biases/Adam_1VarHandleOp*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv3/biases/Adam_1

4conv3/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/biases/Adam_1*
_class
loc:@conv3/biases*
_output_shapes
: 
w
conv3/biases/Adam_1/AssignAssignVariableOpconv3/biases/Adam_1%conv3/biases/Adam_1/Initializer/zeros*
dtype0

'conv3/biases/Adam_1/Read/ReadVariableOpReadVariableOpconv3/biases/Adam_1*
_class
loc:@conv3/biases*
_output_shapes
: *
dtype0

#conv3/alphas/Adam/Initializer/zerosConst*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0*
valueB *    

conv3/alphas/AdamVarHandleOp*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv3/alphas/Adam

2conv3/alphas/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/alphas/Adam*
_class
loc:@conv3/alphas*
_output_shapes
: 
q
conv3/alphas/Adam/AssignAssignVariableOpconv3/alphas/Adam#conv3/alphas/Adam/Initializer/zeros*
dtype0

%conv3/alphas/Adam/Read/ReadVariableOpReadVariableOpconv3/alphas/Adam*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0

%conv3/alphas/Adam_1/Initializer/zerosConst*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0*
valueB *    

conv3/alphas/Adam_1VarHandleOp*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv3/alphas/Adam_1

4conv3/alphas/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv3/alphas/Adam_1*
_class
loc:@conv3/alphas*
_output_shapes
: 
w
conv3/alphas/Adam_1/AssignAssignVariableOpconv3/alphas/Adam_1%conv3/alphas/Adam_1/Initializer/zeros*
dtype0

'conv3/alphas/Adam_1/Read/ReadVariableOpReadVariableOpconv3/alphas/Adam_1*
_class
loc:@conv3/alphas*
_output_shapes
: *
dtype0
Ż
&conv4_1/weights/Adam/Initializer/zerosConst*"
_class
loc:@conv4_1/weights*&
_output_shapes
: *
dtype0*%
valueB *    
°
conv4_1/weights/AdamVarHandleOp*"
_class
loc:@conv4_1/weights*
_output_shapes
: *
dtype0*
shape: *%
shared_nameconv4_1/weights/Adam

5conv4_1/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_1/weights/Adam*"
_class
loc:@conv4_1/weights*
_output_shapes
: 
z
conv4_1/weights/Adam/AssignAssignVariableOpconv4_1/weights/Adam&conv4_1/weights/Adam/Initializer/zeros*
dtype0
Š
(conv4_1/weights/Adam/Read/ReadVariableOpReadVariableOpconv4_1/weights/Adam*"
_class
loc:@conv4_1/weights*&
_output_shapes
: *
dtype0
ą
(conv4_1/weights/Adam_1/Initializer/zerosConst*"
_class
loc:@conv4_1/weights*&
_output_shapes
: *
dtype0*%
valueB *    
´
conv4_1/weights/Adam_1VarHandleOp*"
_class
loc:@conv4_1/weights*
_output_shapes
: *
dtype0*
shape: *'
shared_nameconv4_1/weights/Adam_1
Ą
7conv4_1/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_1/weights/Adam_1*"
_class
loc:@conv4_1/weights*
_output_shapes
: 

conv4_1/weights/Adam_1/AssignAssignVariableOpconv4_1/weights/Adam_1(conv4_1/weights/Adam_1/Initializer/zeros*
dtype0
­
*conv4_1/weights/Adam_1/Read/ReadVariableOpReadVariableOpconv4_1/weights/Adam_1*"
_class
loc:@conv4_1/weights*&
_output_shapes
: *
dtype0

%conv4_1/biases/Adam/Initializer/zerosConst*!
_class
loc:@conv4_1/biases*
_output_shapes
:*
dtype0*
valueB*    
Ą
conv4_1/biases/AdamVarHandleOp*!
_class
loc:@conv4_1/biases*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv4_1/biases/Adam

4conv4_1/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_1/biases/Adam*!
_class
loc:@conv4_1/biases*
_output_shapes
: 
w
conv4_1/biases/Adam/AssignAssignVariableOpconv4_1/biases/Adam%conv4_1/biases/Adam/Initializer/zeros*
dtype0

'conv4_1/biases/Adam/Read/ReadVariableOpReadVariableOpconv4_1/biases/Adam*!
_class
loc:@conv4_1/biases*
_output_shapes
:*
dtype0

'conv4_1/biases/Adam_1/Initializer/zerosConst*!
_class
loc:@conv4_1/biases*
_output_shapes
:*
dtype0*
valueB*    
Ľ
conv4_1/biases/Adam_1VarHandleOp*!
_class
loc:@conv4_1/biases*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv4_1/biases/Adam_1

6conv4_1/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_1/biases/Adam_1*!
_class
loc:@conv4_1/biases*
_output_shapes
: 
}
conv4_1/biases/Adam_1/AssignAssignVariableOpconv4_1/biases/Adam_1'conv4_1/biases/Adam_1/Initializer/zeros*
dtype0

)conv4_1/biases/Adam_1/Read/ReadVariableOpReadVariableOpconv4_1/biases/Adam_1*!
_class
loc:@conv4_1/biases*
_output_shapes
:*
dtype0
Ż
&conv4_2/weights/Adam/Initializer/zerosConst*"
_class
loc:@conv4_2/weights*&
_output_shapes
: *
dtype0*%
valueB *    
°
conv4_2/weights/AdamVarHandleOp*"
_class
loc:@conv4_2/weights*
_output_shapes
: *
dtype0*
shape: *%
shared_nameconv4_2/weights/Adam

5conv4_2/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_2/weights/Adam*"
_class
loc:@conv4_2/weights*
_output_shapes
: 
z
conv4_2/weights/Adam/AssignAssignVariableOpconv4_2/weights/Adam&conv4_2/weights/Adam/Initializer/zeros*
dtype0
Š
(conv4_2/weights/Adam/Read/ReadVariableOpReadVariableOpconv4_2/weights/Adam*"
_class
loc:@conv4_2/weights*&
_output_shapes
: *
dtype0
ą
(conv4_2/weights/Adam_1/Initializer/zerosConst*"
_class
loc:@conv4_2/weights*&
_output_shapes
: *
dtype0*%
valueB *    
´
conv4_2/weights/Adam_1VarHandleOp*"
_class
loc:@conv4_2/weights*
_output_shapes
: *
dtype0*
shape: *'
shared_nameconv4_2/weights/Adam_1
Ą
7conv4_2/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_2/weights/Adam_1*"
_class
loc:@conv4_2/weights*
_output_shapes
: 

conv4_2/weights/Adam_1/AssignAssignVariableOpconv4_2/weights/Adam_1(conv4_2/weights/Adam_1/Initializer/zeros*
dtype0
­
*conv4_2/weights/Adam_1/Read/ReadVariableOpReadVariableOpconv4_2/weights/Adam_1*"
_class
loc:@conv4_2/weights*&
_output_shapes
: *
dtype0

%conv4_2/biases/Adam/Initializer/zerosConst*!
_class
loc:@conv4_2/biases*
_output_shapes
:*
dtype0*
valueB*    
Ą
conv4_2/biases/AdamVarHandleOp*!
_class
loc:@conv4_2/biases*
_output_shapes
: *
dtype0*
shape:*$
shared_nameconv4_2/biases/Adam

4conv4_2/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_2/biases/Adam*!
_class
loc:@conv4_2/biases*
_output_shapes
: 
w
conv4_2/biases/Adam/AssignAssignVariableOpconv4_2/biases/Adam%conv4_2/biases/Adam/Initializer/zeros*
dtype0

'conv4_2/biases/Adam/Read/ReadVariableOpReadVariableOpconv4_2/biases/Adam*!
_class
loc:@conv4_2/biases*
_output_shapes
:*
dtype0

'conv4_2/biases/Adam_1/Initializer/zerosConst*!
_class
loc:@conv4_2/biases*
_output_shapes
:*
dtype0*
valueB*    
Ľ
conv4_2/biases/Adam_1VarHandleOp*!
_class
loc:@conv4_2/biases*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv4_2/biases/Adam_1

6conv4_2/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_2/biases/Adam_1*!
_class
loc:@conv4_2/biases*
_output_shapes
: 
}
conv4_2/biases/Adam_1/AssignAssignVariableOpconv4_2/biases/Adam_1'conv4_2/biases/Adam_1/Initializer/zeros*
dtype0

)conv4_2/biases/Adam_1/Read/ReadVariableOpReadVariableOpconv4_2/biases/Adam_1*!
_class
loc:@conv4_2/biases*
_output_shapes
:*
dtype0
Ż
&conv4_3/weights/Adam/Initializer/zerosConst*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 
*
dtype0*%
valueB 
*    
°
conv4_3/weights/AdamVarHandleOp*"
_class
loc:@conv4_3/weights*
_output_shapes
: *
dtype0*
shape: 
*%
shared_nameconv4_3/weights/Adam

5conv4_3/weights/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_3/weights/Adam*"
_class
loc:@conv4_3/weights*
_output_shapes
: 
z
conv4_3/weights/Adam/AssignAssignVariableOpconv4_3/weights/Adam&conv4_3/weights/Adam/Initializer/zeros*
dtype0
Š
(conv4_3/weights/Adam/Read/ReadVariableOpReadVariableOpconv4_3/weights/Adam*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 
*
dtype0
ą
(conv4_3/weights/Adam_1/Initializer/zerosConst*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 
*
dtype0*%
valueB 
*    
´
conv4_3/weights/Adam_1VarHandleOp*"
_class
loc:@conv4_3/weights*
_output_shapes
: *
dtype0*
shape: 
*'
shared_nameconv4_3/weights/Adam_1
Ą
7conv4_3/weights/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_3/weights/Adam_1*"
_class
loc:@conv4_3/weights*
_output_shapes
: 

conv4_3/weights/Adam_1/AssignAssignVariableOpconv4_3/weights/Adam_1(conv4_3/weights/Adam_1/Initializer/zeros*
dtype0
­
*conv4_3/weights/Adam_1/Read/ReadVariableOpReadVariableOpconv4_3/weights/Adam_1*"
_class
loc:@conv4_3/weights*&
_output_shapes
: 
*
dtype0

%conv4_3/biases/Adam/Initializer/zerosConst*!
_class
loc:@conv4_3/biases*
_output_shapes
:
*
dtype0*
valueB
*    
Ą
conv4_3/biases/AdamVarHandleOp*!
_class
loc:@conv4_3/biases*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameconv4_3/biases/Adam

4conv4_3/biases/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_3/biases/Adam*!
_class
loc:@conv4_3/biases*
_output_shapes
: 
w
conv4_3/biases/Adam/AssignAssignVariableOpconv4_3/biases/Adam%conv4_3/biases/Adam/Initializer/zeros*
dtype0

'conv4_3/biases/Adam/Read/ReadVariableOpReadVariableOpconv4_3/biases/Adam*!
_class
loc:@conv4_3/biases*
_output_shapes
:
*
dtype0

'conv4_3/biases/Adam_1/Initializer/zerosConst*!
_class
loc:@conv4_3/biases*
_output_shapes
:
*
dtype0*
valueB
*    
Ľ
conv4_3/biases/Adam_1VarHandleOp*!
_class
loc:@conv4_3/biases*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameconv4_3/biases/Adam_1

6conv4_3/biases/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv4_3/biases/Adam_1*!
_class
loc:@conv4_3/biases*
_output_shapes
: 
}
conv4_3/biases/Adam_1/AssignAssignVariableOpconv4_3/biases/Adam_1'conv4_3/biases/Adam_1/Initializer/zeros*
dtype0

)conv4_3/biases/Adam_1/Read/ReadVariableOpReadVariableOpconv4_3/biases/Adam_1*!
_class
loc:@conv4_3/biases*
_output_shapes
:
*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *wž?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
~
:Adam/update_conv1/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

<Adam/update_conv1/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
č
+Adam/update_conv1/weights/ResourceApplyAdamResourceApplyAdamconv1/weightsconv1/weights/Adamconv1/weights/Adam_1:Adam/update_conv1/weights/ResourceApplyAdam/ReadVariableOp<Adam/update_conv1/weights/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_9*
T0* 
_class
loc:@conv1/weights
}
9Adam/update_conv1/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_conv1/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

*Adam/update_conv1/biases/ResourceApplyAdamResourceApplyAdamconv1/biasesconv1/biases/Adamconv1/biases/Adam_19Adam/update_conv1/biases/ResourceApplyAdam/ReadVariableOp;Adam/update_conv1/biases/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon7gradients/conv1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv1/biases
}
9Adam/update_conv1/alphas/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_conv1/alphas/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

*Adam/update_conv1/alphas/ResourceApplyAdamResourceApplyAdamconv1/alphasconv1/alphas/Adamconv1/alphas/Adam_19Adam/update_conv1/alphas/ResourceApplyAdam/ReadVariableOp;Adam/update_conv1/alphas/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon1gradients/conv1/mul_grad/tuple/control_dependency*
T0*
_class
loc:@conv1/alphas
~
:Adam/update_conv2/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

<Adam/update_conv2/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
č
+Adam/update_conv2/weights/ResourceApplyAdamResourceApplyAdamconv2/weightsconv2/weights/Adamconv2/weights/Adam_1:Adam/update_conv2/weights/ResourceApplyAdam/ReadVariableOp<Adam/update_conv2/weights/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_7*
T0* 
_class
loc:@conv2/weights
}
9Adam/update_conv2/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_conv2/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

*Adam/update_conv2/biases/ResourceApplyAdamResourceApplyAdamconv2/biasesconv2/biases/Adamconv2/biases/Adam_19Adam/update_conv2/biases/ResourceApplyAdam/ReadVariableOp;Adam/update_conv2/biases/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon7gradients/conv2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv2/biases
}
9Adam/update_conv2/alphas/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_conv2/alphas/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

*Adam/update_conv2/alphas/ResourceApplyAdamResourceApplyAdamconv2/alphasconv2/alphas/Adamconv2/alphas/Adam_19Adam/update_conv2/alphas/ResourceApplyAdam/ReadVariableOp;Adam/update_conv2/alphas/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon1gradients/conv2/mul_grad/tuple/control_dependency*
T0*
_class
loc:@conv2/alphas
~
:Adam/update_conv3/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

<Adam/update_conv3/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
č
+Adam/update_conv3/weights/ResourceApplyAdamResourceApplyAdamconv3/weightsconv3/weights/Adamconv3/weights/Adam_1:Adam/update_conv3/weights/ResourceApplyAdam/ReadVariableOp<Adam/update_conv3/weights/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
T0* 
_class
loc:@conv3/weights
}
9Adam/update_conv3/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_conv3/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

*Adam/update_conv3/biases/ResourceApplyAdamResourceApplyAdamconv3/biasesconv3/biases/Adamconv3/biases/Adam_19Adam/update_conv3/biases/ResourceApplyAdam/ReadVariableOp;Adam/update_conv3/biases/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon7gradients/conv3/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@conv3/biases
}
9Adam/update_conv3/alphas/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

;Adam/update_conv3/alphas/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

*Adam/update_conv3/alphas/ResourceApplyAdamResourceApplyAdamconv3/alphasconv3/alphas/Adamconv3/alphas/Adam_19Adam/update_conv3/alphas/ResourceApplyAdam/ReadVariableOp;Adam/update_conv3/alphas/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon1gradients/conv3/mul_grad/tuple/control_dependency*
T0*
_class
loc:@conv3/alphas

<Adam/update_conv4_1/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

>Adam/update_conv4_1/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ö
-Adam/update_conv4_1/weights/ResourceApplyAdamResourceApplyAdamconv4_1/weightsconv4_1/weights/Adamconv4_1/weights/Adam_1<Adam/update_conv4_1/weights/ResourceApplyAdam/ReadVariableOp>Adam/update_conv4_1/weights/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
T0*"
_class
loc:@conv4_1/weights

;Adam/update_conv4_1/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

=Adam/update_conv4_1/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

,Adam/update_conv4_1/biases/ResourceApplyAdamResourceApplyAdamconv4_1/biasesconv4_1/biases/Adamconv4_1/biases/Adam_1;Adam/update_conv4_1/biases/ResourceApplyAdam/ReadVariableOp=Adam/update_conv4_1/biases/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv4_1/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@conv4_1/biases

<Adam/update_conv4_2/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

>Adam/update_conv4_2/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ö
-Adam/update_conv4_2/weights/ResourceApplyAdamResourceApplyAdamconv4_2/weightsconv4_2/weights/Adamconv4_2/weights/Adam_1<Adam/update_conv4_2/weights/ResourceApplyAdam/ReadVariableOp>Adam/update_conv4_2/weights/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
T0*"
_class
loc:@conv4_2/weights

;Adam/update_conv4_2/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

=Adam/update_conv4_2/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

,Adam/update_conv4_2/biases/ResourceApplyAdamResourceApplyAdamconv4_2/biasesconv4_2/biases/Adamconv4_2/biases/Adam_1;Adam/update_conv4_2/biases/ResourceApplyAdam/ReadVariableOp=Adam/update_conv4_2/biases/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv4_2/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@conv4_2/biases

<Adam/update_conv4_3/weights/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

>Adam/update_conv4_3/weights/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0
ô
-Adam/update_conv4_3/weights/ResourceApplyAdamResourceApplyAdamconv4_3/weightsconv4_3/weights/Adamconv4_3/weights/Adam_1<Adam/update_conv4_3/weights/ResourceApplyAdam/ReadVariableOp>Adam/update_conv4_3/weights/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN*
T0*"
_class
loc:@conv4_3/weights

;Adam/update_conv4_3/biases/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

=Adam/update_conv4_3/biases/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
_output_shapes
: *
dtype0

,Adam/update_conv4_3/biases/ResourceApplyAdamResourceApplyAdamconv4_3/biasesconv4_3/biases/Adamconv4_3/biases/Adam_1;Adam/update_conv4_3/biases/ResourceApplyAdam/ReadVariableOp=Adam/update_conv4_3/biases/ResourceApplyAdam/ReadVariableOp_1ExponentialDecay
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv4_3/BiasAdd_grad/tuple/control_dependency_1*
T0*!
_class
loc:@conv4_3/biases

Adam/ReadVariableOpReadVariableOpbeta1_power+^Adam/update_conv1/alphas/ResourceApplyAdam+^Adam/update_conv1/biases/ResourceApplyAdam,^Adam/update_conv1/weights/ResourceApplyAdam+^Adam/update_conv2/alphas/ResourceApplyAdam+^Adam/update_conv2/biases/ResourceApplyAdam,^Adam/update_conv2/weights/ResourceApplyAdam+^Adam/update_conv3/alphas/ResourceApplyAdam+^Adam/update_conv3/biases/ResourceApplyAdam,^Adam/update_conv3/weights/ResourceApplyAdam-^Adam/update_conv4_1/biases/ResourceApplyAdam.^Adam/update_conv4_1/weights/ResourceApplyAdam-^Adam/update_conv4_2/biases/ResourceApplyAdam.^Adam/update_conv4_2/weights/ResourceApplyAdam-^Adam/update_conv4_3/biases/ResourceApplyAdam.^Adam/update_conv4_3/weights/ResourceApplyAdam*
_output_shapes
: *
dtype0
r
Adam/mulMulAdam/ReadVariableOp
Adam/beta1*
T0*
_class
loc:@conv1/alphas*
_output_shapes
: 

Adam/AssignVariableOpAssignVariableOpbeta1_powerAdam/mul*
_class
loc:@conv1/alphas*
dtype0*
validate_shape(
Ç
Adam/ReadVariableOp_1ReadVariableOpbeta1_power^Adam/AssignVariableOp+^Adam/update_conv1/alphas/ResourceApplyAdam+^Adam/update_conv1/biases/ResourceApplyAdam,^Adam/update_conv1/weights/ResourceApplyAdam+^Adam/update_conv2/alphas/ResourceApplyAdam+^Adam/update_conv2/biases/ResourceApplyAdam,^Adam/update_conv2/weights/ResourceApplyAdam+^Adam/update_conv3/alphas/ResourceApplyAdam+^Adam/update_conv3/biases/ResourceApplyAdam,^Adam/update_conv3/weights/ResourceApplyAdam-^Adam/update_conv4_1/biases/ResourceApplyAdam.^Adam/update_conv4_1/weights/ResourceApplyAdam-^Adam/update_conv4_2/biases/ResourceApplyAdam.^Adam/update_conv4_2/weights/ResourceApplyAdam-^Adam/update_conv4_3/biases/ResourceApplyAdam.^Adam/update_conv4_3/weights/ResourceApplyAdam*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0

Adam/ReadVariableOp_2ReadVariableOpbeta2_power+^Adam/update_conv1/alphas/ResourceApplyAdam+^Adam/update_conv1/biases/ResourceApplyAdam,^Adam/update_conv1/weights/ResourceApplyAdam+^Adam/update_conv2/alphas/ResourceApplyAdam+^Adam/update_conv2/biases/ResourceApplyAdam,^Adam/update_conv2/weights/ResourceApplyAdam+^Adam/update_conv3/alphas/ResourceApplyAdam+^Adam/update_conv3/biases/ResourceApplyAdam,^Adam/update_conv3/weights/ResourceApplyAdam-^Adam/update_conv4_1/biases/ResourceApplyAdam.^Adam/update_conv4_1/weights/ResourceApplyAdam-^Adam/update_conv4_2/biases/ResourceApplyAdam.^Adam/update_conv4_2/weights/ResourceApplyAdam-^Adam/update_conv4_3/biases/ResourceApplyAdam.^Adam/update_conv4_3/weights/ResourceApplyAdam*
_output_shapes
: *
dtype0
v

Adam/mul_1MulAdam/ReadVariableOp_2
Adam/beta2*
T0*
_class
loc:@conv1/alphas*
_output_shapes
: 

Adam/AssignVariableOp_1AssignVariableOpbeta2_power
Adam/mul_1*
_class
loc:@conv1/alphas*
dtype0*
validate_shape(
É
Adam/ReadVariableOp_3ReadVariableOpbeta2_power^Adam/AssignVariableOp_1+^Adam/update_conv1/alphas/ResourceApplyAdam+^Adam/update_conv1/biases/ResourceApplyAdam,^Adam/update_conv1/weights/ResourceApplyAdam+^Adam/update_conv2/alphas/ResourceApplyAdam+^Adam/update_conv2/biases/ResourceApplyAdam,^Adam/update_conv2/weights/ResourceApplyAdam+^Adam/update_conv3/alphas/ResourceApplyAdam+^Adam/update_conv3/biases/ResourceApplyAdam,^Adam/update_conv3/weights/ResourceApplyAdam-^Adam/update_conv4_1/biases/ResourceApplyAdam.^Adam/update_conv4_1/weights/ResourceApplyAdam-^Adam/update_conv4_2/biases/ResourceApplyAdam.^Adam/update_conv4_2/weights/ResourceApplyAdam-^Adam/update_conv4_3/biases/ResourceApplyAdam.^Adam/update_conv4_3/weights/ResourceApplyAdam*
_class
loc:@conv1/alphas*
_output_shapes
: *
dtype0
ú
Adam/updateNoOp^Adam/AssignVariableOp^Adam/AssignVariableOp_1+^Adam/update_conv1/alphas/ResourceApplyAdam+^Adam/update_conv1/biases/ResourceApplyAdam,^Adam/update_conv1/weights/ResourceApplyAdam+^Adam/update_conv2/alphas/ResourceApplyAdam+^Adam/update_conv2/biases/ResourceApplyAdam,^Adam/update_conv2/weights/ResourceApplyAdam+^Adam/update_conv3/alphas/ResourceApplyAdam+^Adam/update_conv3/biases/ResourceApplyAdam,^Adam/update_conv3/weights/ResourceApplyAdam-^Adam/update_conv4_1/biases/ResourceApplyAdam.^Adam/update_conv4_1/weights/ResourceApplyAdam-^Adam/update_conv4_2/biases/ResourceApplyAdam.^Adam/update_conv4_2/weights/ResourceApplyAdam-^Adam/update_conv4_3/biases/ResourceApplyAdam.^Adam/update_conv4_3/weights/ResourceApplyAdam
w

Adam/ConstConst^Adam/update*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
value	B :
[
AdamAssignAddVariableOpVariable
Adam/Const*
_class
loc:@Variable*
dtype0


initNoOp^Variable/Assign^beta1_power/Assign^beta2_power/Assign^conv1/alphas/Adam/Assign^conv1/alphas/Adam_1/Assign^conv1/alphas/Assign^conv1/biases/Adam/Assign^conv1/biases/Adam_1/Assign^conv1/biases/Assign^conv1/weights/Adam/Assign^conv1/weights/Adam_1/Assign^conv1/weights/Assign^conv2/alphas/Adam/Assign^conv2/alphas/Adam_1/Assign^conv2/alphas/Assign^conv2/biases/Adam/Assign^conv2/biases/Adam_1/Assign^conv2/biases/Assign^conv2/weights/Adam/Assign^conv2/weights/Adam_1/Assign^conv2/weights/Assign^conv3/alphas/Adam/Assign^conv3/alphas/Adam_1/Assign^conv3/alphas/Assign^conv3/biases/Adam/Assign^conv3/biases/Adam_1/Assign^conv3/biases/Assign^conv3/weights/Adam/Assign^conv3/weights/Adam_1/Assign^conv3/weights/Assign^conv4_1/biases/Adam/Assign^conv4_1/biases/Adam_1/Assign^conv4_1/biases/Assign^conv4_1/weights/Adam/Assign^conv4_1/weights/Adam_1/Assign^conv4_1/weights/Assign^conv4_2/biases/Adam/Assign^conv4_2/biases/Adam_1/Assign^conv4_2/biases/Assign^conv4_2/weights/Adam/Assign^conv4_2/weights/Adam_1/Assign^conv4_2/weights/Assign^conv4_3/biases/Adam/Assign^conv4_3/biases/Adam_1/Assign^conv4_3/biases/Assign^conv4_3/weights/Adam/Assign^conv4_3/weights/Adam_1/Assign^conv4_3/weights/Assign
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
ć
save/SaveV2/tensor_namesConst*
_output_shapes
:0*
dtype0*
valueB0BVariableBbeta1_powerBbeta2_powerBconv1/alphasBconv1/alphas/AdamBconv1/alphas/Adam_1Bconv1/biasesBconv1/biases/AdamBconv1/biases/Adam_1Bconv1/weightsBconv1/weights/AdamBconv1/weights/Adam_1Bconv2/alphasBconv2/alphas/AdamBconv2/alphas/Adam_1Bconv2/biasesBconv2/biases/AdamBconv2/biases/Adam_1Bconv2/weightsBconv2/weights/AdamBconv2/weights/Adam_1Bconv3/alphasBconv3/alphas/AdamBconv3/alphas/Adam_1Bconv3/biasesBconv3/biases/AdamBconv3/biases/Adam_1Bconv3/weightsBconv3/weights/AdamBconv3/weights/Adam_1Bconv4_1/biasesBconv4_1/biases/AdamBconv4_1/biases/Adam_1Bconv4_1/weightsBconv4_1/weights/AdamBconv4_1/weights/Adam_1Bconv4_2/biasesBconv4_2/biases/AdamBconv4_2/biases/Adam_1Bconv4_2/weightsBconv4_2/weights/AdamBconv4_2/weights/Adam_1Bconv4_3/biasesBconv4_3/biases/AdamBconv4_3/biases/Adam_1Bconv4_3/weightsBconv4_3/weights/AdamBconv4_3/weights/Adam_1
Ă
save/SaveV2/shape_and_slicesConst*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Ý
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable/Read/ReadVariableOpbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp conv1/alphas/Read/ReadVariableOp%conv1/alphas/Adam/Read/ReadVariableOp'conv1/alphas/Adam_1/Read/ReadVariableOp conv1/biases/Read/ReadVariableOp%conv1/biases/Adam/Read/ReadVariableOp'conv1/biases/Adam_1/Read/ReadVariableOp!conv1/weights/Read/ReadVariableOp&conv1/weights/Adam/Read/ReadVariableOp(conv1/weights/Adam_1/Read/ReadVariableOp conv2/alphas/Read/ReadVariableOp%conv2/alphas/Adam/Read/ReadVariableOp'conv2/alphas/Adam_1/Read/ReadVariableOp conv2/biases/Read/ReadVariableOp%conv2/biases/Adam/Read/ReadVariableOp'conv2/biases/Adam_1/Read/ReadVariableOp!conv2/weights/Read/ReadVariableOp&conv2/weights/Adam/Read/ReadVariableOp(conv2/weights/Adam_1/Read/ReadVariableOp conv3/alphas/Read/ReadVariableOp%conv3/alphas/Adam/Read/ReadVariableOp'conv3/alphas/Adam_1/Read/ReadVariableOp conv3/biases/Read/ReadVariableOp%conv3/biases/Adam/Read/ReadVariableOp'conv3/biases/Adam_1/Read/ReadVariableOp!conv3/weights/Read/ReadVariableOp&conv3/weights/Adam/Read/ReadVariableOp(conv3/weights/Adam_1/Read/ReadVariableOp"conv4_1/biases/Read/ReadVariableOp'conv4_1/biases/Adam/Read/ReadVariableOp)conv4_1/biases/Adam_1/Read/ReadVariableOp#conv4_1/weights/Read/ReadVariableOp(conv4_1/weights/Adam/Read/ReadVariableOp*conv4_1/weights/Adam_1/Read/ReadVariableOp"conv4_2/biases/Read/ReadVariableOp'conv4_2/biases/Adam/Read/ReadVariableOp)conv4_2/biases/Adam_1/Read/ReadVariableOp#conv4_2/weights/Read/ReadVariableOp(conv4_2/weights/Adam/Read/ReadVariableOp*conv4_2/weights/Adam_1/Read/ReadVariableOp"conv4_3/biases/Read/ReadVariableOp'conv4_3/biases/Adam/Read/ReadVariableOp)conv4_3/biases/Adam_1/Read/ReadVariableOp#conv4_3/weights/Read/ReadVariableOp(conv4_3/weights/Adam/Read/ReadVariableOp*conv4_3/weights/Adam_1/Read/ReadVariableOp*>
dtypes4
220
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
ř
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0BVariableBbeta1_powerBbeta2_powerBconv1/alphasBconv1/alphas/AdamBconv1/alphas/Adam_1Bconv1/biasesBconv1/biases/AdamBconv1/biases/Adam_1Bconv1/weightsBconv1/weights/AdamBconv1/weights/Adam_1Bconv2/alphasBconv2/alphas/AdamBconv2/alphas/Adam_1Bconv2/biasesBconv2/biases/AdamBconv2/biases/Adam_1Bconv2/weightsBconv2/weights/AdamBconv2/weights/Adam_1Bconv3/alphasBconv3/alphas/AdamBconv3/alphas/Adam_1Bconv3/biasesBconv3/biases/AdamBconv3/biases/Adam_1Bconv3/weightsBconv3/weights/AdamBconv3/weights/Adam_1Bconv4_1/biasesBconv4_1/biases/AdamBconv4_1/biases/Adam_1Bconv4_1/weightsBconv4_1/weights/AdamBconv4_1/weights/Adam_1Bconv4_2/biasesBconv4_2/biases/AdamBconv4_2/biases/Adam_1Bconv4_2/weightsBconv4_2/weights/AdamBconv4_2/weights/Adam_1Bconv4_3/biasesBconv4_3/biases/AdamBconv4_3/biases/Adam_1Bconv4_3/weightsBconv4_3/weights/AdamBconv4_3/weights/Adam_1
Ő
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*Ö
_output_shapesĂ
Ŕ::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
O
save/AssignVariableOpAssignVariableOpVariablesave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
V
save/AssignVariableOp_1AssignVariableOpbeta1_powersave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
V
save/AssignVariableOp_2AssignVariableOpbeta2_powersave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
W
save/AssignVariableOp_3AssignVariableOpconv1/alphassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0*
_output_shapes
:
\
save/AssignVariableOp_4AssignVariableOpconv1/alphas/Adamsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0*
_output_shapes
:
^
save/AssignVariableOp_5AssignVariableOpconv1/alphas/Adam_1save/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:6*
T0*
_output_shapes
:
W
save/AssignVariableOp_6AssignVariableOpconv1/biasessave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:7*
T0*
_output_shapes
:
\
save/AssignVariableOp_7AssignVariableOpconv1/biases/Adamsave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:8*
T0*
_output_shapes
:
^
save/AssignVariableOp_8AssignVariableOpconv1/biases/Adam_1save/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:9*
T0*
_output_shapes
:
X
save/AssignVariableOp_9AssignVariableOpconv1/weightssave/Identity_9*
dtype0
R
save/Identity_10Identitysave/RestoreV2:10*
T0*
_output_shapes
:
_
save/AssignVariableOp_10AssignVariableOpconv1/weights/Adamsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:11*
T0*
_output_shapes
:
a
save/AssignVariableOp_11AssignVariableOpconv1/weights/Adam_1save/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:12*
T0*
_output_shapes
:
Y
save/AssignVariableOp_12AssignVariableOpconv2/alphassave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:13*
T0*
_output_shapes
:
^
save/AssignVariableOp_13AssignVariableOpconv2/alphas/Adamsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:14*
T0*
_output_shapes
:
`
save/AssignVariableOp_14AssignVariableOpconv2/alphas/Adam_1save/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:15*
T0*
_output_shapes
:
Y
save/AssignVariableOp_15AssignVariableOpconv2/biasessave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:16*
T0*
_output_shapes
:
^
save/AssignVariableOp_16AssignVariableOpconv2/biases/Adamsave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:17*
T0*
_output_shapes
:
`
save/AssignVariableOp_17AssignVariableOpconv2/biases/Adam_1save/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:18*
T0*
_output_shapes
:
Z
save/AssignVariableOp_18AssignVariableOpconv2/weightssave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:19*
T0*
_output_shapes
:
_
save/AssignVariableOp_19AssignVariableOpconv2/weights/Adamsave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:20*
T0*
_output_shapes
:
a
save/AssignVariableOp_20AssignVariableOpconv2/weights/Adam_1save/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:21*
T0*
_output_shapes
:
Y
save/AssignVariableOp_21AssignVariableOpconv3/alphassave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:22*
T0*
_output_shapes
:
^
save/AssignVariableOp_22AssignVariableOpconv3/alphas/Adamsave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:23*
T0*
_output_shapes
:
`
save/AssignVariableOp_23AssignVariableOpconv3/alphas/Adam_1save/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:24*
T0*
_output_shapes
:
Y
save/AssignVariableOp_24AssignVariableOpconv3/biasessave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:25*
T0*
_output_shapes
:
^
save/AssignVariableOp_25AssignVariableOpconv3/biases/Adamsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:26*
T0*
_output_shapes
:
`
save/AssignVariableOp_26AssignVariableOpconv3/biases/Adam_1save/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:27*
T0*
_output_shapes
:
Z
save/AssignVariableOp_27AssignVariableOpconv3/weightssave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:28*
T0*
_output_shapes
:
_
save/AssignVariableOp_28AssignVariableOpconv3/weights/Adamsave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:29*
T0*
_output_shapes
:
a
save/AssignVariableOp_29AssignVariableOpconv3/weights/Adam_1save/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:30*
T0*
_output_shapes
:
[
save/AssignVariableOp_30AssignVariableOpconv4_1/biasessave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:31*
T0*
_output_shapes
:
`
save/AssignVariableOp_31AssignVariableOpconv4_1/biases/Adamsave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:32*
T0*
_output_shapes
:
b
save/AssignVariableOp_32AssignVariableOpconv4_1/biases/Adam_1save/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:33*
T0*
_output_shapes
:
\
save/AssignVariableOp_33AssignVariableOpconv4_1/weightssave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:34*
T0*
_output_shapes
:
a
save/AssignVariableOp_34AssignVariableOpconv4_1/weights/Adamsave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:35*
T0*
_output_shapes
:
c
save/AssignVariableOp_35AssignVariableOpconv4_1/weights/Adam_1save/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:36*
T0*
_output_shapes
:
[
save/AssignVariableOp_36AssignVariableOpconv4_2/biasessave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:37*
T0*
_output_shapes
:
`
save/AssignVariableOp_37AssignVariableOpconv4_2/biases/Adamsave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:38*
T0*
_output_shapes
:
b
save/AssignVariableOp_38AssignVariableOpconv4_2/biases/Adam_1save/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:39*
T0*
_output_shapes
:
\
save/AssignVariableOp_39AssignVariableOpconv4_2/weightssave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:40*
T0*
_output_shapes
:
a
save/AssignVariableOp_40AssignVariableOpconv4_2/weights/Adamsave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:41*
T0*
_output_shapes
:
c
save/AssignVariableOp_41AssignVariableOpconv4_2/weights/Adam_1save/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:42*
T0*
_output_shapes
:
[
save/AssignVariableOp_42AssignVariableOpconv4_3/biasessave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:43*
T0*
_output_shapes
:
`
save/AssignVariableOp_43AssignVariableOpconv4_3/biases/Adamsave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:44*
T0*
_output_shapes
:
b
save/AssignVariableOp_44AssignVariableOpconv4_3/biases/Adam_1save/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:45*
T0*
_output_shapes
:
\
save/AssignVariableOp_45AssignVariableOpconv4_3/weightssave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:46*
T0*
_output_shapes
:
a
save/AssignVariableOp_46AssignVariableOpconv4_3/weights/Adamsave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:47*
T0*
_output_shapes
:
c
save/AssignVariableOp_47AssignVariableOpconv4_3/weights/Adam_1save/Identity_47*
dtype0


save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
X
bbox_loss/tagsConst*
_output_shapes
: *
dtype0*
valueB B	bbox_loss
S
	bbox_lossScalarSummarybbox_loss/tagsMean_1*
T0*
_output_shapes
: 
`
landmark_loss/tagsConst*
_output_shapes
: *
dtype0*
valueB Blandmark_loss
[
landmark_lossScalarSummarylandmark_loss/tagsMean_2*
T0*
_output_shapes
: 
^
cls_accuracy/tagsConst*
_output_shapes
: *
dtype0*
valueB Bcls_accuracy
Y
cls_accuracyScalarSummarycls_accuracy/tagsMean_3*
T0*
_output_shapes
: 
Ť
Merge/MergeSummaryMergeSummary"input_producer/fraction_of_32_fullbatch/fraction_of_384_full	bbox_losslandmark_losscls_accuracy*
N*
_output_shapes
: 
[
save/filename_1/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save/filename_1PlaceholderWithDefaultsave/filename_1/input*
_output_shapes
: *
dtype0*
shape: 
i
save/Const_1PlaceholderWithDefaultsave/filename_1*
_output_shapes
: *
dtype0*
shape: 
}
save/StaticRegexFullMatchStaticRegexFullMatchsave/Const_1"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_3Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp\part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_2save/Const_3"/device:CPU:**
T0*
_output_shapes
: 
h
save/StringJoin
StringJoinsave/Const_1save/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
÷
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0BVariableBbeta1_powerBbeta2_powerBconv1/alphasBconv1/alphas/AdamBconv1/alphas/Adam_1Bconv1/biasesBconv1/biases/AdamBconv1/biases/Adam_1Bconv1/weightsBconv1/weights/AdamBconv1/weights/Adam_1Bconv2/alphasBconv2/alphas/AdamBconv2/alphas/Adam_1Bconv2/biasesBconv2/biases/AdamBconv2/biases/Adam_1Bconv2/weightsBconv2/weights/AdamBconv2/weights/Adam_1Bconv3/alphasBconv3/alphas/AdamBconv3/alphas/Adam_1Bconv3/biasesBconv3/biases/AdamBconv3/biases/Adam_1Bconv3/weightsBconv3/weights/AdamBconv3/weights/Adam_1Bconv4_1/biasesBconv4_1/biases/AdamBconv4_1/biases/Adam_1Bconv4_1/weightsBconv4_1/weights/AdamBconv4_1/weights/Adam_1Bconv4_2/biasesBconv4_2/biases/AdamBconv4_2/biases/Adam_1Bconv4_2/weightsBconv4_2/weights/AdamBconv4_2/weights/Adam_1Bconv4_3/biasesBconv4_3/biases/AdamBconv4_3/biases/Adam_1Bconv4_3/weightsBconv4_3/weights/AdamBconv4_3/weights/Adam_1
Ô
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü
save/SaveV2_1SaveV2save/ShardedFilenamesave/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicesVariable/Read/ReadVariableOpbeta1_power/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp conv1/alphas/Read/ReadVariableOp%conv1/alphas/Adam/Read/ReadVariableOp'conv1/alphas/Adam_1/Read/ReadVariableOp conv1/biases/Read/ReadVariableOp%conv1/biases/Adam/Read/ReadVariableOp'conv1/biases/Adam_1/Read/ReadVariableOp!conv1/weights/Read/ReadVariableOp&conv1/weights/Adam/Read/ReadVariableOp(conv1/weights/Adam_1/Read/ReadVariableOp conv2/alphas/Read/ReadVariableOp%conv2/alphas/Adam/Read/ReadVariableOp'conv2/alphas/Adam_1/Read/ReadVariableOp conv2/biases/Read/ReadVariableOp%conv2/biases/Adam/Read/ReadVariableOp'conv2/biases/Adam_1/Read/ReadVariableOp!conv2/weights/Read/ReadVariableOp&conv2/weights/Adam/Read/ReadVariableOp(conv2/weights/Adam_1/Read/ReadVariableOp conv3/alphas/Read/ReadVariableOp%conv3/alphas/Adam/Read/ReadVariableOp'conv3/alphas/Adam_1/Read/ReadVariableOp conv3/biases/Read/ReadVariableOp%conv3/biases/Adam/Read/ReadVariableOp'conv3/biases/Adam_1/Read/ReadVariableOp!conv3/weights/Read/ReadVariableOp&conv3/weights/Adam/Read/ReadVariableOp(conv3/weights/Adam_1/Read/ReadVariableOp"conv4_1/biases/Read/ReadVariableOp'conv4_1/biases/Adam/Read/ReadVariableOp)conv4_1/biases/Adam_1/Read/ReadVariableOp#conv4_1/weights/Read/ReadVariableOp(conv4_1/weights/Adam/Read/ReadVariableOp*conv4_1/weights/Adam_1/Read/ReadVariableOp"conv4_2/biases/Read/ReadVariableOp'conv4_2/biases/Adam/Read/ReadVariableOp)conv4_2/biases/Adam_1/Read/ReadVariableOp#conv4_2/weights/Read/ReadVariableOp(conv4_2/weights/Adam/Read/ReadVariableOp*conv4_2/weights/Adam_1/Read/ReadVariableOp"conv4_3/biases/Read/ReadVariableOp'conv4_3/biases/Adam/Read/ReadVariableOp)conv4_3/biases/Adam_1/Read/ReadVariableOp#conv4_3/weights/Read/ReadVariableOp(conv4_3/weights/Adam/Read/ReadVariableOp*conv4_3/weights/Adam_1/Read/ReadVariableOp"/device:CPU:0*>
dtypes4
220
¤
save/control_dependency_1Identitysave/ShardedFilename^save/SaveV2_1"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
˘
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency_1"/device:CPU:0*
N*
T0*
_output_shapes
:
w
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixessave/Const_1"/device:CPU:0

save/Identity_48Identitysave/Const_1^save/MergeV2Checkpoints^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
ú
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0BVariableBbeta1_powerBbeta2_powerBconv1/alphasBconv1/alphas/AdamBconv1/alphas/Adam_1Bconv1/biasesBconv1/biases/AdamBconv1/biases/Adam_1Bconv1/weightsBconv1/weights/AdamBconv1/weights/Adam_1Bconv2/alphasBconv2/alphas/AdamBconv2/alphas/Adam_1Bconv2/biasesBconv2/biases/AdamBconv2/biases/Adam_1Bconv2/weightsBconv2/weights/AdamBconv2/weights/Adam_1Bconv3/alphasBconv3/alphas/AdamBconv3/alphas/Adam_1Bconv3/biasesBconv3/biases/AdamBconv3/biases/Adam_1Bconv3/weightsBconv3/weights/AdamBconv3/weights/Adam_1Bconv4_1/biasesBconv4_1/biases/AdamBconv4_1/biases/Adam_1Bconv4_1/weightsBconv4_1/weights/AdamBconv4_1/weights/Adam_1Bconv4_2/biasesBconv4_2/biases/AdamBconv4_2/biases/Adam_1Bconv4_2/weightsBconv4_2/weights/AdamBconv4_2/weights/Adam_1Bconv4_3/biasesBconv4_3/biases/AdamBconv4_3/biases/Adam_1Bconv4_3/weightsBconv4_3/weights/AdamBconv4_3/weights/Adam_1
×
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save/RestoreV2_1	RestoreV2save/Const_1save/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*Ö
_output_shapesĂ
Ŕ::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220
Q
save/Identity_49Identitysave/RestoreV2_1*
T0*
_output_shapes
:
U
save/AssignVariableOp_48AssignVariableOpVariablesave/Identity_49*
dtype0
S
save/Identity_50Identitysave/RestoreV2_1:1*
T0*
_output_shapes
:
X
save/AssignVariableOp_49AssignVariableOpbeta1_powersave/Identity_50*
dtype0
S
save/Identity_51Identitysave/RestoreV2_1:2*
T0*
_output_shapes
:
X
save/AssignVariableOp_50AssignVariableOpbeta2_powersave/Identity_51*
dtype0
S
save/Identity_52Identitysave/RestoreV2_1:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_51AssignVariableOpconv1/alphassave/Identity_52*
dtype0
S
save/Identity_53Identitysave/RestoreV2_1:4*
T0*
_output_shapes
:
^
save/AssignVariableOp_52AssignVariableOpconv1/alphas/Adamsave/Identity_53*
dtype0
S
save/Identity_54Identitysave/RestoreV2_1:5*
T0*
_output_shapes
:
`
save/AssignVariableOp_53AssignVariableOpconv1/alphas/Adam_1save/Identity_54*
dtype0
S
save/Identity_55Identitysave/RestoreV2_1:6*
T0*
_output_shapes
:
Y
save/AssignVariableOp_54AssignVariableOpconv1/biasessave/Identity_55*
dtype0
S
save/Identity_56Identitysave/RestoreV2_1:7*
T0*
_output_shapes
:
^
save/AssignVariableOp_55AssignVariableOpconv1/biases/Adamsave/Identity_56*
dtype0
S
save/Identity_57Identitysave/RestoreV2_1:8*
T0*
_output_shapes
:
`
save/AssignVariableOp_56AssignVariableOpconv1/biases/Adam_1save/Identity_57*
dtype0
S
save/Identity_58Identitysave/RestoreV2_1:9*
T0*
_output_shapes
:
Z
save/AssignVariableOp_57AssignVariableOpconv1/weightssave/Identity_58*
dtype0
T
save/Identity_59Identitysave/RestoreV2_1:10*
T0*
_output_shapes
:
_
save/AssignVariableOp_58AssignVariableOpconv1/weights/Adamsave/Identity_59*
dtype0
T
save/Identity_60Identitysave/RestoreV2_1:11*
T0*
_output_shapes
:
a
save/AssignVariableOp_59AssignVariableOpconv1/weights/Adam_1save/Identity_60*
dtype0
T
save/Identity_61Identitysave/RestoreV2_1:12*
T0*
_output_shapes
:
Y
save/AssignVariableOp_60AssignVariableOpconv2/alphassave/Identity_61*
dtype0
T
save/Identity_62Identitysave/RestoreV2_1:13*
T0*
_output_shapes
:
^
save/AssignVariableOp_61AssignVariableOpconv2/alphas/Adamsave/Identity_62*
dtype0
T
save/Identity_63Identitysave/RestoreV2_1:14*
T0*
_output_shapes
:
`
save/AssignVariableOp_62AssignVariableOpconv2/alphas/Adam_1save/Identity_63*
dtype0
T
save/Identity_64Identitysave/RestoreV2_1:15*
T0*
_output_shapes
:
Y
save/AssignVariableOp_63AssignVariableOpconv2/biasessave/Identity_64*
dtype0
T
save/Identity_65Identitysave/RestoreV2_1:16*
T0*
_output_shapes
:
^
save/AssignVariableOp_64AssignVariableOpconv2/biases/Adamsave/Identity_65*
dtype0
T
save/Identity_66Identitysave/RestoreV2_1:17*
T0*
_output_shapes
:
`
save/AssignVariableOp_65AssignVariableOpconv2/biases/Adam_1save/Identity_66*
dtype0
T
save/Identity_67Identitysave/RestoreV2_1:18*
T0*
_output_shapes
:
Z
save/AssignVariableOp_66AssignVariableOpconv2/weightssave/Identity_67*
dtype0
T
save/Identity_68Identitysave/RestoreV2_1:19*
T0*
_output_shapes
:
_
save/AssignVariableOp_67AssignVariableOpconv2/weights/Adamsave/Identity_68*
dtype0
T
save/Identity_69Identitysave/RestoreV2_1:20*
T0*
_output_shapes
:
a
save/AssignVariableOp_68AssignVariableOpconv2/weights/Adam_1save/Identity_69*
dtype0
T
save/Identity_70Identitysave/RestoreV2_1:21*
T0*
_output_shapes
:
Y
save/AssignVariableOp_69AssignVariableOpconv3/alphassave/Identity_70*
dtype0
T
save/Identity_71Identitysave/RestoreV2_1:22*
T0*
_output_shapes
:
^
save/AssignVariableOp_70AssignVariableOpconv3/alphas/Adamsave/Identity_71*
dtype0
T
save/Identity_72Identitysave/RestoreV2_1:23*
T0*
_output_shapes
:
`
save/AssignVariableOp_71AssignVariableOpconv3/alphas/Adam_1save/Identity_72*
dtype0
T
save/Identity_73Identitysave/RestoreV2_1:24*
T0*
_output_shapes
:
Y
save/AssignVariableOp_72AssignVariableOpconv3/biasessave/Identity_73*
dtype0
T
save/Identity_74Identitysave/RestoreV2_1:25*
T0*
_output_shapes
:
^
save/AssignVariableOp_73AssignVariableOpconv3/biases/Adamsave/Identity_74*
dtype0
T
save/Identity_75Identitysave/RestoreV2_1:26*
T0*
_output_shapes
:
`
save/AssignVariableOp_74AssignVariableOpconv3/biases/Adam_1save/Identity_75*
dtype0
T
save/Identity_76Identitysave/RestoreV2_1:27*
T0*
_output_shapes
:
Z
save/AssignVariableOp_75AssignVariableOpconv3/weightssave/Identity_76*
dtype0
T
save/Identity_77Identitysave/RestoreV2_1:28*
T0*
_output_shapes
:
_
save/AssignVariableOp_76AssignVariableOpconv3/weights/Adamsave/Identity_77*
dtype0
T
save/Identity_78Identitysave/RestoreV2_1:29*
T0*
_output_shapes
:
a
save/AssignVariableOp_77AssignVariableOpconv3/weights/Adam_1save/Identity_78*
dtype0
T
save/Identity_79Identitysave/RestoreV2_1:30*
T0*
_output_shapes
:
[
save/AssignVariableOp_78AssignVariableOpconv4_1/biasessave/Identity_79*
dtype0
T
save/Identity_80Identitysave/RestoreV2_1:31*
T0*
_output_shapes
:
`
save/AssignVariableOp_79AssignVariableOpconv4_1/biases/Adamsave/Identity_80*
dtype0
T
save/Identity_81Identitysave/RestoreV2_1:32*
T0*
_output_shapes
:
b
save/AssignVariableOp_80AssignVariableOpconv4_1/biases/Adam_1save/Identity_81*
dtype0
T
save/Identity_82Identitysave/RestoreV2_1:33*
T0*
_output_shapes
:
\
save/AssignVariableOp_81AssignVariableOpconv4_1/weightssave/Identity_82*
dtype0
T
save/Identity_83Identitysave/RestoreV2_1:34*
T0*
_output_shapes
:
a
save/AssignVariableOp_82AssignVariableOpconv4_1/weights/Adamsave/Identity_83*
dtype0
T
save/Identity_84Identitysave/RestoreV2_1:35*
T0*
_output_shapes
:
c
save/AssignVariableOp_83AssignVariableOpconv4_1/weights/Adam_1save/Identity_84*
dtype0
T
save/Identity_85Identitysave/RestoreV2_1:36*
T0*
_output_shapes
:
[
save/AssignVariableOp_84AssignVariableOpconv4_2/biasessave/Identity_85*
dtype0
T
save/Identity_86Identitysave/RestoreV2_1:37*
T0*
_output_shapes
:
`
save/AssignVariableOp_85AssignVariableOpconv4_2/biases/Adamsave/Identity_86*
dtype0
T
save/Identity_87Identitysave/RestoreV2_1:38*
T0*
_output_shapes
:
b
save/AssignVariableOp_86AssignVariableOpconv4_2/biases/Adam_1save/Identity_87*
dtype0
T
save/Identity_88Identitysave/RestoreV2_1:39*
T0*
_output_shapes
:
\
save/AssignVariableOp_87AssignVariableOpconv4_2/weightssave/Identity_88*
dtype0
T
save/Identity_89Identitysave/RestoreV2_1:40*
T0*
_output_shapes
:
a
save/AssignVariableOp_88AssignVariableOpconv4_2/weights/Adamsave/Identity_89*
dtype0
T
save/Identity_90Identitysave/RestoreV2_1:41*
T0*
_output_shapes
:
c
save/AssignVariableOp_89AssignVariableOpconv4_2/weights/Adam_1save/Identity_90*
dtype0
T
save/Identity_91Identitysave/RestoreV2_1:42*
T0*
_output_shapes
:
[
save/AssignVariableOp_90AssignVariableOpconv4_3/biasessave/Identity_91*
dtype0
T
save/Identity_92Identitysave/RestoreV2_1:43*
T0*
_output_shapes
:
`
save/AssignVariableOp_91AssignVariableOpconv4_3/biases/Adamsave/Identity_92*
dtype0
T
save/Identity_93Identitysave/RestoreV2_1:44*
T0*
_output_shapes
:
b
save/AssignVariableOp_92AssignVariableOpconv4_3/biases/Adam_1save/Identity_93*
dtype0
T
save/Identity_94Identitysave/RestoreV2_1:45*
T0*
_output_shapes
:
\
save/AssignVariableOp_93AssignVariableOpconv4_3/weightssave/Identity_94*
dtype0
T
save/Identity_95Identitysave/RestoreV2_1:46*
T0*
_output_shapes
:
a
save/AssignVariableOp_94AssignVariableOpconv4_3/weights/Adamsave/Identity_95*
dtype0
T
save/Identity_96Identitysave/RestoreV2_1:47*
T0*
_output_shapes
:
c
save/AssignVariableOp_95AssignVariableOpconv4_3/weights/Adam_1save/Identity_96*
dtype0
Ş

save/restore_shardNoOp^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_63^save/AssignVariableOp_64^save/AssignVariableOp_65^save/AssignVariableOp_66^save/AssignVariableOp_67^save/AssignVariableOp_68^save/AssignVariableOp_69^save/AssignVariableOp_70^save/AssignVariableOp_71^save/AssignVariableOp_72^save/AssignVariableOp_73^save/AssignVariableOp_74^save/AssignVariableOp_75^save/AssignVariableOp_76^save/AssignVariableOp_77^save/AssignVariableOp_78^save/AssignVariableOp_79^save/AssignVariableOp_80^save/AssignVariableOp_81^save/AssignVariableOp_82^save/AssignVariableOp_83^save/AssignVariableOp_84^save/AssignVariableOp_85^save/AssignVariableOp_86^save/AssignVariableOp_87^save/AssignVariableOp_88^save/AssignVariableOp_89^save/AssignVariableOp_90^save/AssignVariableOp_91^save/AssignVariableOp_92^save/AssignVariableOp_93^save/AssignVariableOp_94^save/AssignVariableOp_95
/
save/restore_all_1NoOp^save/restore_shardŚ/
ă
Ń
cond_false_407
cond_placeholder#
cond_cond_piecewiseconstant_and

cond_cond_const_10%
!cond_cond_piecewiseconstant_and_1

cond_cond_const_11
cond_cond_const_9
cond_cond_identity
	cond/condStatelessIfcond_cond_piecewiseconstant_andcond_cond_const_10!cond_cond_piecewiseconstant_and_1cond_cond_const_11cond_cond_const_9*
Tcond0
*
Tin
2
*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *&
else_branchR
cond_cond_false_412*
output_shapes
: *%
then_branchR
cond_cond_true_411S
cond/cond/IdentityIdentitycond/cond:output:0*
T0*
_output_shapes
: "1
cond_cond_identitycond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Á

3PiecewiseConstant_case_Assert_AssertGuard_false_389+
'assert_piecewiseconstant_case_lessequal
)
%assert_piecewiseconstant_case_preds_c

identity

Assert/data_0Const*
_output_shapes
: *
dtype0*Ë
valueÁBž BˇInput error: exclusive=True: more than 1 conditions (PiecewiseConstant/LessEqual:0, PiecewiseConstant/Greater:0, PiecewiseConstant/and:0, PiecewiseConstant/and_1:0) evaluated as True:Ź
AssertAssert'assert_piecewiseconstant_case_lessequalAssert/data_0:output:0%assert_piecewiseconstant_case_preds_c*
T
2
*
_output_shapes
 *
	summarizeg
IdentityIdentity'assert_piecewiseconstant_case_lessequal^Assert*
T0
*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes

: :: 

_output_shapes
: : 

_output_shapes
:
Ŕ
˛
$PiecewiseConstant_case_cond_true_401
	const_9_0
placeholder

placeholder_1
placeholder_2

placeholder_3
placeholder_4

placeholder_5
const_9"
const_9	const_9_0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
č
ë
%PiecewiseConstant_case_cond_false_402
cond_const_9"
cond_piecewiseconstant_greater

cond_const_12
cond_piecewiseconstant_and

cond_const_10 
cond_piecewiseconstant_and_1

cond_const_11
cond_identity
condStatelessIfcond_piecewiseconstant_greatercond_const_12cond_piecewiseconstant_andcond_const_10cond_piecewiseconstant_and_1cond_const_11cond_const_9*
Tcond0
*
Tin

2

*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *!
else_branchR
cond_false_407*
output_shapes
: * 
then_branchR
cond_true_406I
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
: "'
cond_identitycond/Identity:output:0*!
_input_shapes
: : : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ń
­
cond_true_406
cond_const_12_0
cond_placeholder

cond_placeholder_1
cond_placeholder_2

cond_placeholder_3
cond_placeholder_4
cond_const_12" 
cond_const_12cond_const_12_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ź
u
cond_cond_cond_true_416
cond_cond_cond_const_11_0
cond_cond_cond_placeholder
cond_cond_cond_const_11"4
cond_cond_cond_const_11cond_cond_cond_const_11_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: 
˙
˛
cond_cond_false_412
cond_cond_placeholder*
&cond_cond_cond_piecewiseconstant_and_1

cond_cond_cond_const_11
cond_cond_cond_const_9
cond_cond_cond_identityě
cond/cond/condStatelessIf&cond_cond_cond_piecewiseconstant_and_1cond_cond_cond_const_11cond_cond_cond_const_9*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
cond_cond_cond_false_417*
output_shapes
: **
then_branchR
cond_cond_cond_true_416]
cond/cond/cond/IdentityIdentitycond/cond/cond:output:0*
T0*
_output_shapes
: ";
cond_cond_cond_identity cond/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


2PiecewiseConstant_case_Assert_AssertGuard_true_388-
)identity_piecewiseconstant_case_lessequal

placeholder

identity
"
NoOpNoOp*
_output_shapes
 g
IdentityIdentity)identity_piecewiseconstant_case_lessequal^NoOp*
T0
*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes

: :: 

_output_shapes
: : 

_output_shapes
:


cond_cond_true_411
cond_cond_const_10_0
cond_cond_placeholder

cond_cond_placeholder_1
cond_cond_placeholder_2
cond_cond_const_10"*
cond_cond_const_10cond_cond_const_10_0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Š
t
cond_cond_cond_false_417
cond_cond_cond_placeholder
cond_cond_cond_const_9_0
cond_cond_cond_const_9"2
cond_cond_cond_const_9cond_cond_cond_const_9_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 

_output_shapes
: :

_output_shapes
: "ľ	C
save/Const_1:0save/Identity_48:0save/restore_all_1 (5 @F8"ˇ
model_variablesŁ 
|
conv1/weights:0conv1/weights/Assign#conv1/weights/Read/ReadVariableOp:0(2*conv1/weights/Initializer/random_uniform:08
o
conv1/biases:0conv1/biases/Assign"conv1/biases/Read/ReadVariableOp:0(2 conv1/biases/Initializer/zeros:08
o
conv1/alphas:0conv1/alphas/Assign"conv1/alphas/Read/ReadVariableOp:0(2 conv1/alphas/Initializer/Const:08
|
conv2/weights:0conv2/weights/Assign#conv2/weights/Read/ReadVariableOp:0(2*conv2/weights/Initializer/random_uniform:08
o
conv2/biases:0conv2/biases/Assign"conv2/biases/Read/ReadVariableOp:0(2 conv2/biases/Initializer/zeros:08
o
conv2/alphas:0conv2/alphas/Assign"conv2/alphas/Read/ReadVariableOp:0(2 conv2/alphas/Initializer/Const:08
|
conv3/weights:0conv3/weights/Assign#conv3/weights/Read/ReadVariableOp:0(2*conv3/weights/Initializer/random_uniform:08
o
conv3/biases:0conv3/biases/Assign"conv3/biases/Read/ReadVariableOp:0(2 conv3/biases/Initializer/zeros:08
o
conv3/alphas:0conv3/alphas/Assign"conv3/alphas/Read/ReadVariableOp:0(2 conv3/alphas/Initializer/Const:08

conv4_1/weights:0conv4_1/weights/Assign%conv4_1/weights/Read/ReadVariableOp:0(2,conv4_1/weights/Initializer/random_uniform:08
w
conv4_1/biases:0conv4_1/biases/Assign$conv4_1/biases/Read/ReadVariableOp:0(2"conv4_1/biases/Initializer/zeros:08

conv4_2/weights:0conv4_2/weights/Assign%conv4_2/weights/Read/ReadVariableOp:0(2,conv4_2/weights/Initializer/random_uniform:08
w
conv4_2/biases:0conv4_2/biases/Assign$conv4_2/biases/Read/ReadVariableOp:0(2"conv4_2/biases/Initializer/zeros:08

conv4_3/weights:0conv4_3/weights/Assign%conv4_3/weights/Read/ReadVariableOp:0(2,conv4_3/weights/Initializer/random_uniform:08
w
conv4_3/biases:0conv4_3/biases/Assign$conv4_3/biases/Read/ReadVariableOp:0(2"conv4_3/biases/Initializer/zeros:08"
queue_runners

input_producer)input_producer/input_producer_EnqueueMany#input_producer/input_producer_Close"%input_producer/input_producer_Close_1*
{
batch/fifo_queuebatch/fifo_queue_enqueuebatch/fifo_queue_enqueuebatch/fifo_queue_Close"batch/fifo_queue_Close_1*"Ľ
regularization_losses

)conv1/kernel/Regularizer/l2_regularizer:0
)conv2/kernel/Regularizer/l2_regularizer:0
)conv3/kernel/Regularizer/l2_regularizer:0
+conv4_1/kernel/Regularizer/l2_regularizer:0
+conv4_2/kernel/Regularizer/l2_regularizer:0
+conv4_3/kernel/Regularizer/l2_regularizer:0"
	summariest
r
$input_producer/fraction_of_32_full:0
batch/fraction_of_384_full:0
bbox_loss:0
landmark_loss:0
cls_accuracy:0"
train_op

Adam"ť
trainable_variablesŁ 
|
conv1/weights:0conv1/weights/Assign#conv1/weights/Read/ReadVariableOp:0(2*conv1/weights/Initializer/random_uniform:08
o
conv1/biases:0conv1/biases/Assign"conv1/biases/Read/ReadVariableOp:0(2 conv1/biases/Initializer/zeros:08
o
conv1/alphas:0conv1/alphas/Assign"conv1/alphas/Read/ReadVariableOp:0(2 conv1/alphas/Initializer/Const:08
|
conv2/weights:0conv2/weights/Assign#conv2/weights/Read/ReadVariableOp:0(2*conv2/weights/Initializer/random_uniform:08
o
conv2/biases:0conv2/biases/Assign"conv2/biases/Read/ReadVariableOp:0(2 conv2/biases/Initializer/zeros:08
o
conv2/alphas:0conv2/alphas/Assign"conv2/alphas/Read/ReadVariableOp:0(2 conv2/alphas/Initializer/Const:08
|
conv3/weights:0conv3/weights/Assign#conv3/weights/Read/ReadVariableOp:0(2*conv3/weights/Initializer/random_uniform:08
o
conv3/biases:0conv3/biases/Assign"conv3/biases/Read/ReadVariableOp:0(2 conv3/biases/Initializer/zeros:08
o
conv3/alphas:0conv3/alphas/Assign"conv3/alphas/Read/ReadVariableOp:0(2 conv3/alphas/Initializer/Const:08

conv4_1/weights:0conv4_1/weights/Assign%conv4_1/weights/Read/ReadVariableOp:0(2,conv4_1/weights/Initializer/random_uniform:08
w
conv4_1/biases:0conv4_1/biases/Assign$conv4_1/biases/Read/ReadVariableOp:0(2"conv4_1/biases/Initializer/zeros:08

conv4_2/weights:0conv4_2/weights/Assign%conv4_2/weights/Read/ReadVariableOp:0(2,conv4_2/weights/Initializer/random_uniform:08
w
conv4_2/biases:0conv4_2/biases/Assign$conv4_2/biases/Read/ReadVariableOp:0(2"conv4_2/biases/Initializer/zeros:08

conv4_3/weights:0conv4_3/weights/Assign%conv4_3/weights/Read/ReadVariableOp:0(2,conv4_3/weights/Initializer/random_uniform:08
w
conv4_3/biases:0conv4_3/biases/Assign$conv4_3/biases/Read/ReadVariableOp:0(2"conv4_3/biases/Initializer/zeros:08"ţ1
	variablesđ1í1
|
conv1/weights:0conv1/weights/Assign#conv1/weights/Read/ReadVariableOp:0(2*conv1/weights/Initializer/random_uniform:08
o
conv1/biases:0conv1/biases/Assign"conv1/biases/Read/ReadVariableOp:0(2 conv1/biases/Initializer/zeros:08
o
conv1/alphas:0conv1/alphas/Assign"conv1/alphas/Read/ReadVariableOp:0(2 conv1/alphas/Initializer/Const:08
|
conv2/weights:0conv2/weights/Assign#conv2/weights/Read/ReadVariableOp:0(2*conv2/weights/Initializer/random_uniform:08
o
conv2/biases:0conv2/biases/Assign"conv2/biases/Read/ReadVariableOp:0(2 conv2/biases/Initializer/zeros:08
o
conv2/alphas:0conv2/alphas/Assign"conv2/alphas/Read/ReadVariableOp:0(2 conv2/alphas/Initializer/Const:08
|
conv3/weights:0conv3/weights/Assign#conv3/weights/Read/ReadVariableOp:0(2*conv3/weights/Initializer/random_uniform:08
o
conv3/biases:0conv3/biases/Assign"conv3/biases/Read/ReadVariableOp:0(2 conv3/biases/Initializer/zeros:08
o
conv3/alphas:0conv3/alphas/Assign"conv3/alphas/Read/ReadVariableOp:0(2 conv3/alphas/Initializer/Const:08

conv4_1/weights:0conv4_1/weights/Assign%conv4_1/weights/Read/ReadVariableOp:0(2,conv4_1/weights/Initializer/random_uniform:08
w
conv4_1/biases:0conv4_1/biases/Assign$conv4_1/biases/Read/ReadVariableOp:0(2"conv4_1/biases/Initializer/zeros:08

conv4_2/weights:0conv4_2/weights/Assign%conv4_2/weights/Read/ReadVariableOp:0(2,conv4_2/weights/Initializer/random_uniform:08
w
conv4_2/biases:0conv4_2/biases/Assign$conv4_2/biases/Read/ReadVariableOp:0(2"conv4_2/biases/Initializer/zeros:08

conv4_3/weights:0conv4_3/weights/Assign%conv4_3/weights/Read/ReadVariableOp:0(2,conv4_3/weights/Initializer/random_uniform:08
w
conv4_3/biases:0conv4_3/biases/Assign$conv4_3/biases/Read/ReadVariableOp:0(2"conv4_3/biases/Initializer/zeros:08
e

Variable:0Variable/AssignVariable/Read/ReadVariableOp:0(2$Variable/Initializer/initial_value:0
q
beta1_power:0beta1_power/Assign!beta1_power/Read/ReadVariableOp:0(2'beta1_power/Initializer/initial_value:0
q
beta2_power:0beta2_power/Assign!beta2_power/Read/ReadVariableOp:0(2'beta2_power/Initializer/initial_value:0

conv1/weights/Adam:0conv1/weights/Adam/Assign(conv1/weights/Adam/Read/ReadVariableOp:0(2&conv1/weights/Adam/Initializer/zeros:0

conv1/weights/Adam_1:0conv1/weights/Adam_1/Assign*conv1/weights/Adam_1/Read/ReadVariableOp:0(2(conv1/weights/Adam_1/Initializer/zeros:0

conv1/biases/Adam:0conv1/biases/Adam/Assign'conv1/biases/Adam/Read/ReadVariableOp:0(2%conv1/biases/Adam/Initializer/zeros:0

conv1/biases/Adam_1:0conv1/biases/Adam_1/Assign)conv1/biases/Adam_1/Read/ReadVariableOp:0(2'conv1/biases/Adam_1/Initializer/zeros:0

conv1/alphas/Adam:0conv1/alphas/Adam/Assign'conv1/alphas/Adam/Read/ReadVariableOp:0(2%conv1/alphas/Adam/Initializer/zeros:0

conv1/alphas/Adam_1:0conv1/alphas/Adam_1/Assign)conv1/alphas/Adam_1/Read/ReadVariableOp:0(2'conv1/alphas/Adam_1/Initializer/zeros:0

conv2/weights/Adam:0conv2/weights/Adam/Assign(conv2/weights/Adam/Read/ReadVariableOp:0(2&conv2/weights/Adam/Initializer/zeros:0

conv2/weights/Adam_1:0conv2/weights/Adam_1/Assign*conv2/weights/Adam_1/Read/ReadVariableOp:0(2(conv2/weights/Adam_1/Initializer/zeros:0

conv2/biases/Adam:0conv2/biases/Adam/Assign'conv2/biases/Adam/Read/ReadVariableOp:0(2%conv2/biases/Adam/Initializer/zeros:0

conv2/biases/Adam_1:0conv2/biases/Adam_1/Assign)conv2/biases/Adam_1/Read/ReadVariableOp:0(2'conv2/biases/Adam_1/Initializer/zeros:0

conv2/alphas/Adam:0conv2/alphas/Adam/Assign'conv2/alphas/Adam/Read/ReadVariableOp:0(2%conv2/alphas/Adam/Initializer/zeros:0

conv2/alphas/Adam_1:0conv2/alphas/Adam_1/Assign)conv2/alphas/Adam_1/Read/ReadVariableOp:0(2'conv2/alphas/Adam_1/Initializer/zeros:0

conv3/weights/Adam:0conv3/weights/Adam/Assign(conv3/weights/Adam/Read/ReadVariableOp:0(2&conv3/weights/Adam/Initializer/zeros:0

conv3/weights/Adam_1:0conv3/weights/Adam_1/Assign*conv3/weights/Adam_1/Read/ReadVariableOp:0(2(conv3/weights/Adam_1/Initializer/zeros:0

conv3/biases/Adam:0conv3/biases/Adam/Assign'conv3/biases/Adam/Read/ReadVariableOp:0(2%conv3/biases/Adam/Initializer/zeros:0

conv3/biases/Adam_1:0conv3/biases/Adam_1/Assign)conv3/biases/Adam_1/Read/ReadVariableOp:0(2'conv3/biases/Adam_1/Initializer/zeros:0

conv3/alphas/Adam:0conv3/alphas/Adam/Assign'conv3/alphas/Adam/Read/ReadVariableOp:0(2%conv3/alphas/Adam/Initializer/zeros:0

conv3/alphas/Adam_1:0conv3/alphas/Adam_1/Assign)conv3/alphas/Adam_1/Read/ReadVariableOp:0(2'conv3/alphas/Adam_1/Initializer/zeros:0

conv4_1/weights/Adam:0conv4_1/weights/Adam/Assign*conv4_1/weights/Adam/Read/ReadVariableOp:0(2(conv4_1/weights/Adam/Initializer/zeros:0

conv4_1/weights/Adam_1:0conv4_1/weights/Adam_1/Assign,conv4_1/weights/Adam_1/Read/ReadVariableOp:0(2*conv4_1/weights/Adam_1/Initializer/zeros:0

conv4_1/biases/Adam:0conv4_1/biases/Adam/Assign)conv4_1/biases/Adam/Read/ReadVariableOp:0(2'conv4_1/biases/Adam/Initializer/zeros:0

conv4_1/biases/Adam_1:0conv4_1/biases/Adam_1/Assign+conv4_1/biases/Adam_1/Read/ReadVariableOp:0(2)conv4_1/biases/Adam_1/Initializer/zeros:0

conv4_2/weights/Adam:0conv4_2/weights/Adam/Assign*conv4_2/weights/Adam/Read/ReadVariableOp:0(2(conv4_2/weights/Adam/Initializer/zeros:0

conv4_2/weights/Adam_1:0conv4_2/weights/Adam_1/Assign,conv4_2/weights/Adam_1/Read/ReadVariableOp:0(2*conv4_2/weights/Adam_1/Initializer/zeros:0

conv4_2/biases/Adam:0conv4_2/biases/Adam/Assign)conv4_2/biases/Adam/Read/ReadVariableOp:0(2'conv4_2/biases/Adam/Initializer/zeros:0

conv4_2/biases/Adam_1:0conv4_2/biases/Adam_1/Assign+conv4_2/biases/Adam_1/Read/ReadVariableOp:0(2)conv4_2/biases/Adam_1/Initializer/zeros:0

conv4_3/weights/Adam:0conv4_3/weights/Adam/Assign*conv4_3/weights/Adam/Read/ReadVariableOp:0(2(conv4_3/weights/Adam/Initializer/zeros:0

conv4_3/weights/Adam_1:0conv4_3/weights/Adam_1/Assign,conv4_3/weights/Adam_1/Read/ReadVariableOp:0(2*conv4_3/weights/Adam_1/Initializer/zeros:0

conv4_3/biases/Adam:0conv4_3/biases/Adam/Assign)conv4_3/biases/Adam/Read/ReadVariableOp:0(2'conv4_3/biases/Adam/Initializer/zeros:0

conv4_3/biases/Adam_1:0conv4_3/biases/Adam_1/Assign+conv4_3/biases/Adam_1/Read/ReadVariableOp:0(2)conv4_3/biases/Adam_1/Initializer/zeros:0