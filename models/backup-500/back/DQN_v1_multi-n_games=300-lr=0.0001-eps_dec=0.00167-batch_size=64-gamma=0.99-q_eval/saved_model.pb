��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8Ĳ
�
"Adam/v/deep_q_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/v/deep_q_network/dense_3/bias
�
6Adam/v/deep_q_network/dense_3/bias/Read/ReadVariableOpReadVariableOp"Adam/v/deep_q_network/dense_3/bias*
_output_shapes	
:�*
dtype0
�
"Adam/m/deep_q_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/m/deep_q_network/dense_3/bias
�
6Adam/m/deep_q_network/dense_3/bias/Read/ReadVariableOpReadVariableOp"Adam/m/deep_q_network/dense_3/bias*
_output_shapes	
:�*
dtype0
�
$Adam/v/deep_q_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*5
shared_name&$Adam/v/deep_q_network/dense_3/kernel
�
8Adam/v/deep_q_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/deep_q_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
$Adam/m/deep_q_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*5
shared_name&$Adam/m/deep_q_network/dense_3/kernel
�
8Adam/m/deep_q_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/deep_q_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
"Adam/v/deep_q_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/deep_q_network/dense_2/bias
�
6Adam/v/deep_q_network/dense_2/bias/Read/ReadVariableOpReadVariableOp"Adam/v/deep_q_network/dense_2/bias*
_output_shapes
:*
dtype0
�
"Adam/m/deep_q_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/deep_q_network/dense_2/bias
�
6Adam/m/deep_q_network/dense_2/bias/Read/ReadVariableOpReadVariableOp"Adam/m/deep_q_network/dense_2/bias*
_output_shapes
:*
dtype0
�
$Adam/v/deep_q_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/v/deep_q_network/dense_2/kernel
�
8Adam/v/deep_q_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/deep_q_network/dense_2/kernel*
_output_shapes

:*
dtype0
�
$Adam/m/deep_q_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/m/deep_q_network/dense_2/kernel
�
8Adam/m/deep_q_network/dense_2/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/deep_q_network/dense_2/kernel*
_output_shapes

:*
dtype0
�
"Adam/v/deep_q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/v/deep_q_network/dense_1/bias
�
6Adam/v/deep_q_network/dense_1/bias/Read/ReadVariableOpReadVariableOp"Adam/v/deep_q_network/dense_1/bias*
_output_shapes
:*
dtype0
�
"Adam/m/deep_q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/m/deep_q_network/dense_1/bias
�
6Adam/m/deep_q_network/dense_1/bias/Read/ReadVariableOpReadVariableOp"Adam/m/deep_q_network/dense_1/bias*
_output_shapes
:*
dtype0
�
$Adam/v/deep_q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*5
shared_name&$Adam/v/deep_q_network/dense_1/kernel
�
8Adam/v/deep_q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/deep_q_network/dense_1/kernel*
_output_shapes

:2*
dtype0
�
$Adam/m/deep_q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*5
shared_name&$Adam/m/deep_q_network/dense_1/kernel
�
8Adam/m/deep_q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/deep_q_network/dense_1/kernel*
_output_shapes

:2*
dtype0
�
 Adam/v/deep_q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*1
shared_name" Adam/v/deep_q_network/dense/bias
�
4Adam/v/deep_q_network/dense/bias/Read/ReadVariableOpReadVariableOp Adam/v/deep_q_network/dense/bias*
_output_shapes
:2*
dtype0
�
 Adam/m/deep_q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*1
shared_name" Adam/m/deep_q_network/dense/bias
�
4Adam/m/deep_q_network/dense/bias/Read/ReadVariableOpReadVariableOp Adam/m/deep_q_network/dense/bias*
_output_shapes
:2*
dtype0
�
"Adam/v/deep_q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*3
shared_name$"Adam/v/deep_q_network/dense/kernel
�
6Adam/v/deep_q_network/dense/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/deep_q_network/dense/kernel*
_output_shapes

:2*
dtype0
�
"Adam/m/deep_q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*3
shared_name$"Adam/m/deep_q_network/dense/kernel
�
6Adam/m/deep_q_network/dense/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/deep_q_network/dense/kernel*
_output_shapes

:2*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
deep_q_network/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namedeep_q_network/dense_3/bias
�
/deep_q_network/dense_3/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_3/bias*
_output_shapes	
:�*
dtype0
�
deep_q_network/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_namedeep_q_network/dense_3/kernel
�
1deep_q_network/dense_3/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
deep_q_network/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namedeep_q_network/dense_2/bias
�
/deep_q_network/dense_2/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_2/bias*
_output_shapes
:*
dtype0
�
deep_q_network/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namedeep_q_network/dense_2/kernel
�
1deep_q_network/dense_2/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_2/kernel*
_output_shapes

:*
dtype0
�
deep_q_network/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namedeep_q_network/dense_1/bias
�
/deep_q_network/dense_1/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_1/bias*
_output_shapes
:*
dtype0
�
deep_q_network/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*.
shared_namedeep_q_network/dense_1/kernel
�
1deep_q_network/dense_1/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense_1/kernel*
_output_shapes

:2*
dtype0
�
deep_q_network/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2**
shared_namedeep_q_network/dense/bias
�
-deep_q_network/dense/bias/Read/ReadVariableOpReadVariableOpdeep_q_network/dense/bias*
_output_shapes
:2*
dtype0
�
deep_q_network/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_namedeep_q_network/dense/kernel
�
/deep_q_network/dense/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network/dense/kernel*
_output_shapes

:2*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1deep_q_network/dense/kerneldeep_q_network/dense/biasdeep_q_network/dense_1/kerneldeep_q_network/dense_1/biasdeep_q_network/dense_2/kerneldeep_q_network/dense_2/biasdeep_q_network/dense_3/kerneldeep_q_network/dense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_26548691

NoOpNoOp
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�0
value�0B�0 B�0
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
core_layers
	fc3

	optimizer
loss

signatures*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 

0
1
2*
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias*
�
%
_variables
&_iterations
'_learning_rate
(_index_dict
)
_momentums
*_velocities
+_update_step_xla*
* 

,serving_default* 
[U
VARIABLE_VALUEdeep_q_network/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdeep_q_network/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdeep_q_network/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdeep_q_network/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdeep_q_network/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
	3*
* 
* 
* 
* 
* 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

kernel
bias*
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 
�
&0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
F0
H1
J2
L3
N4
P5
R6
T7*
<
G0
I1
K2
M3
O4
Q5
S6
U7*
* 
* 

0
1*

0
1*
* 
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 

0
1*

0
1*
* 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

btrace_0* 

ctrace_0* 

0
1*

0
1*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
* 
* 
* 
* 
* 
* 
* 
mg
VARIABLE_VALUE"Adam/m/deep_q_network/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/deep_q_network/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/deep_q_network/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/deep_q_network/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/deep_q_network/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/deep_q_network/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/deep_q_network/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/deep_q_network/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/deep_q_network/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/deep_q_network/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/deep_q_network/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/deep_q_network/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/m/deep_q_network/dense_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/deep_q_network/dense_3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/deep_q_network/dense_3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/deep_q_network/dense_3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedeep_q_network/dense/kerneldeep_q_network/dense/biasdeep_q_network/dense_1/kerneldeep_q_network/dense_1/biasdeep_q_network/dense_2/kerneldeep_q_network/dense_2/biasdeep_q_network/dense_3/kerneldeep_q_network/dense_3/bias	iterationlearning_rate"Adam/m/deep_q_network/dense/kernel"Adam/v/deep_q_network/dense/kernel Adam/m/deep_q_network/dense/bias Adam/v/deep_q_network/dense/bias$Adam/m/deep_q_network/dense_1/kernel$Adam/v/deep_q_network/dense_1/kernel"Adam/m/deep_q_network/dense_1/bias"Adam/v/deep_q_network/dense_1/bias$Adam/m/deep_q_network/dense_2/kernel$Adam/v/deep_q_network/dense_2/kernel"Adam/m/deep_q_network/dense_2/bias"Adam/v/deep_q_network/dense_2/bias$Adam/m/deep_q_network/dense_3/kernel$Adam/v/deep_q_network/dense_3/kernel"Adam/m/deep_q_network/dense_3/bias"Adam/v/deep_q_network/dense_3/biasConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_26548949
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeep_q_network/dense/kerneldeep_q_network/dense/biasdeep_q_network/dense_1/kerneldeep_q_network/dense_1/biasdeep_q_network/dense_2/kerneldeep_q_network/dense_2/biasdeep_q_network/dense_3/kerneldeep_q_network/dense_3/bias	iterationlearning_rate"Adam/m/deep_q_network/dense/kernel"Adam/v/deep_q_network/dense/kernel Adam/m/deep_q_network/dense/bias Adam/v/deep_q_network/dense/bias$Adam/m/deep_q_network/dense_1/kernel$Adam/v/deep_q_network/dense_1/kernel"Adam/m/deep_q_network/dense_1/bias"Adam/v/deep_q_network/dense_1/bias$Adam/m/deep_q_network/dense_2/kernel$Adam/v/deep_q_network/dense_2/kernel"Adam/m/deep_q_network/dense_2/bias"Adam/v/deep_q_network/dense_2/bias$Adam/m/deep_q_network/dense_3/kernel$Adam/v/deep_q_network/dense_3/kernel"Adam/m/deep_q_network/dense_3/bias"Adam/v/deep_q_network/dense_3/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_26549036��
�
�
*__inference_dense_2_layer_call_fn_26548760

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_26548587o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
26548756:($
"
_user_specified_name
26548754:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_1_layer_call_and_return_conditional_losses_26548571

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
E__inference_dense_2_layer_call_and_return_conditional_losses_26548587

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_1_layer_call_fn_26548740

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_26548571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
26548736:($
"
_user_specified_name
26548734:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_dense_3_layer_call_fn_26548700

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_26548603p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
26548696:($
"
_user_specified_name
26548694:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_2_layer_call_and_return_conditional_losses_26548771

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
1__inference_deep_q_network_layer_call_fn_26548631
input_1
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	�
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_deep_q_network_layer_call_and_return_conditional_losses_26548610p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
26548627:($
"
_user_specified_name
26548625:($
"
_user_specified_name
26548623:($
"
_user_specified_name
26548621:($
"
_user_specified_name
26548619:($
"
_user_specified_name
26548617:($
"
_user_specified_name
26548615:($
"
_user_specified_name
26548613:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
&__inference_signature_wrapper_26548691
input_1
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:	�
	unknown_6:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_26548542p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
26548687:($
"
_user_specified_name
26548685:($
"
_user_specified_name
26548683:($
"
_user_specified_name
26548681:($
"
_user_specified_name
26548679:($
"
_user_specified_name
26548677:($
"
_user_specified_name
26548675:($
"
_user_specified_name
26548673:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
L__inference_deep_q_network_layer_call_and_return_conditional_losses_26548610
input_1 
dense_26548556:2
dense_26548558:2"
dense_1_26548572:2
dense_1_26548574:"
dense_2_26548588:
dense_2_26548590:#
dense_3_26548604:	�
dense_3_26548606:	�
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_26548556dense_26548558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_26548555�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_26548572dense_1_26548574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_26548571�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_26548588dense_2_26548590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_26548587�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_26548604dense_3_26548606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_26548603x
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:($
"
_user_specified_name
26548606:($
"
_user_specified_name
26548604:($
"
_user_specified_name
26548590:($
"
_user_specified_name
26548588:($
"
_user_specified_name
26548574:($
"
_user_specified_name
26548572:($
"
_user_specified_name
26548558:($
"
_user_specified_name
26548556:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�0
�
#__inference__wrapped_model_26548542
input_1E
3deep_q_network_dense_matmul_readvariableop_resource:2B
4deep_q_network_dense_biasadd_readvariableop_resource:2G
5deep_q_network_dense_1_matmul_readvariableop_resource:2D
6deep_q_network_dense_1_biasadd_readvariableop_resource:G
5deep_q_network_dense_2_matmul_readvariableop_resource:D
6deep_q_network_dense_2_biasadd_readvariableop_resource:H
5deep_q_network_dense_3_matmul_readvariableop_resource:	�E
6deep_q_network_dense_3_biasadd_readvariableop_resource:	�
identity��+deep_q_network/dense/BiasAdd/ReadVariableOp�*deep_q_network/dense/MatMul/ReadVariableOp�-deep_q_network/dense_1/BiasAdd/ReadVariableOp�,deep_q_network/dense_1/MatMul/ReadVariableOp�-deep_q_network/dense_2/BiasAdd/ReadVariableOp�,deep_q_network/dense_2/MatMul/ReadVariableOp�-deep_q_network/dense_3/BiasAdd/ReadVariableOp�,deep_q_network/dense_3/MatMul/ReadVariableOp�
*deep_q_network/dense/MatMul/ReadVariableOpReadVariableOp3deep_q_network_dense_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
deep_q_network/dense/MatMulMatMulinput_12deep_q_network/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
+deep_q_network/dense/BiasAdd/ReadVariableOpReadVariableOp4deep_q_network_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
deep_q_network/dense/BiasAddBiasAdd%deep_q_network/dense/MatMul:product:03deep_q_network/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2z
deep_q_network/dense/ReluRelu%deep_q_network/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
,deep_q_network/dense_1/MatMul/ReadVariableOpReadVariableOp5deep_q_network_dense_1_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
deep_q_network/dense_1/MatMulMatMul'deep_q_network/dense/Relu:activations:04deep_q_network/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-deep_q_network/dense_1/BiasAdd/ReadVariableOpReadVariableOp6deep_q_network_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
deep_q_network/dense_1/BiasAddBiasAdd'deep_q_network/dense_1/MatMul:product:05deep_q_network/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
deep_q_network/dense_1/ReluRelu'deep_q_network/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,deep_q_network/dense_2/MatMul/ReadVariableOpReadVariableOp5deep_q_network_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
deep_q_network/dense_2/MatMulMatMul)deep_q_network/dense_1/Relu:activations:04deep_q_network/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-deep_q_network/dense_2/BiasAdd/ReadVariableOpReadVariableOp6deep_q_network_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
deep_q_network/dense_2/BiasAddBiasAdd'deep_q_network/dense_2/MatMul:product:05deep_q_network/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
deep_q_network/dense_2/ReluRelu'deep_q_network/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,deep_q_network/dense_3/MatMul/ReadVariableOpReadVariableOp5deep_q_network_dense_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
deep_q_network/dense_3/MatMulMatMul)deep_q_network/dense_2/Relu:activations:04deep_q_network/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-deep_q_network/dense_3/BiasAdd/ReadVariableOpReadVariableOp6deep_q_network_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
deep_q_network/dense_3/BiasAddBiasAdd'deep_q_network/dense_3/MatMul:product:05deep_q_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
deep_q_network/dense_3/SoftmaxSoftmax'deep_q_network/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������x
IdentityIdentity(deep_q_network/dense_3/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp,^deep_q_network/dense/BiasAdd/ReadVariableOp+^deep_q_network/dense/MatMul/ReadVariableOp.^deep_q_network/dense_1/BiasAdd/ReadVariableOp-^deep_q_network/dense_1/MatMul/ReadVariableOp.^deep_q_network/dense_2/BiasAdd/ReadVariableOp-^deep_q_network/dense_2/MatMul/ReadVariableOp.^deep_q_network/dense_3/BiasAdd/ReadVariableOp-^deep_q_network/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2Z
+deep_q_network/dense/BiasAdd/ReadVariableOp+deep_q_network/dense/BiasAdd/ReadVariableOp2X
*deep_q_network/dense/MatMul/ReadVariableOp*deep_q_network/dense/MatMul/ReadVariableOp2^
-deep_q_network/dense_1/BiasAdd/ReadVariableOp-deep_q_network/dense_1/BiasAdd/ReadVariableOp2\
,deep_q_network/dense_1/MatMul/ReadVariableOp,deep_q_network/dense_1/MatMul/ReadVariableOp2^
-deep_q_network/dense_2/BiasAdd/ReadVariableOp-deep_q_network/dense_2/BiasAdd/ReadVariableOp2\
,deep_q_network/dense_2/MatMul/ReadVariableOp,deep_q_network/dense_2/MatMul/ReadVariableOp2^
-deep_q_network/dense_3/BiasAdd/ReadVariableOp-deep_q_network/dense_3/BiasAdd/ReadVariableOp2\
,deep_q_network/dense_3/MatMul/ReadVariableOp,deep_q_network/dense_3/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
!__inference__traced_save_26548949
file_prefixD
2read_disablecopyonread_deep_q_network_dense_kernel:2@
2read_1_disablecopyonread_deep_q_network_dense_bias:2H
6read_2_disablecopyonread_deep_q_network_dense_1_kernel:2B
4read_3_disablecopyonread_deep_q_network_dense_1_bias:H
6read_4_disablecopyonread_deep_q_network_dense_2_kernel:B
4read_5_disablecopyonread_deep_q_network_dense_2_bias:I
6read_6_disablecopyonread_deep_q_network_dense_3_kernel:	�C
4read_7_disablecopyonread_deep_q_network_dense_3_bias:	�,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: N
<read_10_disablecopyonread_adam_m_deep_q_network_dense_kernel:2N
<read_11_disablecopyonread_adam_v_deep_q_network_dense_kernel:2H
:read_12_disablecopyonread_adam_m_deep_q_network_dense_bias:2H
:read_13_disablecopyonread_adam_v_deep_q_network_dense_bias:2P
>read_14_disablecopyonread_adam_m_deep_q_network_dense_1_kernel:2P
>read_15_disablecopyonread_adam_v_deep_q_network_dense_1_kernel:2J
<read_16_disablecopyonread_adam_m_deep_q_network_dense_1_bias:J
<read_17_disablecopyonread_adam_v_deep_q_network_dense_1_bias:P
>read_18_disablecopyonread_adam_m_deep_q_network_dense_2_kernel:P
>read_19_disablecopyonread_adam_v_deep_q_network_dense_2_kernel:J
<read_20_disablecopyonread_adam_m_deep_q_network_dense_2_bias:J
<read_21_disablecopyonread_adam_v_deep_q_network_dense_2_bias:Q
>read_22_disablecopyonread_adam_m_deep_q_network_dense_3_kernel:	�Q
>read_23_disablecopyonread_adam_v_deep_q_network_dense_3_kernel:	�K
<read_24_disablecopyonread_adam_m_deep_q_network_dense_3_bias:	�K
<read_25_disablecopyonread_adam_v_deep_q_network_dense_3_bias:	�
savev2_const
identity_53��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead2read_disablecopyonread_deep_q_network_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp2read_disablecopyonread_deep_q_network_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_1/DisableCopyOnReadDisableCopyOnRead2read_1_disablecopyonread_deep_q_network_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp2read_1_disablecopyonread_deep_q_network_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_2/DisableCopyOnReadDisableCopyOnRead6read_2_disablecopyonread_deep_q_network_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp6read_2_disablecopyonread_deep_q_network_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_deep_q_network_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_deep_q_network_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead6read_4_disablecopyonread_deep_q_network_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp6read_4_disablecopyonread_deep_q_network_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_deep_q_network_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_deep_q_network_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_6/DisableCopyOnReadDisableCopyOnRead6read_6_disablecopyonread_deep_q_network_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp6read_6_disablecopyonread_deep_q_network_dense_3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_7/DisableCopyOnReadDisableCopyOnRead4read_7_disablecopyonread_deep_q_network_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp4read_7_disablecopyonread_deep_q_network_dense_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead<read_10_disablecopyonread_adam_m_deep_q_network_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp<read_10_disablecopyonread_adam_m_deep_q_network_dense_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_11/DisableCopyOnReadDisableCopyOnRead<read_11_disablecopyonread_adam_v_deep_q_network_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp<read_11_disablecopyonread_adam_v_deep_q_network_dense_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_12/DisableCopyOnReadDisableCopyOnRead:read_12_disablecopyonread_adam_m_deep_q_network_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp:read_12_disablecopyonread_adam_m_deep_q_network_dense_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_13/DisableCopyOnReadDisableCopyOnRead:read_13_disablecopyonread_adam_v_deep_q_network_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp:read_13_disablecopyonread_adam_v_deep_q_network_dense_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_14/DisableCopyOnReadDisableCopyOnRead>read_14_disablecopyonread_adam_m_deep_q_network_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp>read_14_disablecopyonread_adam_m_deep_q_network_dense_1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_15/DisableCopyOnReadDisableCopyOnRead>read_15_disablecopyonread_adam_v_deep_q_network_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp>read_15_disablecopyonread_adam_v_deep_q_network_dense_1_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_adam_m_deep_q_network_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_adam_m_deep_q_network_dense_1_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead<read_17_disablecopyonread_adam_v_deep_q_network_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp<read_17_disablecopyonread_adam_v_deep_q_network_dense_1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnRead>read_18_disablecopyonread_adam_m_deep_q_network_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp>read_18_disablecopyonread_adam_m_deep_q_network_dense_2_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_19/DisableCopyOnReadDisableCopyOnRead>read_19_disablecopyonread_adam_v_deep_q_network_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp>read_19_disablecopyonread_adam_v_deep_q_network_dense_2_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_20/DisableCopyOnReadDisableCopyOnRead<read_20_disablecopyonread_adam_m_deep_q_network_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp<read_20_disablecopyonread_adam_m_deep_q_network_dense_2_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead<read_21_disablecopyonread_adam_v_deep_q_network_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp<read_21_disablecopyonread_adam_v_deep_q_network_dense_2_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead>read_22_disablecopyonread_adam_m_deep_q_network_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp>read_22_disablecopyonread_adam_m_deep_q_network_dense_3_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_23/DisableCopyOnReadDisableCopyOnRead>read_23_disablecopyonread_adam_v_deep_q_network_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp>read_23_disablecopyonread_adam_v_deep_q_network_dense_3_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_24/DisableCopyOnReadDisableCopyOnRead<read_24_disablecopyonread_adam_m_deep_q_network_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp<read_24_disablecopyonread_adam_m_deep_q_network_dense_3_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnRead<read_25_disablecopyonread_adam_v_deep_q_network_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp<read_25_disablecopyonread_adam_v_deep_q_network_dense_3_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *)
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_52Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_53IdentityIdentity_52:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense_3/bias:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense_3/bias:D@
>
_user_specified_name&$Adam/v/deep_q_network/dense_3/kernel:D@
>
_user_specified_name&$Adam/m/deep_q_network/dense_3/kernel:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense_2/bias:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense_2/bias:D@
>
_user_specified_name&$Adam/v/deep_q_network/dense_2/kernel:D@
>
_user_specified_name&$Adam/m/deep_q_network/dense_2/kernel:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense_1/bias:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense_1/bias:D@
>
_user_specified_name&$Adam/v/deep_q_network/dense_1/kernel:D@
>
_user_specified_name&$Adam/m/deep_q_network/dense_1/kernel:@<
:
_user_specified_name" Adam/v/deep_q_network/dense/bias:@<
:
_user_specified_name" Adam/m/deep_q_network/dense/bias:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense/kernel:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:;7
5
_user_specified_namedeep_q_network/dense_3/bias:=9
7
_user_specified_namedeep_q_network/dense_3/kernel:;7
5
_user_specified_namedeep_q_network/dense_2/bias:=9
7
_user_specified_namedeep_q_network/dense_2/kernel:;7
5
_user_specified_namedeep_q_network/dense_1/bias:=9
7
_user_specified_namedeep_q_network/dense_1/kernel:95
3
_user_specified_namedeep_q_network/dense/bias:;7
5
_user_specified_namedeep_q_network/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
E__inference_dense_3_layer_call_and_return_conditional_losses_26548711

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_layer_call_and_return_conditional_losses_26548731

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_3_layer_call_and_return_conditional_losses_26548603

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_1_layer_call_and_return_conditional_losses_26548751

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
C__inference_dense_layer_call_and_return_conditional_losses_26548555

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
$__inference__traced_restore_26549036
file_prefix>
,assignvariableop_deep_q_network_dense_kernel:2:
,assignvariableop_1_deep_q_network_dense_bias:2B
0assignvariableop_2_deep_q_network_dense_1_kernel:2<
.assignvariableop_3_deep_q_network_dense_1_bias:B
0assignvariableop_4_deep_q_network_dense_2_kernel:<
.assignvariableop_5_deep_q_network_dense_2_bias:C
0assignvariableop_6_deep_q_network_dense_3_kernel:	�=
.assignvariableop_7_deep_q_network_dense_3_bias:	�&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: H
6assignvariableop_10_adam_m_deep_q_network_dense_kernel:2H
6assignvariableop_11_adam_v_deep_q_network_dense_kernel:2B
4assignvariableop_12_adam_m_deep_q_network_dense_bias:2B
4assignvariableop_13_adam_v_deep_q_network_dense_bias:2J
8assignvariableop_14_adam_m_deep_q_network_dense_1_kernel:2J
8assignvariableop_15_adam_v_deep_q_network_dense_1_kernel:2D
6assignvariableop_16_adam_m_deep_q_network_dense_1_bias:D
6assignvariableop_17_adam_v_deep_q_network_dense_1_bias:J
8assignvariableop_18_adam_m_deep_q_network_dense_2_kernel:J
8assignvariableop_19_adam_v_deep_q_network_dense_2_kernel:D
6assignvariableop_20_adam_m_deep_q_network_dense_2_bias:D
6assignvariableop_21_adam_v_deep_q_network_dense_2_bias:K
8assignvariableop_22_adam_m_deep_q_network_dense_3_kernel:	�K
8assignvariableop_23_adam_v_deep_q_network_dense_3_kernel:	�E
6assignvariableop_24_adam_m_deep_q_network_dense_3_bias:	�E
6assignvariableop_25_adam_v_deep_q_network_dense_3_bias:	�
identity_27��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp,assignvariableop_deep_q_network_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp,assignvariableop_1_deep_q_network_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_deep_q_network_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_deep_q_network_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_deep_q_network_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_deep_q_network_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp0assignvariableop_6_deep_q_network_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_deep_q_network_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_adam_m_deep_q_network_dense_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp6assignvariableop_11_adam_v_deep_q_network_dense_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_adam_m_deep_q_network_dense_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_v_deep_q_network_dense_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp8assignvariableop_14_adam_m_deep_q_network_dense_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adam_v_deep_q_network_dense_1_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_adam_m_deep_q_network_dense_1_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_v_deep_q_network_dense_1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adam_m_deep_q_network_dense_2_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_v_deep_q_network_dense_2_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_m_deep_q_network_dense_2_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_v_deep_q_network_dense_2_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_m_deep_q_network_dense_3_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_adam_v_deep_q_network_dense_3_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_m_deep_q_network_dense_3_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_v_deep_q_network_dense_3_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_27Identity_27:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense_3/bias:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense_3/bias:D@
>
_user_specified_name&$Adam/v/deep_q_network/dense_3/kernel:D@
>
_user_specified_name&$Adam/m/deep_q_network/dense_3/kernel:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense_2/bias:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense_2/bias:D@
>
_user_specified_name&$Adam/v/deep_q_network/dense_2/kernel:D@
>
_user_specified_name&$Adam/m/deep_q_network/dense_2/kernel:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense_1/bias:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense_1/bias:D@
>
_user_specified_name&$Adam/v/deep_q_network/dense_1/kernel:D@
>
_user_specified_name&$Adam/m/deep_q_network/dense_1/kernel:@<
:
_user_specified_name" Adam/v/deep_q_network/dense/bias:@<
:
_user_specified_name" Adam/m/deep_q_network/dense/bias:B>
<
_user_specified_name$"Adam/v/deep_q_network/dense/kernel:B>
<
_user_specified_name$"Adam/m/deep_q_network/dense/kernel:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:;7
5
_user_specified_namedeep_q_network/dense_3/bias:=9
7
_user_specified_namedeep_q_network/dense_3/kernel:;7
5
_user_specified_namedeep_q_network/dense_2/bias:=9
7
_user_specified_namedeep_q_network/dense_2/kernel:;7
5
_user_specified_namedeep_q_network/dense_1/bias:=9
7
_user_specified_namedeep_q_network/dense_1/kernel:95
3
_user_specified_namedeep_q_network/dense/bias:;7
5
_user_specified_namedeep_q_network/dense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_dense_layer_call_fn_26548720

inputs
unknown:2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_26548555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
26548716:($
"
_user_specified_name
26548714:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������=
output_11
StatefulPartitionedCall:0����������tensorflow/serving/predict:�h
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
core_layers
	fc3

	optimizer
loss

signatures"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
1__inference_deep_q_network_layer_call_fn_26548631�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
L__inference_deep_q_network_layer_call_and_return_conditional_losses_26548610�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
#__inference__wrapped_model_26548542input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
0
1
2"
trackable_list_wrapper
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
%
_variables
&_iterations
'_learning_rate
(_index_dict
)
_momentums
*_velocities
+_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
,serving_default"
signature_map
-:+22deep_q_network/dense/kernel
':%22deep_q_network/dense/bias
/:-22deep_q_network/dense_1/kernel
):'2deep_q_network/dense_1/bias
/:-2deep_q_network/dense_2/kernel
):'2deep_q_network/dense_2/bias
0:.	�2deep_q_network/dense_3/kernel
*:(�2deep_q_network/dense_3/bias
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_deep_q_network_layer_call_fn_26548631input_1"�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_deep_q_network_layer_call_and_return_conditional_losses_26548610input_1"�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Dtrace_02�
*__inference_dense_3_layer_call_fn_26548700�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
�
Etrace_02�
E__inference_dense_3_layer_call_and_return_conditional_losses_26548711�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
�
&0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
X
F0
H1
J2
L3
N4
P5
R6
T7"
trackable_list_wrapper
X
G0
I1
K2
M3
O4
Q5
S6
U7"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
&__inference_signature_wrapper_26548691input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
[trace_02�
(__inference_dense_layer_call_fn_26548720�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
�
\trace_02�
C__inference_dense_layer_call_and_return_conditional_losses_26548731�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z\trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
btrace_02�
*__inference_dense_1_layer_call_fn_26548740�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
�
ctrace_02�
E__inference_dense_1_layer_call_and_return_conditional_losses_26548751�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zctrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
*__inference_dense_2_layer_call_fn_26548760�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
�
jtrace_02�
E__inference_dense_2_layer_call_and_return_conditional_losses_26548771�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zjtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_3_layer_call_fn_26548700inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_3_layer_call_and_return_conditional_losses_26548711inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2:022"Adam/m/deep_q_network/dense/kernel
2:022"Adam/v/deep_q_network/dense/kernel
,:*22 Adam/m/deep_q_network/dense/bias
,:*22 Adam/v/deep_q_network/dense/bias
4:222$Adam/m/deep_q_network/dense_1/kernel
4:222$Adam/v/deep_q_network/dense_1/kernel
.:,2"Adam/m/deep_q_network/dense_1/bias
.:,2"Adam/v/deep_q_network/dense_1/bias
4:22$Adam/m/deep_q_network/dense_2/kernel
4:22$Adam/v/deep_q_network/dense_2/kernel
.:,2"Adam/m/deep_q_network/dense_2/bias
.:,2"Adam/v/deep_q_network/dense_2/bias
5:3	�2$Adam/m/deep_q_network/dense_3/kernel
5:3	�2$Adam/v/deep_q_network/dense_3/kernel
/:-�2"Adam/m/deep_q_network/dense_3/bias
/:-�2"Adam/v/deep_q_network/dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_layer_call_fn_26548720inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_layer_call_and_return_conditional_losses_26548731inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_1_layer_call_fn_26548740inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_1_layer_call_and_return_conditional_losses_26548751inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dense_2_layer_call_fn_26548760inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dense_2_layer_call_and_return_conditional_losses_26548771inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_26548542r0�-
&�#
!�
input_1���������
� "4�1
/
output_1#� 
output_1�����������
L__inference_deep_q_network_layer_call_and_return_conditional_losses_26548610k0�-
&�#
!�
input_1���������
� "-�*
#� 
tensor_0����������
� �
1__inference_deep_q_network_layer_call_fn_26548631`0�-
&�#
!�
input_1���������
� ""�
unknown�����������
E__inference_dense_1_layer_call_and_return_conditional_losses_26548751c/�,
%�"
 �
inputs���������2
� ",�)
"�
tensor_0���������
� �
*__inference_dense_1_layer_call_fn_26548740X/�,
%�"
 �
inputs���������2
� "!�
unknown����������
E__inference_dense_2_layer_call_and_return_conditional_losses_26548771c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
*__inference_dense_2_layer_call_fn_26548760X/�,
%�"
 �
inputs���������
� "!�
unknown����������
E__inference_dense_3_layer_call_and_return_conditional_losses_26548711d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_3_layer_call_fn_26548700Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
C__inference_dense_layer_call_and_return_conditional_losses_26548731c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������2
� �
(__inference_dense_layer_call_fn_26548720X/�,
%�"
 �
inputs���������
� "!�
unknown���������2�
&__inference_signature_wrapper_26548691};�8
� 
1�.
,
input_1!�
input_1���������"4�1
/
output_1#� 
output_1����������