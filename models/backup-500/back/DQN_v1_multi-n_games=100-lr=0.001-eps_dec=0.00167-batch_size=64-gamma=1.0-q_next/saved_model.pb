��
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
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8ի
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
deep_q_network_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namedeep_q_network_1/dense_7/bias
�
1deep_q_network_1/dense_7/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_7/bias*
_output_shapes	
:�*
dtype0
�
deep_q_network_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!deep_q_network_1/dense_7/kernel
�
3deep_q_network_1/dense_7/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_7/kernel*
_output_shapes
:	�*
dtype0
�
deep_q_network_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namedeep_q_network_1/dense_6/bias
�
1deep_q_network_1/dense_6/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_6/bias*
_output_shapes
:*
dtype0
�
deep_q_network_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!deep_q_network_1/dense_6/kernel
�
3deep_q_network_1/dense_6/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_6/kernel*
_output_shapes

:*
dtype0
�
deep_q_network_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namedeep_q_network_1/dense_5/bias
�
1deep_q_network_1/dense_5/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_5/bias*
_output_shapes
:*
dtype0
�
deep_q_network_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!deep_q_network_1/dense_5/kernel
�
3deep_q_network_1/dense_5/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_5/kernel*
_output_shapes

:2*
dtype0
�
deep_q_network_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*.
shared_namedeep_q_network_1/dense_4/bias
�
1deep_q_network_1/dense_4/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_4/bias*
_output_shapes
:2*
dtype0
�
deep_q_network_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*0
shared_name!deep_q_network_1/dense_4/kernel
�
3deep_q_network_1/dense_4/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_4/kernel*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1deep_q_network_1/dense_4/kerneldeep_q_network_1/dense_4/biasdeep_q_network_1/dense_5/kerneldeep_q_network_1/dense_5/biasdeep_q_network_1/dense_6/kerneldeep_q_network_1/dense_6/biasdeep_q_network_1/dense_7/kerneldeep_q_network_1/dense_7/bias*
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
GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_8789481

NoOpNoOp
� 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*� 
value� B�  B�
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
O
%
_variables
&_iterations
'_learning_rate
(_update_step_xla*
* 

)serving_default* 
_Y
VARIABLE_VALUEdeep_q_network_1/dense_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_5/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_5/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_6/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_6/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_7/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_7/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
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
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

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
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Atrace_0* 

Btrace_0* 

&0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*

Htrace_0* 

Itrace_0* 

0
1*

0
1*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Otrace_0* 

Ptrace_0* 

0
1*

0
1*
* 
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedeep_q_network_1/dense_4/kerneldeep_q_network_1/dense_4/biasdeep_q_network_1/dense_5/kerneldeep_q_network_1/dense_5/biasdeep_q_network_1/dense_6/kerneldeep_q_network_1/dense_6/biasdeep_q_network_1/dense_7/kerneldeep_q_network_1/dense_7/bias	iterationlearning_rateConst*
Tin
2*
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_8789643
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeep_q_network_1/dense_4/kerneldeep_q_network_1/dense_4/biasdeep_q_network_1/dense_5/kerneldeep_q_network_1/dense_5/biasdeep_q_network_1/dense_6/kerneldeep_q_network_1/dense_6/biasdeep_q_network_1/dense_7/kerneldeep_q_network_1/dense_7/bias	iterationlearning_rate*
Tin
2*
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_8789682��
�

�
D__inference_dense_5_layer_call_and_return_conditional_losses_8789361

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
�
�
M__inference_deep_q_network_1_layer_call_and_return_conditional_losses_8789400
input_1!
dense_4_8789346:2
dense_4_8789348:2!
dense_5_8789362:2
dense_5_8789364:!
dense_6_8789378:
dense_6_8789380:"
dense_7_8789394:	�
dense_7_8789396:	�
identity��dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_4_8789346dense_4_8789348*
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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_8789345�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_8789362dense_5_8789364*
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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8789361�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_8789378dense_6_8789380*
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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_8789377�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_8789394dense_7_8789396*
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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_8789393x
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:'#
!
_user_specified_name	8789396:'#
!
_user_specified_name	8789394:'#
!
_user_specified_name	8789380:'#
!
_user_specified_name	8789378:'#
!
_user_specified_name	8789364:'#
!
_user_specified_name	8789362:'#
!
_user_specified_name	8789348:'#
!
_user_specified_name	8789346:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
)__inference_dense_6_layer_call_fn_8789550

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
GPU 2J 8� *M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_8789377o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	8789546:'#
!
_user_specified_name	8789544:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_deep_q_network_1_layer_call_fn_8789421
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
GPU 2J 8� *V
fQRO
M__inference_deep_q_network_1_layer_call_and_return_conditional_losses_8789400p
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	8789417:'#
!
_user_specified_name	8789415:'#
!
_user_specified_name	8789413:'#
!
_user_specified_name	8789411:'#
!
_user_specified_name	8789409:'#
!
_user_specified_name	8789407:'#
!
_user_specified_name	8789405:'#
!
_user_specified_name	8789403:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
D__inference_dense_7_layer_call_and_return_conditional_losses_8789501

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
D__inference_dense_4_layer_call_and_return_conditional_losses_8789345

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
�
�
)__inference_dense_7_layer_call_fn_8789490

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
GPU 2J 8� *M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_8789393p
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	8789486:'#
!
_user_specified_name	8789484:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_dense_7_layer_call_and_return_conditional_losses_8789393

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
D__inference_dense_6_layer_call_and_return_conditional_losses_8789377

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

�
D__inference_dense_5_layer_call_and_return_conditional_losses_8789541

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
D__inference_dense_6_layer_call_and_return_conditional_losses_8789561

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
�Z
�

 __inference__traced_save_8789643
file_prefixH
6read_disablecopyonread_deep_q_network_1_dense_4_kernel:2D
6read_1_disablecopyonread_deep_q_network_1_dense_4_bias:2J
8read_2_disablecopyonread_deep_q_network_1_dense_5_kernel:2D
6read_3_disablecopyonread_deep_q_network_1_dense_5_bias:J
8read_4_disablecopyonread_deep_q_network_1_dense_6_kernel:D
6read_5_disablecopyonread_deep_q_network_1_dense_6_bias:K
8read_6_disablecopyonread_deep_q_network_1_dense_7_kernel:	�E
6read_7_disablecopyonread_deep_q_network_1_dense_7_bias:	�,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: 
savev2_const
identity_21��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnRead6read_disablecopyonread_deep_q_network_1_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp6read_disablecopyonread_deep_q_network_1_dense_4_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
Read_1/DisableCopyOnReadDisableCopyOnRead6read_1_disablecopyonread_deep_q_network_1_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp6read_1_disablecopyonread_deep_q_network_1_dense_4_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
Read_2/DisableCopyOnReadDisableCopyOnRead8read_2_disablecopyonread_deep_q_network_1_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp8read_2_disablecopyonread_deep_q_network_1_dense_5_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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
Read_3/DisableCopyOnReadDisableCopyOnRead6read_3_disablecopyonread_deep_q_network_1_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp6read_3_disablecopyonread_deep_q_network_1_dense_5_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead8read_4_disablecopyonread_deep_q_network_1_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp8read_4_disablecopyonread_deep_q_network_1_dense_6_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead6read_5_disablecopyonread_deep_q_network_1_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp6read_5_disablecopyonread_deep_q_network_1_dense_6_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_6/DisableCopyOnReadDisableCopyOnRead8read_6_disablecopyonread_deep_q_network_1_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp8read_6_disablecopyonread_deep_q_network_1_dense_7_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
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
Read_7/DisableCopyOnReadDisableCopyOnRead6read_7_disablecopyonread_deep_q_network_1_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp6read_7_disablecopyonread_deep_q_network_1_dense_7_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_20Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_21IdentityIdentity_20:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_21Identity_21:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
: : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:=9
7
_user_specified_namedeep_q_network_1/dense_7/bias:?;
9
_user_specified_name!deep_q_network_1/dense_7/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_6/bias:?;
9
_user_specified_name!deep_q_network_1/dense_6/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_5/bias:?;
9
_user_specified_name!deep_q_network_1/dense_5/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_4/bias:?;
9
_user_specified_name!deep_q_network_1/dense_4/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
%__inference_signature_wrapper_8789481
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
GPU 2J 8� *+
f&R$
"__inference__wrapped_model_8789332p
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	8789477:'#
!
_user_specified_name	8789475:'#
!
_user_specified_name	8789473:'#
!
_user_specified_name	8789471:'#
!
_user_specified_name	8789469:'#
!
_user_specified_name	8789467:'#
!
_user_specified_name	8789465:'#
!
_user_specified_name	8789463:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
D__inference_dense_4_layer_call_and_return_conditional_losses_8789521

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
�
�
)__inference_dense_5_layer_call_fn_8789530

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
GPU 2J 8� *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_8789361o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	8789526:'#
!
_user_specified_name	8789524:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
)__inference_dense_4_layer_call_fn_8789510

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
GPU 2J 8� *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_8789345o
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
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	8789506:'#
!
_user_specified_name	8789504:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�1
�
"__inference__wrapped_model_8789332
input_1I
7deep_q_network_1_dense_4_matmul_readvariableop_resource:2F
8deep_q_network_1_dense_4_biasadd_readvariableop_resource:2I
7deep_q_network_1_dense_5_matmul_readvariableop_resource:2F
8deep_q_network_1_dense_5_biasadd_readvariableop_resource:I
7deep_q_network_1_dense_6_matmul_readvariableop_resource:F
8deep_q_network_1_dense_6_biasadd_readvariableop_resource:J
7deep_q_network_1_dense_7_matmul_readvariableop_resource:	�G
8deep_q_network_1_dense_7_biasadd_readvariableop_resource:	�
identity��/deep_q_network_1/dense_4/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_4/MatMul/ReadVariableOp�/deep_q_network_1/dense_5/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_5/MatMul/ReadVariableOp�/deep_q_network_1/dense_6/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_6/MatMul/ReadVariableOp�/deep_q_network_1/dense_7/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_7/MatMul/ReadVariableOp�
.deep_q_network_1/dense_4/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
deep_q_network_1/dense_4/MatMulMatMulinput_16deep_q_network_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
/deep_q_network_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
 deep_q_network_1/dense_4/BiasAddBiasAdd)deep_q_network_1/dense_4/MatMul:product:07deep_q_network_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
deep_q_network_1/dense_4/ReluRelu)deep_q_network_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
.deep_q_network_1/dense_5/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
deep_q_network_1/dense_5/MatMulMatMul+deep_q_network_1/dense_4/Relu:activations:06deep_q_network_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/deep_q_network_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 deep_q_network_1/dense_5/BiasAddBiasAdd)deep_q_network_1/dense_5/MatMul:product:07deep_q_network_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
deep_q_network_1/dense_5/ReluRelu)deep_q_network_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.deep_q_network_1/dense_6/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
deep_q_network_1/dense_6/MatMulMatMul+deep_q_network_1/dense_5/Relu:activations:06deep_q_network_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/deep_q_network_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 deep_q_network_1/dense_6/BiasAddBiasAdd)deep_q_network_1/dense_6/MatMul:product:07deep_q_network_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
deep_q_network_1/dense_6/ReluRelu)deep_q_network_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:����������
.deep_q_network_1/dense_7/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
deep_q_network_1/dense_7/MatMulMatMul+deep_q_network_1/dense_6/Relu:activations:06deep_q_network_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/deep_q_network_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 deep_q_network_1/dense_7/BiasAddBiasAdd)deep_q_network_1/dense_7/MatMul:product:07deep_q_network_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
 deep_q_network_1/dense_7/SoftmaxSoftmax)deep_q_network_1/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������z
IdentityIdentity*deep_q_network_1/dense_7/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp0^deep_q_network_1/dense_4/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_4/MatMul/ReadVariableOp0^deep_q_network_1/dense_5/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_5/MatMul/ReadVariableOp0^deep_q_network_1/dense_6/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_6/MatMul/ReadVariableOp0^deep_q_network_1/dense_7/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_7/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2b
/deep_q_network_1/dense_4/BiasAdd/ReadVariableOp/deep_q_network_1/dense_4/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_4/MatMul/ReadVariableOp.deep_q_network_1/dense_4/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_5/BiasAdd/ReadVariableOp/deep_q_network_1/dense_5/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_5/MatMul/ReadVariableOp.deep_q_network_1/dense_5/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_6/BiasAdd/ReadVariableOp/deep_q_network_1/dense_6/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_6/MatMul/ReadVariableOp.deep_q_network_1/dense_6/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_7/BiasAdd/ReadVariableOp/deep_q_network_1/dense_7/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_7/MatMul/ReadVariableOp.deep_q_network_1/dense_7/MatMul/ReadVariableOp:($
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
�4
�
#__inference__traced_restore_8789682
file_prefixB
0assignvariableop_deep_q_network_1_dense_4_kernel:2>
0assignvariableop_1_deep_q_network_1_dense_4_bias:2D
2assignvariableop_2_deep_q_network_1_dense_5_kernel:2>
0assignvariableop_3_deep_q_network_1_dense_5_bias:D
2assignvariableop_4_deep_q_network_1_dense_6_kernel:>
0assignvariableop_5_deep_q_network_1_dense_6_bias:E
2assignvariableop_6_deep_q_network_1_dense_7_kernel:	�?
0assignvariableop_7_deep_q_network_1_dense_7_bias:	�&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: 
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp0assignvariableop_deep_q_network_1_dense_4_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp0assignvariableop_1_deep_q_network_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_deep_q_network_1_dense_5_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_deep_q_network_1_dense_5_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_deep_q_network_1_dense_6_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_deep_q_network_1_dense_6_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_deep_q_network_1_dense_7_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_deep_q_network_1_dense_7_biasIdentity_7:output:0"/device:CPU:0*&
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
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_11Identity_11:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:-
)
'
_user_specified_namelearning_rate:)	%
#
_user_specified_name	iteration:=9
7
_user_specified_namedeep_q_network_1/dense_7/bias:?;
9
_user_specified_name!deep_q_network_1/dense_7/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_6/bias:?;
9
_user_specified_name!deep_q_network_1/dense_6/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_5/bias:?;
9
_user_specified_name!deep_q_network_1/dense_5/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_4/bias:?;
9
_user_specified_name!deep_q_network_1/dense_4/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
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
StatefulPartitionedCall:0����������tensorflow/serving/predict:�^
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
2__inference_deep_q_network_1_layer_call_fn_8789421�
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
M__inference_deep_q_network_1_layer_call_and_return_conditional_losses_8789400�
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
"__inference__wrapped_model_8789332input_1"�
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
j
%
_variables
&_iterations
'_learning_rate
(_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
)serving_default"
signature_map
1:/22deep_q_network_1/dense_4/kernel
+:)22deep_q_network_1/dense_4/bias
1:/22deep_q_network_1/dense_5/kernel
+:)2deep_q_network_1/dense_5/bias
1:/2deep_q_network_1/dense_6/kernel
+:)2deep_q_network_1/dense_6/bias
2:0	�2deep_q_network_1/dense_7/kernel
,:*�2deep_q_network_1/dense_7/bias
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
2__inference_deep_q_network_1_layer_call_fn_8789421input_1"�
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
M__inference_deep_q_network_1_layer_call_and_return_conditional_losses_8789400input_1"�
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
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

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
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
Atrace_02�
)__inference_dense_7_layer_call_fn_8789490�
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
 zAtrace_0
�
Btrace_02�
D__inference_dense_7_layer_call_and_return_conditional_losses_8789501�
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
 zBtrace_0
'
&0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
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
%__inference_signature_wrapper_8789481input_1"�
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
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_02�
)__inference_dense_4_layer_call_fn_8789510�
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
 zHtrace_0
�
Itrace_02�
D__inference_dense_4_layer_call_and_return_conditional_losses_8789521�
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
 zItrace_0
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
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
Otrace_02�
)__inference_dense_5_layer_call_fn_8789530�
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
 zOtrace_0
�
Ptrace_02�
D__inference_dense_5_layer_call_and_return_conditional_losses_8789541�
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
 zPtrace_0
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
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
Vtrace_02�
)__inference_dense_6_layer_call_fn_8789550�
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
 zVtrace_0
�
Wtrace_02�
D__inference_dense_6_layer_call_and_return_conditional_losses_8789561�
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
 zWtrace_0
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
)__inference_dense_7_layer_call_fn_8789490inputs"�
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
D__inference_dense_7_layer_call_and_return_conditional_losses_8789501inputs"�
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
)__inference_dense_4_layer_call_fn_8789510inputs"�
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
D__inference_dense_4_layer_call_and_return_conditional_losses_8789521inputs"�
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
)__inference_dense_5_layer_call_fn_8789530inputs"�
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
D__inference_dense_5_layer_call_and_return_conditional_losses_8789541inputs"�
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
)__inference_dense_6_layer_call_fn_8789550inputs"�
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
D__inference_dense_6_layer_call_and_return_conditional_losses_8789561inputs"�
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
"__inference__wrapped_model_8789332r0�-
&�#
!�
input_1���������
� "4�1
/
output_1#� 
output_1�����������
M__inference_deep_q_network_1_layer_call_and_return_conditional_losses_8789400k0�-
&�#
!�
input_1���������
� "-�*
#� 
tensor_0����������
� �
2__inference_deep_q_network_1_layer_call_fn_8789421`0�-
&�#
!�
input_1���������
� ""�
unknown�����������
D__inference_dense_4_layer_call_and_return_conditional_losses_8789521c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������2
� �
)__inference_dense_4_layer_call_fn_8789510X/�,
%�"
 �
inputs���������
� "!�
unknown���������2�
D__inference_dense_5_layer_call_and_return_conditional_losses_8789541c/�,
%�"
 �
inputs���������2
� ",�)
"�
tensor_0���������
� �
)__inference_dense_5_layer_call_fn_8789530X/�,
%�"
 �
inputs���������2
� "!�
unknown����������
D__inference_dense_6_layer_call_and_return_conditional_losses_8789561c/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_6_layer_call_fn_8789550X/�,
%�"
 �
inputs���������
� "!�
unknown����������
D__inference_dense_7_layer_call_and_return_conditional_losses_8789501d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_7_layer_call_fn_8789490Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
%__inference_signature_wrapper_8789481};�8
� 
1�.
,
input_1!�
input_1���������"4�1
/
output_1#� 
output_1����������