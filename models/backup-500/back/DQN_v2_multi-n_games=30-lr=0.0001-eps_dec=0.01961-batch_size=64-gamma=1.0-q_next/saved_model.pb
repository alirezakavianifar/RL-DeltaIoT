��
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
 �"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8��
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
deep_q_network_1/dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:� */
shared_name deep_q_network_1/dense_11/bias
�
2deep_q_network_1/dense_11/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_11/bias*
_output_shapes	
:� *
dtype0
�
 deep_q_network_1/dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *1
shared_name" deep_q_network_1/dense_11/kernel
�
4deep_q_network_1/dense_11/kernel/Read/ReadVariableOpReadVariableOp deep_q_network_1/dense_11/kernel*
_output_shapes
:	� *
dtype0
�
deep_q_network_1/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name deep_q_network_1/dense_10/bias
�
2deep_q_network_1/dense_10/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_10/bias*
_output_shapes
:*
dtype0
�
 deep_q_network_1/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*1
shared_name" deep_q_network_1/dense_10/kernel
�
4deep_q_network_1/dense_10/kernel/Read/ReadVariableOpReadVariableOp deep_q_network_1/dense_10/kernel*
_output_shapes

:2*
dtype0
�
deep_q_network_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*.
shared_namedeep_q_network_1/dense_9/bias
�
1deep_q_network_1/dense_9/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_9/bias*
_output_shapes
:2*
dtype0
�
deep_q_network_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*0
shared_name!deep_q_network_1/dense_9/kernel
�
3deep_q_network_1/dense_9/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_9/kernel*
_output_shapes

:d2*
dtype0
�
deep_q_network_1/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*.
shared_namedeep_q_network_1/dense_8/bias
�
1deep_q_network_1/dense_8/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_8/bias*
_output_shapes
:d*
dtype0
�
deep_q_network_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xd*0
shared_name!deep_q_network_1/dense_8/kernel
�
3deep_q_network_1/dense_8/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_8/kernel*
_output_shapes

:xd*
dtype0
�
deep_q_network_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*.
shared_namedeep_q_network_1/dense_7/bias
�
1deep_q_network_1/dense_7/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_7/bias*
_output_shapes
:x*
dtype0
�
deep_q_network_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�x*0
shared_name!deep_q_network_1/dense_7/kernel
�
3deep_q_network_1/dense_7/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_7/kernel*
_output_shapes
:	�x*
dtype0
�
deep_q_network_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namedeep_q_network_1/dense_6/bias
�
1deep_q_network_1/dense_6/bias/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_6/bias*
_output_shapes	
:�*
dtype0
�
deep_q_network_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!deep_q_network_1/dense_6/kernel
�
3deep_q_network_1/dense_6/kernel/Read/ReadVariableOpReadVariableOpdeep_q_network_1/dense_6/kernel*
_output_shapes
:	�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1deep_q_network_1/dense_6/kerneldeep_q_network_1/dense_6/biasdeep_q_network_1/dense_7/kerneldeep_q_network_1/dense_7/biasdeep_q_network_1/dense_8/kerneldeep_q_network_1/dense_8/biasdeep_q_network_1/dense_9/kerneldeep_q_network_1/dense_9/bias deep_q_network_1/dense_10/kerneldeep_q_network_1/dense_10/bias deep_q_network_1/dense_11/kerneldeep_q_network_1/dense_11/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_73713280

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�,B�, B�,
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
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
'
 0
!1
"2
#3
$4*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias*
O
+
_variables
,_iterations
-_learning_rate
._update_step_xla*
* 

/serving_default* 
_Y
VARIABLE_VALUEdeep_q_network_1/dense_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_8/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_9/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdeep_q_network_1/dense_9/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE deep_q_network_1/dense_10/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEdeep_q_network_1/dense_10/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE deep_q_network_1/dense_11/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEdeep_q_network_1/dense_11/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
 0
!1
"2
#3
$4
	5*
* 
* 
* 
* 
* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

kernel
bias*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 

,0*
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
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 

0
1*

0
1*
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 

0
1*

0
1*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

htrace_0* 

itrace_0* 

0
1*

0
1*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 

0
1*

0
1*
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedeep_q_network_1/dense_6/kerneldeep_q_network_1/dense_6/biasdeep_q_network_1/dense_7/kerneldeep_q_network_1/dense_7/biasdeep_q_network_1/dense_8/kerneldeep_q_network_1/dense_8/biasdeep_q_network_1/dense_9/kerneldeep_q_network_1/dense_9/bias deep_q_network_1/dense_10/kerneldeep_q_network_1/dense_10/bias deep_q_network_1/dense_11/kerneldeep_q_network_1/dense_11/bias	iterationlearning_rateConst*
Tin
2*
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
!__inference__traced_save_73713506
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedeep_q_network_1/dense_6/kerneldeep_q_network_1/dense_6/biasdeep_q_network_1/dense_7/kerneldeep_q_network_1/dense_7/biasdeep_q_network_1/dense_8/kerneldeep_q_network_1/dense_8/biasdeep_q_network_1/dense_9/kerneldeep_q_network_1/dense_9/bias deep_q_network_1/dense_10/kerneldeep_q_network_1/dense_10/bias deep_q_network_1/dense_11/kerneldeep_q_network_1/dense_11/bias	iterationlearning_rate*
Tin
2*
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
$__inference__traced_restore_73713557��
�$
�
N__inference_deep_q_network_1_layer_call_and_return_conditional_losses_73713165
input_1#
dense_6_73713079:	�
dense_6_73713081:	�#
dense_7_73713095:	�x
dense_7_73713097:x"
dense_8_73713111:xd
dense_8_73713113:d"
dense_9_73713127:d2
dense_9_73713129:2#
dense_10_73713143:2
dense_10_73713145:$
dense_11_73713159:	�  
dense_11_73713161:	� 
identity�� dense_10/StatefulPartitionedCall� dense_11/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_6_73713079dense_6_73713081*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_73713078�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_73713095dense_7_73713097*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_73713094�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_73713111dense_8_73713113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_73713110�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_73713127dense_9_73713129*
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
GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_73713126�
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_73713143dense_10_73713145*
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
GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_73713142�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0dense_11_73713159dense_11_73713161*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_73713158y
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:($
"
_user_specified_name
73713161:($
"
_user_specified_name
73713159:(
$
"
_user_specified_name
73713145:(	$
"
_user_specified_name
73713143:($
"
_user_specified_name
73713129:($
"
_user_specified_name
73713127:($
"
_user_specified_name
73713113:($
"
_user_specified_name
73713111:($
"
_user_specified_name
73713097:($
"
_user_specified_name
73713095:($
"
_user_specified_name
73713081:($
"
_user_specified_name
73713079:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
3__inference_deep_q_network_1_layer_call_fn_73713194
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�x
	unknown_2:x
	unknown_3:xd
	unknown_4:d
	unknown_5:d2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:	� 

unknown_10:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_deep_q_network_1_layer_call_and_return_conditional_losses_73713165p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
73713190:($
"
_user_specified_name
73713188:(
$
"
_user_specified_name
73713186:(	$
"
_user_specified_name
73713184:($
"
_user_specified_name
73713182:($
"
_user_specified_name
73713180:($
"
_user_specified_name
73713178:($
"
_user_specified_name
73713176:($
"
_user_specified_name
73713174:($
"
_user_specified_name
73713172:($
"
_user_specified_name
73713170:($
"
_user_specified_name
73713168:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�x
�
!__inference__traced_save_73713506
file_prefixI
6read_disablecopyonread_deep_q_network_1_dense_6_kernel:	�E
6read_1_disablecopyonread_deep_q_network_1_dense_6_bias:	�K
8read_2_disablecopyonread_deep_q_network_1_dense_7_kernel:	�xD
6read_3_disablecopyonread_deep_q_network_1_dense_7_bias:xJ
8read_4_disablecopyonread_deep_q_network_1_dense_8_kernel:xdD
6read_5_disablecopyonread_deep_q_network_1_dense_8_bias:dJ
8read_6_disablecopyonread_deep_q_network_1_dense_9_kernel:d2D
6read_7_disablecopyonread_deep_q_network_1_dense_9_bias:2K
9read_8_disablecopyonread_deep_q_network_1_dense_10_kernel:2E
7read_9_disablecopyonread_deep_q_network_1_dense_10_bias:M
:read_10_disablecopyonread_deep_q_network_1_dense_11_kernel:	� G
8read_11_disablecopyonread_deep_q_network_1_dense_11_bias:	� -
#read_12_disablecopyonread_iteration:	 1
'read_13_disablecopyonread_learning_rate: 
savev2_const
identity_29��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnRead6read_disablecopyonread_deep_q_network_1_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp6read_disablecopyonread_deep_q_network_1_dense_6_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_1/DisableCopyOnReadDisableCopyOnRead6read_1_disablecopyonread_deep_q_network_1_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp6read_1_disablecopyonread_deep_q_network_1_dense_6_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnRead8read_2_disablecopyonread_deep_q_network_1_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp8read_2_disablecopyonread_deep_q_network_1_dense_7_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�x*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�xd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�x�
Read_3/DisableCopyOnReadDisableCopyOnRead6read_3_disablecopyonread_deep_q_network_1_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp6read_3_disablecopyonread_deep_q_network_1_dense_7_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:x*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:x_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:x�
Read_4/DisableCopyOnReadDisableCopyOnRead8read_4_disablecopyonread_deep_q_network_1_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp8read_4_disablecopyonread_deep_q_network_1_dense_8_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:xd*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:xdc

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:xd�
Read_5/DisableCopyOnReadDisableCopyOnRead6read_5_disablecopyonread_deep_q_network_1_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp6read_5_disablecopyonread_deep_q_network_1_dense_8_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:d*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:da
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:d�
Read_6/DisableCopyOnReadDisableCopyOnRead8read_6_disablecopyonread_deep_q_network_1_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp8read_6_disablecopyonread_deep_q_network_1_dense_9_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:d2*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:d2e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:d2�
Read_7/DisableCopyOnReadDisableCopyOnRead6read_7_disablecopyonread_deep_q_network_1_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp6read_7_disablecopyonread_deep_q_network_1_dense_9_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_8/DisableCopyOnReadDisableCopyOnRead9read_8_disablecopyonread_deep_q_network_1_dense_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp9read_8_disablecopyonread_deep_q_network_1_dense_10_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:2*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:2e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:2�
Read_9/DisableCopyOnReadDisableCopyOnRead7read_9_disablecopyonread_deep_q_network_1_dense_10_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp7read_9_disablecopyonread_deep_q_network_1_dense_10_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead:read_10_disablecopyonread_deep_q_network_1_dense_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp:read_10_disablecopyonread_deep_q_network_1_dense_11_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_11/DisableCopyOnReadDisableCopyOnRead8read_11_disablecopyonread_deep_q_network_1_dense_11_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp8read_11_disablecopyonread_deep_q_network_1_dense_11_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:� *
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:� b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:� x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_iteration^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_learning_rate^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_29Identity_29:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
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
Read_13/ReadVariableOpRead_13/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:>:
8
_user_specified_name deep_q_network_1/dense_11/bias:@<
:
_user_specified_name" deep_q_network_1/dense_11/kernel:>
:
8
_user_specified_name deep_q_network_1/dense_10/bias:@	<
:
_user_specified_name" deep_q_network_1/dense_10/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_9/bias:?;
9
_user_specified_name!deep_q_network_1/dense_9/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_8/bias:?;
9
_user_specified_name!deep_q_network_1/dense_8/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_7/bias:?;
9
_user_specified_name!deep_q_network_1/dense_7/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_6/bias:?;
9
_user_specified_name!deep_q_network_1/dense_6/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
E__inference_dense_9_layer_call_and_return_conditional_losses_73713126

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
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
:���������d: : 20
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
:���������d
 
_user_specified_nameinputs
�
�
*__inference_dense_9_layer_call_fn_73713369

inputs
unknown:d2
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
GPU 2J 8� *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_73713126o
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
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
73713365:($
"
_user_specified_name
73713363:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
*__inference_dense_6_layer_call_fn_73713309

inputs
unknown:	�
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
E__inference_dense_6_layer_call_and_return_conditional_losses_73713078p
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
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
73713305:($
"
_user_specified_name
73713303:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_6_layer_call_and_return_conditional_losses_73713078

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
+__inference_dense_10_layer_call_fn_73713389

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
GPU 2J 8� *O
fJRH
F__inference_dense_10_layer_call_and_return_conditional_losses_73713142o
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
73713385:($
"
_user_specified_name
73713383:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
*__inference_dense_8_layer_call_fn_73713349

inputs
unknown:xd
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_73713110o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������x: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
73713345:($
"
_user_specified_name
73713343:O K
'
_output_shapes
:���������x
 
_user_specified_nameinputs
�

�
F__inference_dense_11_layer_call_and_return_conditional_losses_73713300

inputs1
matmul_readvariableop_resource:	� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:���������� a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:���������� S
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
*__inference_dense_7_layer_call_fn_73713329

inputs
unknown:	�x
	unknown_0:x
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_73713094o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������x<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
73713325:($
"
_user_specified_name
73713323:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_10_layer_call_and_return_conditional_losses_73713400

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
E__inference_dense_7_layer_call_and_return_conditional_losses_73713340

inputs1
matmul_readvariableop_resource:	�x-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������xa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������xS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_10_layer_call_and_return_conditional_losses_73713142

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
F__inference_dense_11_layer_call_and_return_conditional_losses_73713158

inputs1
matmul_readvariableop_resource:	� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� W
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:���������� a
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:���������� S
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
E__inference_dense_7_layer_call_and_return_conditional_losses_73713094

inputs1
matmul_readvariableop_resource:	�x-
biasadd_readvariableop_resource:x
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�x*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������xP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������xa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������xS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_11_layer_call_fn_73713289

inputs
unknown:	� 
	unknown_0:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_73713158p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� <
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
73713285:($
"
_user_specified_name
73713283:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_6_layer_call_and_return_conditional_losses_73713320

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
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
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
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
E__inference_dense_9_layer_call_and_return_conditional_losses_73713380

inputs0
matmul_readvariableop_resource:d2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d2*
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
:���������d: : 20
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
:���������d
 
_user_specified_nameinputs
�

�
E__inference_dense_8_layer_call_and_return_conditional_losses_73713110

inputs0
matmul_readvariableop_resource:xd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������dS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������x: : 20
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
:���������x
 
_user_specified_nameinputs
�I
�
#__inference__wrapped_model_73713065
input_1J
7deep_q_network_1_dense_6_matmul_readvariableop_resource:	�G
8deep_q_network_1_dense_6_biasadd_readvariableop_resource:	�J
7deep_q_network_1_dense_7_matmul_readvariableop_resource:	�xF
8deep_q_network_1_dense_7_biasadd_readvariableop_resource:xI
7deep_q_network_1_dense_8_matmul_readvariableop_resource:xdF
8deep_q_network_1_dense_8_biasadd_readvariableop_resource:dI
7deep_q_network_1_dense_9_matmul_readvariableop_resource:d2F
8deep_q_network_1_dense_9_biasadd_readvariableop_resource:2J
8deep_q_network_1_dense_10_matmul_readvariableop_resource:2G
9deep_q_network_1_dense_10_biasadd_readvariableop_resource:K
8deep_q_network_1_dense_11_matmul_readvariableop_resource:	� H
9deep_q_network_1_dense_11_biasadd_readvariableop_resource:	� 
identity��0deep_q_network_1/dense_10/BiasAdd/ReadVariableOp�/deep_q_network_1/dense_10/MatMul/ReadVariableOp�0deep_q_network_1/dense_11/BiasAdd/ReadVariableOp�/deep_q_network_1/dense_11/MatMul/ReadVariableOp�/deep_q_network_1/dense_6/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_6/MatMul/ReadVariableOp�/deep_q_network_1/dense_7/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_7/MatMul/ReadVariableOp�/deep_q_network_1/dense_8/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_8/MatMul/ReadVariableOp�/deep_q_network_1/dense_9/BiasAdd/ReadVariableOp�.deep_q_network_1/dense_9/MatMul/ReadVariableOp�
.deep_q_network_1/dense_6/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
deep_q_network_1/dense_6/MatMulMatMulinput_16deep_q_network_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/deep_q_network_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 deep_q_network_1/dense_6/BiasAddBiasAdd)deep_q_network_1/dense_6/MatMul:product:07deep_q_network_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
deep_q_network_1/dense_6/ReluRelu)deep_q_network_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
.deep_q_network_1/dense_7/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	�x*
dtype0�
deep_q_network_1/dense_7/MatMulMatMul+deep_q_network_1/dense_6/Relu:activations:06deep_q_network_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
/deep_q_network_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype0�
 deep_q_network_1/dense_7/BiasAddBiasAdd)deep_q_network_1/dense_7/MatMul:product:07deep_q_network_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x�
deep_q_network_1/dense_7/ReluRelu)deep_q_network_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������x�
.deep_q_network_1/dense_8/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:xd*
dtype0�
deep_q_network_1/dense_8/MatMulMatMul+deep_q_network_1/dense_7/Relu:activations:06deep_q_network_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
/deep_q_network_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
 deep_q_network_1/dense_8/BiasAddBiasAdd)deep_q_network_1/dense_8/MatMul:product:07deep_q_network_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
deep_q_network_1/dense_8/ReluRelu)deep_q_network_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
.deep_q_network_1/dense_9/MatMul/ReadVariableOpReadVariableOp7deep_q_network_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype0�
deep_q_network_1/dense_9/MatMulMatMul+deep_q_network_1/dense_8/Relu:activations:06deep_q_network_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
/deep_q_network_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp8deep_q_network_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
 deep_q_network_1/dense_9/BiasAddBiasAdd)deep_q_network_1/dense_9/MatMul:product:07deep_q_network_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
deep_q_network_1/dense_9/ReluRelu)deep_q_network_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
/deep_q_network_1/dense_10/MatMul/ReadVariableOpReadVariableOp8deep_q_network_1_dense_10_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0�
 deep_q_network_1/dense_10/MatMulMatMul+deep_q_network_1/dense_9/Relu:activations:07deep_q_network_1/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0deep_q_network_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp9deep_q_network_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!deep_q_network_1/dense_10/BiasAddBiasAdd*deep_q_network_1/dense_10/MatMul:product:08deep_q_network_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
deep_q_network_1/dense_10/ReluRelu*deep_q_network_1/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:����������
/deep_q_network_1/dense_11/MatMul/ReadVariableOpReadVariableOp8deep_q_network_1_dense_11_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
 deep_q_network_1/dense_11/MatMulMatMul,deep_q_network_1/dense_10/Relu:activations:07deep_q_network_1/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
0deep_q_network_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp9deep_q_network_1_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
!deep_q_network_1/dense_11/BiasAddBiasAdd*deep_q_network_1/dense_11/MatMul:product:08deep_q_network_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
!deep_q_network_1/dense_11/SoftmaxSoftmax*deep_q_network_1/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:���������� {
IdentityIdentity+deep_q_network_1/dense_11/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:���������� �
NoOpNoOp1^deep_q_network_1/dense_10/BiasAdd/ReadVariableOp0^deep_q_network_1/dense_10/MatMul/ReadVariableOp1^deep_q_network_1/dense_11/BiasAdd/ReadVariableOp0^deep_q_network_1/dense_11/MatMul/ReadVariableOp0^deep_q_network_1/dense_6/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_6/MatMul/ReadVariableOp0^deep_q_network_1/dense_7/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_7/MatMul/ReadVariableOp0^deep_q_network_1/dense_8/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_8/MatMul/ReadVariableOp0^deep_q_network_1/dense_9/BiasAdd/ReadVariableOp/^deep_q_network_1/dense_9/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 2d
0deep_q_network_1/dense_10/BiasAdd/ReadVariableOp0deep_q_network_1/dense_10/BiasAdd/ReadVariableOp2b
/deep_q_network_1/dense_10/MatMul/ReadVariableOp/deep_q_network_1/dense_10/MatMul/ReadVariableOp2d
0deep_q_network_1/dense_11/BiasAdd/ReadVariableOp0deep_q_network_1/dense_11/BiasAdd/ReadVariableOp2b
/deep_q_network_1/dense_11/MatMul/ReadVariableOp/deep_q_network_1/dense_11/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_6/BiasAdd/ReadVariableOp/deep_q_network_1/dense_6/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_6/MatMul/ReadVariableOp.deep_q_network_1/dense_6/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_7/BiasAdd/ReadVariableOp/deep_q_network_1/dense_7/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_7/MatMul/ReadVariableOp.deep_q_network_1/dense_7/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_8/BiasAdd/ReadVariableOp/deep_q_network_1/dense_8/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_8/MatMul/ReadVariableOp.deep_q_network_1/dense_8/MatMul/ReadVariableOp2b
/deep_q_network_1/dense_9/BiasAdd/ReadVariableOp/deep_q_network_1/dense_9/BiasAdd/ReadVariableOp2`
.deep_q_network_1/dense_9/MatMul/ReadVariableOp.deep_q_network_1/dense_9/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
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
�
�
&__inference_signature_wrapper_73713280
input_1
unknown:	�
	unknown_0:	�
	unknown_1:	�x
	unknown_2:x
	unknown_3:xd
	unknown_4:d
	unknown_5:d2
	unknown_6:2
	unknown_7:2
	unknown_8:
	unknown_9:	� 

unknown_10:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_73713065p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
73713276:($
"
_user_specified_name
73713274:(
$
"
_user_specified_name
73713272:(	$
"
_user_specified_name
73713270:($
"
_user_specified_name
73713268:($
"
_user_specified_name
73713266:($
"
_user_specified_name
73713264:($
"
_user_specified_name
73713262:($
"
_user_specified_name
73713260:($
"
_user_specified_name
73713258:($
"
_user_specified_name
73713256:($
"
_user_specified_name
73713254:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�G
�	
$__inference__traced_restore_73713557
file_prefixC
0assignvariableop_deep_q_network_1_dense_6_kernel:	�?
0assignvariableop_1_deep_q_network_1_dense_6_bias:	�E
2assignvariableop_2_deep_q_network_1_dense_7_kernel:	�x>
0assignvariableop_3_deep_q_network_1_dense_7_bias:xD
2assignvariableop_4_deep_q_network_1_dense_8_kernel:xd>
0assignvariableop_5_deep_q_network_1_dense_8_bias:dD
2assignvariableop_6_deep_q_network_1_dense_9_kernel:d2>
0assignvariableop_7_deep_q_network_1_dense_9_bias:2E
3assignvariableop_8_deep_q_network_1_dense_10_kernel:2?
1assignvariableop_9_deep_q_network_1_dense_10_bias:G
4assignvariableop_10_deep_q_network_1_dense_11_kernel:	� A
2assignvariableop_11_deep_q_network_1_dense_11_bias:	� '
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: 
identity_15��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp0assignvariableop_deep_q_network_1_dense_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp0assignvariableop_1_deep_q_network_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp2assignvariableop_2_deep_q_network_1_dense_7_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp0assignvariableop_3_deep_q_network_1_dense_7_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp2assignvariableop_4_deep_q_network_1_dense_8_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_deep_q_network_1_dense_8_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_deep_q_network_1_dense_9_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp0assignvariableop_7_deep_q_network_1_dense_9_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_deep_q_network_1_dense_10_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp1assignvariableop_9_deep_q_network_1_dense_10_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp4assignvariableop_10_deep_q_network_1_dense_11_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp2assignvariableop_11_deep_q_network_1_dense_11_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:>:
8
_user_specified_name deep_q_network_1/dense_11/bias:@<
:
_user_specified_name" deep_q_network_1/dense_11/kernel:>
:
8
_user_specified_name deep_q_network_1/dense_10/bias:@	<
:
_user_specified_name" deep_q_network_1/dense_10/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_9/bias:?;
9
_user_specified_name!deep_q_network_1/dense_9/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_8/bias:?;
9
_user_specified_name!deep_q_network_1/dense_8/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_7/bias:?;
9
_user_specified_name!deep_q_network_1/dense_7/kernel:=9
7
_user_specified_namedeep_q_network_1/dense_6/bias:?;
9
_user_specified_name!deep_q_network_1/dense_6/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
E__inference_dense_8_layer_call_and_return_conditional_losses_73713360

inputs0
matmul_readvariableop_resource:xd-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������dS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������x: : 20
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
:���������x
 
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
StatefulPartitionedCall:0���������� tensorflow/serving/predict:�
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
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
3__inference_deep_q_network_1_layer_call_fn_73713194�
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
 ztrace_0
�
trace_02�
N__inference_deep_q_network_1_layer_call_and_return_conditional_losses_73713165�
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
 ztrace_0
�B�
#__inference__wrapped_model_73713065input_1"�
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
C
 0
!1
"2
#3
$4"
trackable_list_wrapper
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
j
+
_variables
,_iterations
-_learning_rate
._update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
/serving_default"
signature_map
2:0	�2deep_q_network_1/dense_6/kernel
,:*�2deep_q_network_1/dense_6/bias
2:0	�x2deep_q_network_1/dense_7/kernel
+:)x2deep_q_network_1/dense_7/bias
1:/xd2deep_q_network_1/dense_8/kernel
+:)d2deep_q_network_1/dense_8/bias
1:/d22deep_q_network_1/dense_9/kernel
+:)22deep_q_network_1/dense_9/bias
2:022 deep_q_network_1/dense_10/kernel
,:*2deep_q_network_1/dense_10/bias
3:1	� 2 deep_q_network_1/dense_11/kernel
-:+� 2deep_q_network_1/dense_11/bias
 "
trackable_list_wrapper
J
 0
!1
"2
#3
$4
	5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_deep_q_network_1_layer_call_fn_73713194input_1"�
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
N__inference_deep_q_network_1_layer_call_and_return_conditional_losses_73713165input_1"�
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
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
+__inference_dense_11_layer_call_fn_73713289�
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
 zStrace_0
�
Ttrace_02�
F__inference_dense_11_layer_call_and_return_conditional_losses_73713300�
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
 zTtrace_0
'
,0"
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
&__inference_signature_wrapper_73713280input_1"�
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
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
*__inference_dense_6_layer_call_fn_73713309�
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
 zZtrace_0
�
[trace_02�
E__inference_dense_6_layer_call_and_return_conditional_losses_73713320�
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
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
*__inference_dense_7_layer_call_fn_73713329�
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
 zatrace_0
�
btrace_02�
E__inference_dense_7_layer_call_and_return_conditional_losses_73713340�
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
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
*__inference_dense_8_layer_call_fn_73713349�
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
 zhtrace_0
�
itrace_02�
E__inference_dense_8_layer_call_and_return_conditional_losses_73713360�
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
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
*__inference_dense_9_layer_call_fn_73713369�
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
 zotrace_0
�
ptrace_02�
E__inference_dense_9_layer_call_and_return_conditional_losses_73713380�
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
 zptrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
vtrace_02�
+__inference_dense_10_layer_call_fn_73713389�
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
 zvtrace_0
�
wtrace_02�
F__inference_dense_10_layer_call_and_return_conditional_losses_73713400�
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
 zwtrace_0
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
+__inference_dense_11_layer_call_fn_73713289inputs"�
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
F__inference_dense_11_layer_call_and_return_conditional_losses_73713300inputs"�
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
*__inference_dense_6_layer_call_fn_73713309inputs"�
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
E__inference_dense_6_layer_call_and_return_conditional_losses_73713320inputs"�
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
*__inference_dense_7_layer_call_fn_73713329inputs"�
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
E__inference_dense_7_layer_call_and_return_conditional_losses_73713340inputs"�
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
*__inference_dense_8_layer_call_fn_73713349inputs"�
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
E__inference_dense_8_layer_call_and_return_conditional_losses_73713360inputs"�
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
*__inference_dense_9_layer_call_fn_73713369inputs"�
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
E__inference_dense_9_layer_call_and_return_conditional_losses_73713380inputs"�
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
+__inference_dense_10_layer_call_fn_73713389inputs"�
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
F__inference_dense_10_layer_call_and_return_conditional_losses_73713400inputs"�
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
#__inference__wrapped_model_73713065v0�-
&�#
!�
input_1���������
� "4�1
/
output_1#� 
output_1���������� �
N__inference_deep_q_network_1_layer_call_and_return_conditional_losses_73713165o0�-
&�#
!�
input_1���������
� "-�*
#� 
tensor_0���������� 
� �
3__inference_deep_q_network_1_layer_call_fn_73713194d0�-
&�#
!�
input_1���������
� ""�
unknown���������� �
F__inference_dense_10_layer_call_and_return_conditional_losses_73713400c/�,
%�"
 �
inputs���������2
� ",�)
"�
tensor_0���������
� �
+__inference_dense_10_layer_call_fn_73713389X/�,
%�"
 �
inputs���������2
� "!�
unknown����������
F__inference_dense_11_layer_call_and_return_conditional_losses_73713300d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0���������� 
� �
+__inference_dense_11_layer_call_fn_73713289Y/�,
%�"
 �
inputs���������
� ""�
unknown���������� �
E__inference_dense_6_layer_call_and_return_conditional_losses_73713320d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_6_layer_call_fn_73713309Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_7_layer_call_and_return_conditional_losses_73713340d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������x
� �
*__inference_dense_7_layer_call_fn_73713329Y0�-
&�#
!�
inputs����������
� "!�
unknown���������x�
E__inference_dense_8_layer_call_and_return_conditional_losses_73713360c/�,
%�"
 �
inputs���������x
� ",�)
"�
tensor_0���������d
� �
*__inference_dense_8_layer_call_fn_73713349X/�,
%�"
 �
inputs���������x
� "!�
unknown���������d�
E__inference_dense_9_layer_call_and_return_conditional_losses_73713380c/�,
%�"
 �
inputs���������d
� ",�)
"�
tensor_0���������2
� �
*__inference_dense_9_layer_call_fn_73713369X/�,
%�"
 �
inputs���������d
� "!�
unknown���������2�
&__inference_signature_wrapper_73713280�;�8
� 
1�.
,
input_1!�
input_1���������"4�1
/
output_1#� 
output_1���������� 