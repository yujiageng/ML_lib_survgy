#8. TF2GAP8
链接： [TF2GAP8](https://github.com/GreenWaves-Technologies/tf2gap8) 
## 8.1 介绍

TF2GAP8是将TensorFlow CNN模型文件迁移到GAP8设备上的工具。

##8.2 工作流程
以MNIST为例。MNIST目标是识别手写数字。它的第一层卷积上存在32个5x
5的滤波器，和一个2x2的池化层。然后网络在第二卷积层上扩展了64个5x5
的滤波器，2x2最大池和具有10个输出的密集层用来预测。

1. 训练模型
该示例在example文件夹下python文件中实现
	- mnist.py：训练主程序
	- cnn.py ：创建不同图层的源码
运行mnist.py开始训练，生成Inference Graph的PB文件（.pbtxt）和CheckPoint文件(.ckpt)。

2. freeze_graph
freeze_graph具体就是结合CheckPoint文件和Inference Graph PB文件，生成Freeze Graph的PB文件，便于发布。 freeze_graph.py脚本位于TensorFlow安装目录下，执行代码如下：

		result = subprocess.run([TFDir + '/bazel-bin/tensorflow/python/tools/freeze_graph',
                                '--input_graph=' + inputGraph,
                                '--input_checkpoint=' + inputCheckpoint,
                                '--output_graph=' + getFrozenGraphName(inputGraph),
                                '--output_node_names=' + outputNode],
                                stdout=subprocess.PIPE)
	
3. transform_graph
为了将TensorFlow模型部署到GAP8设备上，应该致力于减少模型的内存占用，缩短推断施加，减少耗电。TF2GAP8使用了TensorFlow中的量化工具transform_graph进行模型压缩，删除前向传播过程中未调用到的节点，同时为了匹配GAP8 CNN库的运算，在命令—transforms的选项下向GTT添加了一些节点因子，TF2GAP8中执行代码如下：
	
		result=subprocess.run([TFDir +'/bazel-bin/tensorflow/tools/graph_transforms/transform_graph',
                          '--in_graph='+ frozenGraphName,
                          '--out_graph=' + getOptimizedGraphName(inputGraph),
                          '--inputs=' + inputNode,
                          '--outputs=' + outputNode,
                          '--transforms=strip_unused_nodes remove_nodes(op=Identity) fuse_conv2d_add_relu_maxpool fuse_conv2d_add_relu fuse_conv2d_add_maxpool fuse_GAP8_conv2d_maxpool fuse_reshape_matmul_add_relu_softmax fuse_reshape_matmul_add_softmax'], stdout=subprocess.PIPE)

4. 生成GAP8的模型源码
在此阶段，TF2GAP8工具将优化后模型转化成GAP8可以仿真的源码。调用了TF2GAP8的内部脚本tfgap8.cc,生成相应的源码。执行代码如下：
		
		result=subprocess.run([TFDir + '/bazel-bin/tf2gap8/tf2gap8', getGraphName(inputGraph) + "_optimized" + ".pb",
                           getGraphDir(inputGraph),TFDir + '/tf2gap8',ftp],stdout=subprocess.PIPE)

	主要通过读取优化后模型文件，获取CNN模型的推理过程，如MNIST采用了三层网络，
	第一层卷积层：存在32个5x5的滤波器，和一个2x2的池化层
	第二层卷积层：扩展了64个5x5的滤波器和2x2的最大池
	第三层密集层：具有10个输出的密集层用来预测

	并提取出训练最后各层的权重、偏值等参数。将推理过程和各个参数生成GAP8 SDK可以运行的代码。如下：

	- Network_process.c：模型的推理过程，采用两层卷积层和一层用于预测的密集层。

		l2_x为输入，即测试用的手写图片的特征值，L2_W_[1/2/3]:每一层训练后的权重，L2_B_[1/2/3]:各层训练后的偏值。Dense1中l2_big0:为最后的预测输出。

			#include "network_process.h"

			void network_process () { 
	    	 	ConvLayer1(l2_x,L2_W_0,L2_B_0,l2_big0,14,AllKernels + 0); 
	     	 	ConvLayer2(l2_big0,L2_W_1,L2_B_1,l2_big1,14,AllKernels + 1); 
	   		 	Dense1(l2_big1,L2_W_2,L2_B_.2,(int*)l2_big0,16,13,AllKernels + 2); 
			} 

5. GAP8 MNIST CNN模型执行过程

	1. 通过模型提取特征值算法获取待测手写数字图片的特征值，
	2. 编写C语言程序，将特征值赋给CNN推理的输入参数

			unsigned short l2_x[]={
				...
			};
			
	3. 运行通过TF2GAP8生成的GAP8可执行代码，开始推理。调用GAP8 SDK中CNN的内核函数实现卷积、池化、RELU、SoftMax等操作，具体取决于读取到的模型的CNN具体推理流程。MNIST示例中，第一层卷积层中执行了一下两个CNN内核函数：

			rt_team_fork(gap8_ncore(), (void *) KerConv5x5Stride1_fp, (void *) KerArg1);
			...
			rt_team_fork(gap8_ncore(), (void *) KerPool2x2Stride2_fp, (void *) KerArg2);

		实现了一个5x5滤波器的卷积层和一个2x2的池化层的推理过程。rt_team_fork函数可以并行运行GAP8的多个核执行KerConv5x5Stride1_fp内核函数。第一个参数为使用的核的数量，参数三为当前执行的集群ID。
		
		3.1  使用GAP8的AutoTiler工具的代码生成器将内核CNN函数编译生成适应GAP8内存管理的c语言程序。并运行完成CNN内核函数的调用。
	
		由于GAP8没有实现数据缓存，并且由于GAP8的集群针对线性或分段线性方式处理数据进行了优化，因此通过GAP8 auto-tiler，通过自动化的内存移动生成GAP8可以运行的程序。
		
		CNN内核函数KerDirectConv5x5_fp,通过auto-tiler生成如下代码：

			void GenerateCnnConv5x5(char *Name, unsigned int InPlane,
			  unsigned int OutPlane, unsigned int W, unsigned int H)
			{
			  UserKernel(Name,
			    KernelDimensions(InPlane, W, H, OutPlane),
			    KernelIterationOrder(KER_DIM4, KER_OUT_PLANE, KER_TILE, KER_IN_PLANE),
			    TILE_HOR,
			    CArgs(5,
			      TCArg("short int * __restrict__", "In"),
			      TCArg("short int * __restrict__", "Filter"),
			      TCArg("short int * __restrict__", "Out"),
			      TCArg("unsigned int",             "Norm"),
			      TCArg("short int * __restrict__", "Bias")
			    ),
			    Calls(2,
			      Call("KerSetInBias", LOC_IN_PLANE_PROLOG,
			        Bindings(4,
			          K_Arg("Out", KER_ARG_TILE),
			          K_Arg("Out", KER_ARG_TILE_W),
			          K_Arg("Out", KER_ARG_TILE_H),
			          C_ArgIndex("Bias", KER_OUT_PLANE, 1)
			        )
			      ),
			      Call("KerDirectConv5x5_fp", LOC_INNER_LOOP,
			        Bindings(6, 
			          K_Arg("In", KER_ARG_TILE),
			          K_Arg("In", KER_ARG_TILE_W),
			          K_Arg("In", KER_ARG_TILE_H),
			          K_Arg("Filter", KER_ARG_TILE),
			          K_Arg("Out", KER_ARG_TILE),
			          C_Arg("Norm")
			        )
			      )
			    ),
			    KerArgs(3,
			      KerArg("In", OBJ_IN_DB_3D, W, H, sizeof(short int), 5-1, 0, 0, "In", 0),
			      KerArg("Filter", OBJ_IN_DB_NTILED_4D, 5, 5, 
			        sizeof(short int), 0, 0, 0, "Filter", 0),
			      KerArg("Out", OBJ_OUT_DB_3D, W-5+1, H-5+1, 
			        sizeof(short int), 0, 0, 0, "Out", 0)
			    )
			  );
			}
		
		3.2 运行auto-tiler转换后的CNN内核代码，运行得到对应的输出，层层推理，最终获取输入图片特征值对应的预测值。

#8.3 HWCE

8.3.2 HWCE介绍
HWCE是一种专用的协助处理器，专为加速计算卷积累积内核而设计。特别是，它的目标是加速卷积神经网络（CNN）。 HWCE假设输入和输出像素，卷积权重是16位，8位或4位定点数。

与大多数紧密耦合的加速器不同，HWCE不与特定核心绑定，而是紧密集成在集群中。存储器访问直接执行到集群L1存储器的对数互连，而加速器的控制通过作为集群外围互连的目标的配置端口来执行。

硬件卷积引擎能够在一个周期内执行以下操作：

 1. 使用16位权重和16位像素的单个5x5或4x7卷积
 2. 使用16位权重和16位像素的三个同时3x3卷积
 3. 使用8位权重和16位像素的两个同时5x5或4x7卷积
 4. 使用4位权重和16位像素同时进行四次5x5或4x7卷积
此外，在所有考虑的情况下，权重的精度可以降低到8位或4位，像素的精度可以降低到8位; 这些变化并未带来性能提升，但功耗和内存带宽要求已经放宽。


8.3.1 HWCE的使用

基本内核可以是顺序的也可以是并行的。 顺序内核将在单个核心（群集的核心0）上运行。 顺序内核可以处理HWCE卷积加速器的配置。 并行内核应在群集的所有可用内核上运行。 上述MNIST示例使用了并行内核，因此未启动HWCE加速器，如若使用HWCE加速器，需要使用顺序内核函数开启。

构建顺序函数以打开HWCE加速器。内核函数如下：

	LibKernel("HWCE_Enable",  CALL_SEQUENTIAL, CArgs(0), "");

同时HWCE具有对应的CNN内核函数可供调用：
	
 - HWCE_ProcessOneTile3x3_MultiOut
 - HWCE_ProcessOneTile5x5
 - HWCE_ProcessOneTile7x4
 - HWCE_ProcessOneTile7x7
	
	