链接：[ncnn](https://github.com/Tencent/ncnn)

# 1.主要特点
　　
   ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。ncnn从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台，手机端cpu的速度快于目前所有已知的开源框架。基于ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，开发出人工智能APP。

**支持体系架构优化：**

> 1. X86，针对MMX/SSE/AVX系列指令
> 2. ARM，针对NEON指令

**支持的网络：**

> 1. Classical CNN: VGG AlexNet GoogleNet Inception ...
> 2. Practical CNN: ResNet DenseNet SENet FPN ...
> 3. Light-weight CNN: SqueezeNet MobileNetV1/V2 ShuffleNetV1/V2 MNasNet ...
> 4. Detection: MTCNN facedetection ...
> 5. Detection: VGG-SSD MobileNet-SSD SqueezeNet-SSD MobileNetV2-SSDLite ...
> 6. Detection: Faster-RCNN R-FCN ...
> 7. Detection: YOLOV2 YOLOV3 MobileNet-YOLOV3 ...
> 8. Segmentation: FCN PSPNet ...

**主要技术特点：**

> 1. 支持卷积神经网络，支持多输入和多分支结构，可计算部分分支
> 2. 无任何第三方库依赖，不依赖 BLAS/NNPACK 等计算框架
> 3. 纯C++实现，跨平台，支持 android ios 等
> 4. ARM NEON 汇编级优化，计算速度极快
> 5. 精细的内存管理和数据结构设计，内存占用极低
> 6. 支持多核并行计算加速，ARM big.LITTLE cpu 调度优化
> 7. 整体库体积小于500K，并可轻松精简到小于300K
> 8. 可扩展的模型设计，支持 8bit 量化和半精度浮点存储，可导入caffe/pytorch/mxnet/onnx 模型
> 9. 支持直接内存零拷贝引用加载网络模型
> 10. 可注册自定义层实现并扩展

# 2 架构分析
## 2.1 代码目录分析
目录结构：

    ├── benchmark
    ├── examples
    │   └── squeezencnn
    │       ├── assets
    │       ├── jni
    │       ├── res
    │       │   ├── layout
    │       │   └── values
    │       └── src
    │           └── com
    │               └── tencent
    │                   └── squeezencnn
    ├── images
    ├── src
    │   └── layer
    │       ├── arm
    │       └── x86
    ├── toolchains
    └── tools
        ├── caffe
        ├── darknet
        ├── mxnet
        ├── onnx
        ├── plugin
        ├── pytorch
        └── tensorflow

- **benchmark目录**

  使用benchncnn，并加载网络的param文件，计算网络的benchmark。如果要计算网络逐层的性能，需要在顶层CMakeLists.txt中打开NCNN_BENCHMARK选项。

- **examples目录**

  包含squeezenet、fasterrcnn、rfcn、yolov2、yolov3、mobilenetv2ssdlite、mobilenetssd、squeezenetssd、shufflenetv2的运行实例。通过调用libncnn库完成神经网路计算。

 > 注意：本部分依赖OpenCV，至少包含其core、highgui、imgproc、imgcodecs组件

- **src目录**

  ncnn库的源码，包括网络处理、OpenCV对接、模型处理、layer管理、内存申请等。
layer目录存放算子级别实现，并且分别对x86和ARM进行优化。

 **ARM单独优化算子：**
  ```
1. absval
2. batchnorm
3. bias
4. clip
5. conv，包括1x1、2x2、3x3、4x4、5x5、7x7及8bit卷积核
6. conv_depthwise，包括3x3、5x5及8bit卷积核
7. deconv，包括3x3、4x4卷积核
8. deconv_depthwise
9. dequantize
10.eltwise
11.innerproduct
12.lrn
13.pooling, 2x2、3x3
14.prelu
15.scale
16.sigmoid
17.softmax
  ```

 **x86单独优化算子：**
   ```
conv，包括1x1、3x3、5x5及8bit卷积核
conv_depthwise，包括3x3及8bit卷积核
   ```

- **toolchains目录**

  包含各种x86和ARM体系架构的编译工具链配置文件。

- **tools目录**

  提供caffe、darknet、mxnet、onnx、pytorch、tensorflow的模型转换工具。还包括模型dump、VS图像插件等杂项工具。

## 2.2 yolov3模型执行过程分析

```sequenceDiagram
participant main
participant detect_yolov3
participant extract
participant forward_layer
participant draw_objects
main->main: 加载待检测图片
main->detect_yolov3: image cv::Mat
detect_yolov3->detect_yolov3:  加载yolov3的param和bin文件
detect_yolov3->detect_yolov3:  图片resize
detect_yolov3->detect_yolov3:  create_extractor
detect_yolov3->extract: 计算图last layer name
extract->extract: 计算last layer index
extract->forward_layer: \nlast_index,opt,\nblob_mats(保存layers计算结果)
forward_layer->forward_layer: 核心函数:\n从last layer开始，\n递归遍历计算图的每一个layer，\n对layer做前向推理计算
forward_layer-->extract: last layer推理结果Mat
extract-->detect_yolov3: last layer推理结果Mat
detect_yolov3->detect_yolov3: 根据Mat,\n计算每个推理结果的\n类别、概率、检测框位置
detect_yolov3-->main: objects推理结果
main->draw_objects: objects推理结果
draw_objects->draw_objects:使用opencv在图片中\n输出检测信息、画检测框
draw_objects-->main: 显示处理后图片
```

　　ncnn中的Extractor可以看做是Network对用户的接口。Network一般单模型只需要一个实例，而Extractor可以有多个实例。这样做的好处是进行多个任务的时候可以节省内存，即模型定义模型参数等不会产生多个拷贝。

# 3 重点优化技术
## 3.1 体积与兼容性优化
**1. 无第三方依赖库设计**

- ncnn不依赖任何第三方库，完全独立实现所有计算过程，不需要BLAS/NNPACK等数学计算库。

**2. C++和跨平台设计**

- ncnn代码全部使用C/C++实现，以及跨平台的cmake编译系统，可在已知的绝大多数平台编译运行，如Linux、Windows、MacOS、Android、iOS 等。
- 由于 ncnn 不依赖第三方库，且采用 C++ 03 标准实现，只用到了 std::vector 和 std::string 两个 STL 模板，可轻松移植到其他系统和设备上。

**3. 小体积设计**

- ncnn自身没有依赖项，默认编译选项下的库体积小于500K。
- ncnn在编译时可自定义是否需要文件加载和字符串输出功能，还可自定义去除不需要的层实现，精简到小于300K。
- 使用半精度浮点和8bit量化数，进一步减少模型体积。

![][1]

**4. 自定义layer**

　　ncnn提供了注册自定义layer的扩展实现，可以将自定义的特殊layer内嵌到网络推理过程中，组合出可自定义的网络结构。

## 3.2 计算优化
**1. OpenMP多核加速**

　　[OpenMP](https://zh.wikipedia.org/wiki/OpenMP)是一套跨平台的共享内存方式的多线程并发编程API，在大部分平台中都有实现，包括Windows、Linux、Android、IOS。使用OpenMP加速只需要在串行代码中添加编译指令以及少量API即可。

　　例如OpenMP实现多核加速，只需加入一条编译指令:
```
#pragma omp parallel for
void add(const int* a, const int* b, int* c, const int len)
{
  for(int i=0; i<len; i++)
  {
    c = a[i] + b[i];
  }
}
```

**2. SIMD指令加速**

　　SIMD即单指令多数据指令，目前在x86平台下有MMX/SSE/AVX系列指令，ARM平台下有NEON指令。一般SIMD指令通过intrinsics或者汇编实现。以下以NEON指令为例，通过调用vld1q_f32，vmulq_f32，vst1q_f32接口实现加速。

　　正常的vector相乘：
```
static void normal_vector_mul(const std::vector<float>& vec_a,
                              const std::vector<float>& vec_b,
                              std::vector<float>& vec_result) {
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	//no optimized
	for (size_t i = 0; i < vec_result.size(); i++) {
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}
```

　　NEON优化的vector相乘：
```
static void neon_vector_mul(const std::vector<float>& vec_a,
                            const std::vector<float>& vec_b,
                            std::vector<float>& vec_result) {
	assert(vec_a.size() == vec_b.size());
	assert(vec_a.size() == vec_result.size());
	int i = 0;
	//neon process
	for (; i < (int)vec_result.size() - 3 ; i += 4) {
		const auto data_a = vld1q_f32(&vec_a[i]);
		const auto data_b = vld1q_f32(&vec_b[i]);
		float* dst_ptr = &vec_result[i];
		const auto data_res = vmulq_f32(data_a, data_b);
		vst1q_f32(dst_ptr, data_res);
	}
	//no optimized for the rest of vector
	for (; i < (int)vec_result.size(); i++) {
		vec_result[i] = vec_a[i] * vec_b[i];
	}
}
```

　　NEON的自动向量优化是将代码中的计算自动扩展到４份向量上，４份向量执行相同的指令，所谓“单指令多数据”。使用NEON优化后，vector相乘操作加速在３倍左右。

**3. 计算cache优化**

　　缓存对于高速计算是非常重要的一环，通过合理的安排内存读写，能非常有效的加速计算。

　　- 矩阵计算cache加速:
　　![此处输入图片的描述][2]
```
static void gemm_v1(float* matA, float* matB, float* matC,
                    const int M, const int N, const int K) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			float sum = 0.0f;
			for (int h = 0; h < K; h++) {
				sum += matA[j * M + h] * matB[h * K + i];
			}
			matC[j*M + i] = sum;
		}
	}
}
```

```
static void gemm_v2(float* matA, float* matB, float* matC,
                    const int M, const int N, const int K) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float sum = 0.0f;
			for (int h = 0; h < K; h++) {
				sum += matA[i * M + h] * matB[h * K + j];
			}
			matC[i * M + j] = sum;
		}
	}
}
```

　　如果矩阵以列为方向进行访问，则会出现cache不命中的情况，导致性能下降。gemm_v1与gemm_v2相比，其matC的访问方向为列访问。所以在数据量较大的情况下，gemm_v1速度会下降很多。而在gemm_v2中则只有matB发生较多cache不命中，而这是gemm计算无法避免的。ncnn的矩阵计算中非常注意将矩阵列访问情况降到最低。

　　- 卷积计算cache加速：

　　在卷积运算中，局部感受野以一定step进行平移，在平移过程中经常会复用上一个计算的内存。比如以卷积计算conv3x3_s1为例，每次从matA同时访问4行（一般一次3x3卷积只需要访问3行），由于step是1，所以可以同时生成2行的卷积结果。可以看到有2行数据直接共用了，缓存利用率得到极大提高。
![此处输入图片的描述][3]
![此处输入图片的描述][4]

**4. 卷积计算量削减**

![此处输入图片的描述][5]　　![此处输入图片的描述][6]

　　计算时可以依据需求，先计算公共部分和 prob 分支，待 prob 结果超过阈值后，再计算 bbox 分支。如果 prob 低于阈值，则可以不计算 bbox 分支，减少计算量。

## 3.3 内存优化

**1. 数据结构优化**

 - 在卷积层、全连接层等计算量较大的层实现中，没有采用通常框架中的im2col + 矩阵乘法，因为这种方式会构造出非常大的矩阵，消耗大量内存。因此，ncnn 采用原始的滑动窗口卷积实现，并在此基础上进行优化，大幅节省了内存。
 - 前向网络计算过程中，ncnn 可自动释放中间结果所占用的内存，进一步减少内存占用。

![此处输入图片的描述][7]

**2. 直接内存引用**

　　在某些特定应用场景中，如因平台层 API 只能以内存形式访问模型资源，或者希望将模型本身作为静态数据写在代码里，ncnn 提供了直接从内存引用方式加载网络模型。这种加载方式不会拷贝已在内存中的模型，也无需将模型先写入实体的文件再读入，效率极高。


  [1]: https://ask.qcloudimg.com/raw/i570q2zx2h.png?imageView2/2/w/1620
  [2]: http://xylcbd.coding.me/2017/09/02/ncnn-analysis/gemm.png
  [3]: http://xylcbd.coding.me/2017/09/02/ncnn-analysis/gemm_row0_col0.png
  [4]: http://xylcbd.coding.me/2017/09/02/ncnn-analysis/gemm_row1_col0.png
  [5]: https://ask.qcloudimg.com/raw/z4fzwcjkqv.png?imageView2/2/w/1620
  [6]: https://ask.qcloudimg.com/raw/qamx8vdd1r.png?imageView2/2/w/1620
  [7]: https://ask.qcloudimg.com/raw/a433gguwyx.png?imageView2/2/w/1620
