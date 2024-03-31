#include <tensorRT/common/cuda_tools.hpp> // 包含一些CUDA工具函数的头文件



// 代码包含了YOLO模型在CUDA上运行的几个关键步骤：解码预测的边界框，应用逆仿射变换，计算边界框的交并比（IoU），以及执行非极大值抑制（NMS）以筛选最终的检测结果。这些步骤都是在GPU上并行执行的，以提高处理速度。
namespace Yolo{

    const int NUM_BOX_ELEMENT = 7;      // 每个边界框的元素数量：左、上、右、下、置信度、类别、保留标志
    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        // 使用仿射变换矩阵变换坐标
        *ox = matrix[0] * x + matrix[1] * y + matrix[2]; // 计算变换后的x坐标
        *oy = matrix[3] * x + matrix[4] * y + matrix[5]; // 计算变换后的y坐标
    }

    static __global__ void decode_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects){  
        // 解码内核，将预测的边界框转换为最终的边界框坐标和置信度

        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes) return; // 如果线程索引超出边界框数量，则直接返回

        float* pitem     = predict + (5 + num_classes) * position;
        float objectness = pitem[4]; // 对象性评分
        if(objectness < confidence_threshold)
            return; // 如果对象性评分低于阈值，则忽略此边界框

        float* class_confidence = pitem + 5;
        float confidence        = *class_confidence++;
        int label               = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){ // 遍历所有类别，找到置信度最高的类别
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label      = i;
            }
        }

        confidence *= objectness; // 结合对象性评分和类别置信度
        if(confidence < confidence_threshold)
            return; // 如果组合置信度低于阈值，则忽略此边界框

        int index = atomicAdd(parray, 1); // 原子操作增加计数器，获取当前边界框的索引
        if(index >= max_objects)
            return; // 如果索引超出最大对象数，则直接返回

        // 提取边界框的中心坐标、宽度和高度
        float cx         = *pitem++;
        float cy         = *pitem++;
        float width      = *pitem++;
        float height     = *pitem++;
        // 计算边界框的左上角和右下角坐标
        float left   = cx - width * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        // 应用逆仿射变换
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        // 将解码后的边界框信息写入输出数组
        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
    }

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){
        // 计算两个边界框的交并比（IoU）

        float cleft 	= max(aleft, bleft); // 计算相交矩形的左边界
        float ctop 		= max(atop, btop);   // 计算相交矩形的上边界
        float cright 	= min(aright, bright); // 计算相交矩形的右边界
        float cbottom 	= min(abottom, bbottom); // 计算相交矩形的下边界
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f); // 计算
        if (c_area == 0.0f)   // 如果相交面积为0，则IoU为0
            return 0.0f;

        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop); // 计算边界框A的面积
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop); // 计算边界框B的面积
        return c_area / (a_area + b_area - c_area); // 计算并返回IoU
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){
        // 非极大值抑制（NMS）内核，用于筛选最终的边界框

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count) 
            return; // 如果线程索引超出边界框数量，则直接返回
        
        // left, top, right, bottom, confidence, class, keepflag
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue; // 跳过当前边界框和不同类别的边界框

            if(pitem[4] >= pcurrent[4]){ // 如果另一个边界框的置信度更高
                if(pitem[4] == pcurrent[4] && i < position)
                    continue; // 如果置信度相同，则保留先前的边界框

                float iou = box_iou( // 计算两个边界框的IoU
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){ // 如果IoU超过阈值
                    pcurrent[6] = 0;  // 将当前边界框的保留标志设置为0（忽略）
                    return;
                }
            }
        }
    } 

    void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){
        // 解码内核的调用函数
        
        auto grid = CUDATools::grid_dims(num_bboxes); // 计算网格维度
        auto block = CUDATools::block_dims(num_bboxes); // 计算块维度
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, max_objects)); // 启动解码内核
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
        // NMS内核的调用函数
        
        auto grid = CUDATools::grid_dims(max_objects); // 计算网格维度
        auto block = CUDATools::block_dims(max_objects); // 计算块维度
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold)); // 启动NMS内核
    }
};