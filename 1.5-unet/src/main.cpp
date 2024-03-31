// 目前找到的运行失败y原因: unet不支持tensorrt8
// tensorRT include
#include <NvInfer.h>
// onnx 解析器的头文件
#include <onnx-tensorrt-release-8.6/NvOnnxParser.h>
// 推理用的运行时头文件
#include <NvInferRuntime.h>
// cuda include
#include <cuda_runtime.h>
// system include
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

static vector<int> _classes_colors = {
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
    128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
    64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
};

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

// 通过智能指针管理nv返回的指针参数
// 内存自动释放, 避免泄漏
template<typename _T> shared_ptr<_T> make_nvshared(_T* ptr) {
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){
    #ifdef _WIN32
        return ::PathFileExistsA(path.c_str());
    #else
        return access(path.c_str(), R_OK) == 0;
    #endif
}

// 创建推理引擎
bool build_model(){
    if(exists("unet.trtmodel")){
        printf("unet.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    auto builder    = make_nvshared(nvinfer1::createInferBuilder(logger));  // 用来创建网络的构建器
    auto config     = make_nvshared(builder->createBuilderConfig());    // 网络配置
    auto network    = make_nvshared(builder->createNetworkV2(1));  // 创建一个网络, 显示指定bs
    auto parser     = make_nvshared(nvonnxparser::createParser(*network, logger));  // 将onnx转换成trtmodel的转换器

    if(!parser->parseFromFile("unet.onnx", 1)) {
        printf("Failed to parse unet.onnx \n");
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB \n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 相关配置文件
    auto profile        = builder->createOptimizationProfile();
    auto input_tensor   = network->getInput(0);
    auto input_dims     = input_tensor->getDimensions();

    // 最小/最佳允许batch
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    // 最大
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed. \n");
        return false;
    }

    // 将模型序列化, 并存储为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("unet.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done. \n");
    return true;
}

//////////////////////////////////////////////////////////////////////////////////
vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

// 后处理函数：从模型输出中提取概率和类别索引
// 参数:
// output - 指向模型输出数据的指针
// output_width, output_height - 输出图像的宽度和高度
// num_class - 类别数量
// ibatch - 批处理索引
static tuple<cv::Mat, cv::Mat> post_process(float* output, int output_width, int output_height, int num_class, int ibatch){
    // 创建两个Mat对象，一个用于保存每个像素的最高概率，另一个用于保存对应的类别索引
    cv::Mat output_prob(output_height, output_width, CV_32F); // 概率图
    cv::Mat output_index(output_height, output_width, CV_8U); // 索引图

    // 计算当前批次在输出数据中的起始位置
    float* pnet     = output + ibatch * output_width * output_height * num_class;
    // 获取指向output_prob第一行的指针
    float* prob     = output_prob.ptr<float>(0);
    // 获取指向output_index第一行的指针
    uint8_t* pidx   = output_index.ptr<uint8_t>(0);

    // 遍历输出图像的每个像素
    for(int k=0; k < output_prob.cols * output_prob.rows; ++k, pnet += num_class, ++prob, ++pidx){
        // 寻找具有最高概率的类别索引
        int ic = std::max_element(pnet, pnet + num_class) - pnet; // 此处修正了std::max_element的使用方式
        *prob  = pnet[ic]; // 保存这个像素的最高概率
        *pidx  = ic;       // 保存对应的类别索引
    }
    // 返回概率图和索引图
    return make_tuple(output_prob, output_index);
}

// 在图像上渲染分割结果
// 参数:
// image - 要渲染的图像
// prob - 每个像素点的概率矩阵
// iclass - 每个像素点的类别索引矩阵
static void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass) {
    auto pimage = image.ptr<cv::Vec3b>();  // 获取图像数据的指针
    auto pprob  = prob.ptr<float>(0);      // 获取概率矩阵的指针
    auto pclass = iclass.ptr<uint8_t>(0);  // 获取类别索引矩阵的指针

    // 遍历图像的每一个像素
    for(int i=0; i < image.cols * image.rows; ++i, ++pimage, ++pprob, ++pclass) {
        int iclass          = *pclass;      // 当前像素的类别索引
        float probability   = *pprob;       // 当前像素的概率
        auto& pixel         = *pimage;      // 当前像素的颜色值
        // 计算前景色的强度，基于概率调整，范围限制在0.6到0.8之间
        float foreground    = min(0.6f + probability * 0.2f, 0.8f);
        // 背景色强度为1减去前景色强度
        float background    = 1 - foreground;
        
        // 对每个颜色通道应用颜色混合
        for(int c = 0; c < 3; ++c) {
            // 混合像素颜色和类别颜色
            auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2 - c];
            // 保证颜色值在0到255之间
            pixel[c] = min((int)value, 255);
        }
    }
}


int inference() {
    TRTLogger logger;

    auto engine_data    = load_file("unet.trtmodel");
    auto runtime        = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine         = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    std::cout <<( engine_data.empty() == true ? "空的" : "不空") << std::endl;
    std::cout << "类型: " << typeid(engine_data.data()).name() << std::endl;
    printf("%s %d\n", engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return -1;
    }

    if(engine->getNbBindings() != 2){
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbBindings() - 1);
        return -1;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 512;
    int input_width = 512;
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

     ///////////////////////////////////////////////////
    // letter box
    auto image = cv::imread("street.jpg");
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
    i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image.jpg", input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims   = engine->getBindingDimensions(1);
    int output_height  = output_dims.d[1];
    int output_width   = output_dims.d[2];
    int num_classes    = output_dims.d[3];
    int output_numel = input_batch * output_height * output_width * num_classes;
    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    cv::Mat prob, iclass;
    tie(prob, iclass) = post_process(output_data_host, output_width, output_height, num_classes, 0);
    cv::warpAffine(prob, prob, m2x3_d2i, image.size(), cv::INTER_LINEAR);
    cv::warpAffine(iclass, iclass, m2x3_d2i, image.size(), cv::INTER_NEAREST);
    render(image, prob, iclass);

    printf("Done, Save to image-draw.jpg\n");
    cv::imwrite("image-draw.jpg", image);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}

