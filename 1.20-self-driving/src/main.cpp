// tensorRT include
// 编译用头文件
#include <NvInfer.h>

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

#include <tensorRT/common/ilogger.hpp>
#include <tensorRT/builder/trt_builder.hpp>
#include <app_yolo/yolo.hpp>
#include <app_road/road.hpp>
#include <app_ldrn/ldrn.hpp>
#include <app_lane/lane.hpp>

using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// 上一节的代码
static bool build_model(){

    bool success = true;
    if(!exists("yolov5s.trtmodel"))
        // 当前实现假设TRT::compile会返回一个布尔值，指示编译的成功。确保此函数实现能适当地处理错误，比如通过捕获可能由底层TensorRT API或文件系统操作抛出的异常
        success = success && TRT::compile(TRT::Mode::FP32, 5, "yolov5s.onnx", "yolov5s.trtmodel");

    if(!exists("road-segmentation-adas.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "road-segmentation-adas.onnx", "road-segmentation-adas.trtmodel");
    
    if(!exists("ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx", "ldrn_kitti_resnext101_pretrained_data_grad_256x512.trtmodel");

    if(!exists("new-lane.trtmodel"))
        success = success && TRT::compile(TRT::Mode::FP32, 5, "new-lane.onnx", "new-lane.trtmodel");
    return success;
}

// 将深度图像转换为易于理解和分析的彩色图像
static cv::Mat to_render_depth(const cv::Mat& depth){
    cv::Mat mask;
    // 将深度图转换为8位无符号整型，增强对比度以便更好地观察
    depth.convertTo(mask, CV_8U, -5, 255);
    // 可以选择性地裁剪图像的上部18%，以便专注于图像的特定区域
    // mask = mask(cv::Rect(0, mask.rows * 0.18, mask.cols, mask.rows * (1 - 0.18)));
    // 应用颜色映射来增强可视化效果，这里使用的是PLASMA颜色图
    cv::applyColorMap(mask, mask, cv::COLORMAP_PLASMA);
    return mask; // 返回处理后的图像
}

// 通过在不同位置展示不同的视觉信息（如原始视图、道路视图和深度视图），实现了图像的合并展示，
// 适用于将多个视角或分析结果集成在一个单一的视图中进行比较或展示
static void merge_images(
    const cv::Mat& image, const cv::Mat& road, 
    const cv::Mat& depth, cv::Mat& scene
){
    // 将原始图像复制到场景图像的指定位置
    image.copyTo(scene(cv::Rect(0, 0, image.cols, image.rows)));

    // 裁剪道路图像的下半部分
    auto road_crop = road(cv::Rect(0, road.rows * 0.5, road.cols, road.rows * 0.5));
    // 将裁剪后的道路图像复制到场景图像的指定位置
    road_crop.copyTo(scene(cv::Rect(0, image.rows, road_crop.cols, road_crop.rows)));

    // 错误的裁剪使用了错误的变量 'road' 应该使用 'depth'
    // 裁剪深度图像的部分区域，去除上面的18%，只保留下方的82%
    auto depth_crop = depth(cv::Rect(0, depth.rows * 0.18, depth.cols, depth.rows * (1 - 0.18)));
    // 将裁剪后的深度图像复制到场景图像的指定位置
    depth_crop.copyTo(scene(cv::Rect(image.cols, image.rows * 0.25, depth_crop.cols, depth_crop.rows)));
}

// 处理视频文件, 并将对象检测、道路分割、深度估计和车道检测的结果综合展示在一个视频输出中
static void inference(){
    // 创建不同类型的推理引擎
    // auto image = cv::imread("imgs/dashcam_00.jpg");
    auto yolov5 = Yolo::create_infer("yolov5s.trtmodel", Yolo::Type::V5, 0, 0.25, 0.45);
    auto road   = Road::create_infer("road-segmentation-adas.trtmodel", 0);
    auto ldrn   = Ldrn::create_infer("ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx", 0);
    auto lane   = Lane::create_infer("new-lane.onnx", 0);

    cv::Mat image, scence;
    // 打开视频文件
    cv::VideoCapture cap("4k-tokyo-drive-thru-ikebukuro.mp4");
    // 获取视频的帧率、宽度和高度
    float fps   = cap.get(cv::CAP_PROP_FPS);
    int width   = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height  = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // 初始化场景图像的尺寸, 宽度为原始图像的两倍, 高度为1.5倍
    scence = cv::Mat(height * 1.5, width * 2, CV_8UC3, cv::Scalar::all(0));
    // 设置视频写入器, 用于输出结果视频
    // cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M', 'P', 'G', '2'), fps, scence.size());
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, scence.size());
    //auto scence = cv::Mat(image.rows * 1.5, image.cols * 2, CV_8UC3, cv::Scalar::all(0));

    // 循环读取视频帧
    while(cap.read(image)){
        // 并行执行不同的模型推理
        auto roadmask_fut   = road->commit(image);
        auto boxes_fut      = yolov5->commit(image);
        auto depth_fut      = ldrn->commit(image);
        auto point_fut      = lane->commit(image);
        // 获取推理结果
        auto roadmask       = roadmask_fut.get();
        auto boxes  = boxes_fut.get();
        auto depth  = depth_fut.get();
        auto points = point_fut.get();
        // 调整深度图和道路分割图的尺寸以匹配原始图像
        cv::resize(depth, depth, image.size());
        cv::resize(roadmask, roadmask, image.size());

        // 处理并显示检测到的对象
        for (auto& box: boxes) {
            int cx  = (box.left + box.right) * 0.5 + 0.5;
            int cy  = (box.top + box.bottom) * 0.5 + 0.5;
            float distance = depth.at<float>(cy, cx) / 5;
            if(fabs(cx - (image.cols * 0.5)) <= 200 && cy >= image.rows * 0.85)
                continue;

            cv::Scalar color(0, 255, 0);
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 3);

            auto name       = cocolabels[box.class_label];
            auto caption    = cv::format("%s %.2f", name, distance);
            int text_width  = cv::getTextSize(caption, 0, 0.5, 1, nullptr).width + 10;
            cv::rectangle(image, cv::Point(box.left-3, box.top-20), cv::Point(box.left + text_width, box.top), color, -1);
            cv::putText(image, caption, cv::Point(box.left, box.top-5), 0, 0.5, cv::Scalar::all(0), 1, 16);
        }

        // 为车道检测的点设置颜色
        cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 0, 255),
            cv::Scalar(255, 0, 0)
        };
        for (int i =0; i < 18; ++i){
            for (int j = 0; j < 4; ++j){
                auto& p = points[i*4 + j];
                if(p.x > 0){
                    auto color = colors[j];
                    cv::circle(image, p, 5, color, -1, 16);
                }
            }
        }

        // 合并图像
        merge_images(image, roadmask, to_render_depth(depth), scence);
        // cv::imwrite("merge.jpg", scence);
        // 写入处理后的帧到输出视频
        writer.write(scence);
        INFO("Process");    // 需要定义INFO或者改为适当的日志或打印函数
    }
    // 释放资源
    writer.release();
}


int main() {

    // 新的实现
    if (!build_model()) {
        return -1;
    }
    inference();
    return 0;
}
