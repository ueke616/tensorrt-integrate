

#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#include "builder/trt_builder.hpp"
#include "app_yolo/yolo.hpp"
#include "pybind11.hpp"

using namespace std;
namespace py = pybind11;

// YoloInfer类封装了TensorRT YOLO推理引擎的功能
class YoloInfer { 
public:
    // 构造函数，初始化YOLO推理实例
	YoloInfer(
		string engine, // TensorRT引擎文件路径
		Yolo::Type type, // YOLO模型类型
		int device_id, // GPU设备ID
		float confidence_threshold, // 置信度阈值
		float nms_threshold, // NMS阈值
		Yolo::NMSMethod nms_method, // NMS方法
		int max_objects, // 最大检测对象数
		bool use_multi_preprocess_stream // 是否使用多预处理流
	){
		instance_ = Yolo::create_infer(
			engine, 
			type,
			device_id,
			confidence_threshold,
			nms_threshold,
			nms_method, 
			max_objects, 
			use_multi_preprocess_stream
		);
	}

    // 检查YOLO推理实例是否有效
	bool valid(){
		return instance_ != nullptr;
	}

    // 提交一个图像给YOLO模型并返回一个包含检测结果的future
	shared_future<ObjectDetector::BoxArray> commit(const py::array& image){
		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image must be owner, slice is unsupported, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

private:
    // 内部持有的YOLO推理实例
	shared_ptr<Yolo::Infer> instance_;
}; 

// 编译TensorRT模型的函数
bool compileTRT(
    int max_batch_size, // 最大批次大小
    string source, // ONNX模型文件路径
    string output, // 输出的TensorRT模型文件路径
    bool fp16, // 是否使用FP16精度
    int device_id, // GPU设备ID
    int max_workspace_size // 最大工作空间大小
){
    TRT::set_device(device_id);
    return TRT::compile(
        fp16 ? TRT::Mode::FP16 : TRT::Mode::FP32, // 根据fp16参数选择模式
        max_batch_size, 
        source, 
        output, 
        {}, 
        nullptr, 
        "", 
        "", 
        max_workspace_size
    );
}

// pybind11 模块定义
PYBIND11_MODULE(yolo, m){

    // 定义 python 中的 ObjectBox 类, 用于表示检测到的对象
    py::class_<ObjectDetector::Box>(m, "ObjectBox")
		.def_property("left",        [](ObjectDetector::Box& self){return self.left;}, [](ObjectDetector::Box& self, float nv){self.left = nv;})
		.def_property("top",         [](ObjectDetector::Box& self){return self.top;}, [](ObjectDetector::Box& self, float nv){self.top = nv;})
		.def_property("right",       [](ObjectDetector::Box& self){return self.right;}, [](ObjectDetector::Box& self, float nv){self.right = nv;})
		.def_property("bottom",      [](ObjectDetector::Box& self){return self.bottom;}, [](ObjectDetector::Box& self, float nv){self.bottom = nv;})
		.def_property("confidence",  [](ObjectDetector::Box& self){return self.confidence;}, [](ObjectDetector::Box& self, float nv){self.confidence = nv;})
		.def_property("class_label", [](ObjectDetector::Box& self){return self.class_label;}, [](ObjectDetector::Box& self, int nv){self.class_label = nv;})
		.def_property_readonly("width", [](ObjectDetector::Box& self){return self.right - self.left;})
		.def_property_readonly("height", [](ObjectDetector::Box& self){return self.bottom - self.top;})
		.def_property_readonly("cx", [](ObjectDetector::Box& self){return (self.left + self.right) / 2;})
		.def_property_readonly("cy", [](ObjectDetector::Box& self){return (self.top + self.bottom) / 2;})
		.def("__repr__", [](ObjectDetector::Box& obj){
			return iLogger::format(
				"<Box: left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, class_label=%d, confidence=%.5f>",
				obj.left, obj.top, obj.right, obj.bottom, obj.class_label, obj.confidence
			);	
		});

    // 定义 SharedFutureObjectBoxArray 类, 用于处理异步检测结果
    py::class_<shared_future<ObjectDetector::BoxArray>>(m, "SharedFutureObjectBoxArray")
		.def("get", &shared_future<ObjectDetector::BoxArray>::get);

    // 将YOLO的Type枚举到Python中
    py::enum_<Yolo::Type>(m, "YoloType")
		.value("V5", Yolo::Type::V5)
		.value("V3", Yolo::Type::V3)
		.value("X", Yolo::Type::X);

    // 将YOLO的NMSMethod枚举绑定到 Python 中
	py::enum_<Yolo::NMSMethod>(m, "NMSMethod")
		.value("CPU",     Yolo::NMSMethod::CPU)
		.value("FastGPU", Yolo::NMSMethod::FastGPU);

    // 定义 Python 中的 Yolo 类, 对应 C++ 中的 YoloInfer 类
    py::class_<YoloInfer>(m, "Yolo")
		.def(py::init<string, Yolo::Type, int, float, float, Yolo::NMSMethod, int, bool>(), 
			py::arg("engine"), 
			py::arg("type")                 = Yolo::Type::V5, 
			py::arg("device_id")            = 0, 
			py::arg("confidence_threshold") = 0.4f,
			py::arg("nms_threshold") = 0.5f,
			py::arg("nms_method")    = Yolo::NMSMethod::FastGPU,
			py::arg("max_objects")   = 1024,
			py::arg("use_multi_preprocess_stream") = false
		)
		.def_property_readonly("valid", &YoloInfer::valid, "Infer is valid")
		.def("commit", &YoloInfer::commit, py::arg("image"));

    // 定义compileTRT函数, 用于编译TensorRT模型
    m.def(
		"compileTRT", compileTRT,
		py::arg("max_batch_size"),
		py::arg("source"),
		py::arg("output"),
		py::arg("fp16")                         = false,
		py::arg("device_id")                    = 0,
		py::arg("max_workspace_size")           = 1ul << 28
	);
}