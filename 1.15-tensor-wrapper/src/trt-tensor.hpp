
#ifndef TRT_TENSOR_HPP
#define TRT_TENSOR_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "mix-memory.hpp"

struct CUstream_st;
typedef CUstream_st CUStreamRaw;
typedef CUStreamRaw* CUStream;

/*
    封装的 tensor 功能:
    1. 内存的管理, 可以使用 mixmemory 解决
    2. 内存的复用, 依然可以用 mixmemory 解决
    3. 内存的 copy, 比如说 cpu -> gpu, gpu -> cpu
        解决方案(从caffe上学到的思路):
        a. 定义内存的状态, 表示内存当前最新的内容在哪里(GPU/CPU/Init)
        b. 懒分配原则, 当你需要使用时, 才会考虑分配内存
        c. 获取内存地址, 即表示: 想拿到最新的数据. 
            比如说 tensor.cpu 表示我想拿到最新的数据, 并且把它放到 cpu 上
            比如说 tensor.gpu 表示我想拿到最新的数据, 并且把它放到 gpu 上
    4. 索引的计算, 比如说, 我有 5d tensor(B, D, C, H, W), 此时我要获取 B=1, D=3, C=0, H=5, W=9 的位置元素
        属于非常基础且非常频繁的一个能力
 */
 /**
 索引的计算原则
  shape/dim         index
     B                1
     D                3
     C                0
     H                5
     W                9
  position = 0
  for d, i in zip(dims, indexs):
        position *= d
        position += i
  // position = ((position * B + 1) * D + 3) * C + 0 ...
 */
namespace TRT{

    enum class DataHead : int{
        Init   = 0,
        Device = 1,
        Host   = 2
    };

    enum class DataType : int {
        Float = 0,
        Float16 = 1,
        Int32 = 2,
        UInt8 = 3
    };

    int data_type_size(DataType dt);
    const char* data_head_string(DataHead dh);
    const char* data_type_string(DataType dt);

    class Tensor {
    public:
        Tensor(const Tensor& other) = delete;
        Tensor& operator = (const Tensor& other) = delete;

        explicit Tensor(DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int n, int c, int h, int w, DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int ndims, const int* dims, DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(const std::vector<int>& dims, DataType dtype = DataType::Float, std::shared_ptr<MixMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        virtual ~Tensor();

        int numel() const;
        inline int ndims() const{return shape_.size();}
        inline int size(int index)  const{return shape_[index];}
        inline int shape(int index) const{return shape_[index];}

        inline int batch()   const{return shape_[0];}
        inline int channel() const{return shape_[1];}
        inline int height()  const{return shape_[2];}
        inline int width()   const{return shape_[3];}

        inline DataType type()                const { return dtype_; }
        inline const std::vector<int>& dims() const { return shape_; }
        inline const std::vector<size_t>& strides() const {return strides_;}
        inline int bytes()                    const { return bytes_; }
        inline int bytes(int start_axis)      const { return count(start_axis) * element_size(); }
        inline int element_size()             const { return data_type_size(dtype_); }
        inline DataHead head()                const { return head_; }

        std::shared_ptr<Tensor> clone() const;
        Tensor& release();
        Tensor& set_to(float value);
        bool empty() const;

        template<typename ... _Args>
        int offset(int index, _Args ... index_args) const{
            const int index_array[] = {index, index_args...};
            return offset_array(sizeof...(index_args) + 1, index_array);
        }

        int offset_array(const std::vector<int>& index) const;
        int offset_array(size_t size, const int* index_array) const;

        template<typename ... _Args>
        Tensor& resize(int dim_size, _Args ... dim_size_args){
            const int dim_size_array[] = {dim_size, dim_size_args...};
            return resize(sizeof...(dim_size_args) + 1, dim_size_array);
        }

        Tensor& resize(int ndims, const int* dims);
        Tensor& resize(const std::vector<int>& dims);
        Tensor& resize_single_dim(int idim, int size);
        int  count(int start_axis = 0) const;
        int device() const{return device_id_;}

        Tensor& to_gpu(bool copy=true);
        Tensor& to_cpu(bool copy=true);

        // 如果数据最新在host上, 则直接返回 cpu 地址
        // 如果数据最新在device上, 则先复制到host, 再返回cpu地址
        inline void* cpu() const { ((Tensor*)this)->to_cpu(); return data_->cpu(); }
        inline void* gpu() const { ((Tensor*)this)->to_gpu(); return data_->gpu(); }
        
        template<typename DType> inline const DType* cpu() const { return (DType*)cpu(); }
        template<typename DType> inline DType* cpu()             { return (DType*)cpu(); }

        template<typename DType, typename ... _Args> 
        inline DType* cpu(int i, _Args&& ... args) { return cpu<DType>() + offset(i, args...); }


        template<typename DType> inline const DType* gpu() const { return (DType*)gpu(); }
        template<typename DType> inline DType* gpu()             { return (DType*)gpu(); }

        template<typename DType, typename ... _Args> 
        inline DType* gpu(int i, _Args&& ... args) { return gpu<DType>() + offset(i, args...); }


        template<typename DType, typename ... _Args> 
        inline DType& at(int i, _Args&& ... args) { return *(cpu<DType>() + offset(i, args...)); }
        
        std::shared_ptr<MixMemory> get_data()             const {return data_;}
        std::shared_ptr<MixMemory> get_workspace()        const {return workspace_;}
        Tensor& set_workspace(std::shared_ptr<MixMemory> workspace) {workspace_ = workspace; return *this;}

        bool is_stream_owner() const {return stream_owner_;}
        CUStream get_stream() const{return stream_;}
        Tensor& set_stream(CUStream stream, bool owner=false){stream_ = stream; stream_owner_ = owner; return *this;}

        Tensor& set_mat     (int n, const cv::Mat& image);
        Tensor& set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]);
        cv::Mat at_mat(int n = 0, int c = 0) { return cv::Mat(height(), width(), CV_32F, cpu<float>(n, c)); }

        Tensor& synchronize();
        const char* shape_string() const{return shape_string_;}
        const char* descriptor() const;

        Tensor& copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id = CURRENT_DEVICE_ID);
        Tensor& copy_from_cpu(size_t offset, const void* src, size_t num_element);

        void reference_data(const std::vector<int>& shape, void* cpu_data, size_t cpu_size, void* gpu_data, size_t gpu_size, DataType dtype);

        /**
        
        # 以下代码是python中加载Tensor
        import numpy as np

        def load_tensor(file):
            
            with open(file, "rb") as f:
                binary_data = f.read()

            magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)
            assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."
            
            dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=3 * 4)

            if dtype == 0:
                np_dtype = np.float32
            elif dtype == 1:
                np_dtype = np.float16
            else:
                assert False, f"Unsupport dtype = {dtype}, can not convert to numpy dtype"
                
            return np.frombuffer(binary_data, np_dtype, offset=(ndims + 3) * 4).reshape(*dims)

            **/
        bool save_to_file(const std::string& file) const;
        bool load_from_file(const std::string& file);

    private:
        Tensor& compute_shape_string();
        Tensor& adajust_memory_by_update_dims_or_type();
        void setup_data(std::shared_ptr<MixMemory> data);

    private:
        std::vector<int> shape_;
        std::vector<size_t> strides_;
        size_t bytes_    = 0;
        DataHead head_   = DataHead::Init;
        DataType dtype_  = DataType::Float;
        CUStream stream_ = nullptr;
        bool stream_owner_ = false;
        int device_id_   = 0;
        char shape_string_[100];
        char descriptor_string_[100];
        std::shared_ptr<MixMemory> data_;
        std::shared_ptr<MixMemory> workspace_;
    };
}; // namespace TRT

#endif // TRT_TENSOR_HPP