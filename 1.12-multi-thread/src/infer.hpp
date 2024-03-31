#ifndef INFER_HPP
#define INFER_HPP

#include <memory>
#include <string>
#include <future>

// 总结, 原则:
// 1. 头文件, 尽量只包含需要的部分
// 2. 外界不需要的, 尽量不让他看到, 保持定义的简洁
// 3. 不要在头文件写 using namespace 这种
// 	但是可以在 cpp 中写 using namespace
// 	对于命名空间, 应当尽量少的展开 

// RAII + 接口模式	来实现前两条

////////////////////////////////////////////////////////
// 接口类, 他是一个纯虚类
// 原则是: 只暴露调用者需要的函数, 其他一概不暴露
// 比如说 load_model, 咱们通过RAII做了定义, 因此load_model属于不需要的范畴
// 内部如果有启动线程等等, start、stop, 也不需要暴露, 而是初始化的时候就自动启动, 都是RAII的定义

class InferInterface {
public:
    virtual std::shared_future<std::string> forward(std::string pic) = 0; // 纯虚函数

};
std::shared_ptr<InferInterface> create_infer(const std::string& file);

#endif // INFER_HPP
