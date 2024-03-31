#include <infer.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <future>
#include <vector>
#include <chrono>

using namespace std;

// RAII + 接口模式
// 提出问题
// 正常工作代码中,y异常逻辑需要耗费大量时间
// 异常会导致的使用复杂度变高, 编写复杂度变高

// RAII     -> 资源获取即初始化
// 接口模式  -> 设计模式, 是一种封装模式. 实现类与 接口类分离的模式

// RAII优势: 自动资源管理, 异常安全, 简化资源管理代码
struct Job{
	shared_ptr<promise<string>> pro;
	string input;
};

class InferImpl: public InferInterface{
public:
    virtual ~InferImpl(){
    	worker_running_ = false;
	cv_.notify_one(); // 唤醒下一个

	if(worker_thread_.joinable())
		worker_thread_.join();
    }
    
    bool load_model(const string& file){
        // 处理异常逻辑代码
//        if(!context_.empty()){
//            destroy();
//        }
	// 尽量保证资源在哪里分配就在那里释放, 这样能够使得程序足够简单, 而不是太乱
	// context_ = file;	// 加载
	// 线程内传递返回值的问题
	promise<bool> pro;
	worker_running_ = true;
	worker_thread_ = thread(&InferImpl::worker, this, file, std::ref(pro));	// 推理
        return pro.get_future().get();
	// return !context_.empty();
    }

    virtual shared_future<string> forward(string pic) override {
        // 异常逻辑
//        if(context_.empty()){
//            // 说名模型没有加载上
//            // 咱们对异常处理情况的定义很恼火
//            printf("模型没有加载. \n");
//            return;
//        }
        // 正常逻辑
        // printf("使用 %s 进行推理 \n", context_.c_str());
	// 往队列抛任务
	Job job;
	job.pro.reset(new promise<string>());
	job.input = pic;
	
	lock_guard<mutex> l(job_lock_);
	qjobs_.push(job);

	// detection push
	// face push
	// feature push
	// 一次等待结果, 实际上就是让 detection + face + feature 让他们并发执行

	// 被动通知, 一旦有新的任务需要推理, 通知我即可
	// 无法发送通知的家伙
	cv_.notify_one();
	return job.pro->get_future();
    }

    // 实际执行模型推理的部分
    void worker(string file, promise<bool>& pro) {
    	// worker 内实现, 模型的加载\使用\释放
	// 这样能保证资源在一个线程上, 线程在初始化后能持续使用, 管理方便
	string context_ = file;
	if(context_.empty()){
		pro.set_value(false);
		return;  // file加载失败
	} else {
		pro.set_value(true);
	}

	int max_batch_size = 5;
	vector<Job> jobs;
	int batch_id = 0;
	while(worker_running_){
		// 等待接受的家伙
		// 在队列任务并执行的过程
		unique_lock<mutex> l(job_lock_);
		cv_.wait(l, [&](){
			// true 退出等待
			// false 继续等待
			return !qjobs_.empty() || !worker_running_;
		});

		// 是因为程序发送终止信号而退出 wait 的
		if(!worker_running_){
			break;
		}

		while(jobs.size() < max_batch_size && !qjobs_.empty()) {
			jobs.emplace_back(qjobs_.front());
                        qjobs_.pop();	

			// 可以在这里, 一次拿一批出来, 最大拿maxbatchsize个job进行一次性处理
			// jobs inference => batch inference
		}
		// 执行 batch 推理
		for(int i=0; i < jobs.size(); ++i){
			auto& job = jobs[i];
			char result[100];
	        	sprintf(result, "%s :batch->%d[%d]", job.input.c_str(), batch_id, jobs.size());
			
			job.pro->set_value(result);
		}
		batch_id++;
		jobs.clear();
		this_thread::sleep_for(chrono::milliseconds(1000));
	}

	// 释放模型
	printf("释放: %s \n", context_.c_str());
	context_.clear();
	printf("Worker done. \n");
    } 

    // void destroy(){
    //     context_.clear();
    // }    
private:
    atomic<bool> worker_running_{false};
    thread worker_thread_;
    queue<Job> qjobs_;
    mutex job_lock_;
    condition_variable cv_;
    // string context_;
};

// RAII
// 属于一种约定,
// 获取infer实例, 即表示加载模型
// 加载模型失败, 则表示资源获取失败, 他们强绑定
// 加载模型成功, 则资源获取成功
// 1. 避免外部执行 load_model, 永远只有在这一个地方执行load_model,b不可能出现在其它地方(RAII没有完全限制, 只z做y一部分)
// 2. 一个实例的 load_model 不会 执行 超过 1 次
// 3. 获取的模型一定 初始化 成功, 因此 forward 函数, 不必判断模型是否加载成功
// load_model 中可以删掉对于重复load 的判断
// forward 函数中, 可以删除, 对是否加载成功的判断
// 接口模式
// 1. 解决 load_model 还能被外部看到的问题, 拒绝外面调用load model
// 2. 解决成员变量对外可见的问题
//      对于成员函数是特殊类型的, 比如说是cudaStream_t, 那么使用者必定会包含 cuda_runtime.h, 否则会语法解析失败
//      命名空间污染, 头文件污染
//      不干净的结果造成程序错误, 异常, 容易出现各种编译错误等等非预期的结果
shared_ptr<InferInterface> create_infer(const string& file){
    shared_ptr<InferImpl> instance(new InferImpl());
    if(!instance->load_model(file))
        instance.reset();
    return instance;
};
