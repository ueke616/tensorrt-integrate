// #include <thread>
// #include <queue>
// #include <mutex>    // 锁对象(互斥信号量)
// #include <condition_variable>   // 条件变量(解决资源积累问题)
// #include <string>
// #include <cstdio>
// #include <memory>
// #include <future>   // 包含future和promise, 可以把信息从消费者送回生产者
// #include <chrono>

// using namespace std;
// // 共享资源访问问题
// // queue, stl 对象不是 thread-safe
// // 如果生产频率高于消费频率, 则队列出现堆积现象
// // Image, 1280x720x3 = 2.5MB
// struct Job{
//     shared_ptr<promise<string>> pro;
//     string input;
// };

// queue<Job> qjobs_;
// mutex lock_;    // 锁对象
// condition_variable cv_;     // 条件变量
// const int limit_ = 5;

// void vedio_capture(){   // producter
//     int pic_id = 0;
//     while(true){
//         Job job;
//         {   // 要加锁的代码块
//             // 加锁的代码块成原子操作, 不会出现资源冲突
//             unique_lock<mutex> l(lock_);
//             char name[10];
//             sprintf(name, "PIC-%d", pic_id++);  // 批量赋值
//             printf("生产一个新图片: %s, qjobs_.size = %d\n", name, qjobs_.size());
//             // if (qjobs_.size() > limit){
//             //     wait();
//             // }
//             // wait 的流程是: 一旦进入 wait, 则解锁; 一旦退出wait, 则加锁
//             cv_.wait(l, [&](){ //队列满则等待
//                 // return false 表示继续等待
//                 // return true  表示不等待, 跳出 wait
//                 return qjobs_.size() < limit_;
//             });
//             // 如果队列满了, 我不生产, 我去等待队列有空间再生产
//             // 通知的问题, 如何通知到 wait, 让他即时的可以退出
//             job.pro.reset(new promise<string>());   // 生产者告诉消费者, 一定要在未来某个时候返回一个结果, 通过promise
//             job.input = name;
//             qjobs_.push(job);

//             // 同步模式
//             // detection -> infer
//             // face      -> infer
//             // feature   -> infer

//             // 异步模式
//             // detection -> infer
//             // face      -> infer
//             // feature   -> infer

            

//             // 拿到推理结果, 跟推理之前的图像一起进行画框, 然后走下面流程 
//         }
//         // 一次进行 3 个结果的回收, 然后进行处理
//         // 等待这个 job 处理完毕, 拿结果
//         // .get过后, 实现等待, 直到promise->set_value被执行了, 这里的返回值就是result
//         auto result = job.pro->get_future().get();
//         printf("JOB %s -> %s \n", job.input.c_str(), result.c_str());
//         this_thread::sleep_for(chrono::milliseconds(500));
//     }
// }

// void infer_worker(){    // resumer
//     while(true){
//         if (!qjobs_.empty()){
//             {
//                 lock_guard<mutex> l(lock_);
//                 auto pjob = qjobs_.front();
//                 qjobs_.pop();

//                 // 消费掉一个, 就可以通知wait, 去结束等待
//                 cv_.notify_one();
//                 printf("消费掉一个图片: %s \n", pjob.input.c_str());

//                 auto result = pjob.input + "--- infer";
//                 // new_pic 送回生产者, 怎么办
//                 pjob.pro->set_value(result);
//             }
//             this_thread::sleep_for(chrono::microseconds(1000));
//         }
//         this_thread::yield(); // 队列已空, 交出cpu的控制权, 重新等待下一个时间片
//     }
// }

// int main(){

//     thread t0(vedio_capture);
//     thread t1(infer_worker);

//     t0.join();
//     t1.join();
//     return 0;
// }

