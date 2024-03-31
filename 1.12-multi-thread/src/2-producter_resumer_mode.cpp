// #include <thread>
// #include <queue>
// #include <mutex>    // 锁对象(互斥信号量)
// #include <string>
// #include <cstdio>
// #include <chrono>

// using namespace std;
// // 共享资源访问问题
// // queue, stl 对象不是 thread-safe
// // 如果生产频率高于消费频率, 则队列出现堆积现象
// // Image, 1280x720x3 = 2.5MB
// queue<string> qjobs_;
// mutex lock_;    // 锁对象

// void vedio_capture(){   // producter
//     int pic_id = 0;
//     while(true){
//         {   // 要加锁的代码块
//             // 加锁的代码块成原子操作, 不会出现资源冲突
//             lock_guard<mutex> l(lock_);
//             char name[10];
//             sprintf(name, "PIC-%d", pic_id++);  // 批量赋值
//             printf("生产一个新图片: %s, qjobs_.size = %d\n", name, qjobs_.size());
//             qjobs_.push(name);
//         }
//         this_thread::sleep_for(chrono::milliseconds(500));
//     }
// }

// void infer_worker(){    // resumer
//     while(true){
//         if (!qjobs_.empty()){
//             {
//                 lock_guard<mutex> l(lock_);
//                 auto pic = qjobs_.front();
//                 qjobs_.pop();

//                 printf("消费掉一个图片: %s \n", pic.c_str());
//             }
//         }
//         this_thread::sleep_for(chrono::microseconds(1000));
//         // this_thread::yield(); // 队列已空, 交出cpu的控制权, 重新等待下一个时间片
//     }
// }

// int main(){

//     thread t0(vedio_capture);
//     thread t1(infer_worker);

//     t0.join();
//     t1.join();
//     return 0;
// }

