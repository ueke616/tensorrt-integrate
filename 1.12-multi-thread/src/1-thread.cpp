
// 1. thread, 启动线程
#include <thread>
#include <chrono>   // 时间相关
#include <cstdio>
#include <iostream>

using namespace std;

void worker(int a, string& str){
    printf("hello thread. %d \n", a);
    this_thread::sleep_for(chrono::milliseconds(10000));  // 等待一秒钟
    printf("worker done.  \n");
    str = "reference string";
}

class Infer {
    public:
        Infer() {
            work_thread_ = thread(&Infer::infer_worker, this);
        }
    private:
        thread work_thread_;

        // static void infer_worker(Infer* self) {}
        void infer_worker() {
            this->work_thread_;
            work_thread_;
        }
};

// int main(){
//     // 完整的启动线程案例
//     // thread t(func, args...);
//     // thread t(worker);
//     // thread t(worker, 5678);
//     string param;
//     thread t(worker, 5678, std::ref(param));
//     // t.join();   // 等待线程结束
//     /*
//     没启动不join, 若启动必join
//     1. t.join() 如果不加, 会在析构时提示异常, 出现core dumped
//         只要线程t 启动了, 就必须要 join
//     2. 若 t 没有启动线程, 去执行 t.join, 会异常
//         只要线程 t 没有启动, 一定不能 join
//     */  
// //    t.detach();      // 分离线程, 取消管理权, 使得线程称为野线程, 不建议使用
//    // 3. 野线程, 不需要 join, 线程交给系统管理, 程序退出后, 所有线程才退出

//    if(t.joinable())     // 如果可以join, 那就join
//         t.join();
//     cout << param << endl;

//     printf("done. \n");
//     return 0;
// }

