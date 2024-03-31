// #include <thread>
// #include <queue>
// #include <mutex>
// #include <string>
// #include <chrono>
// #include <condition_variable>
// #include <memory>
// #include <future>
// #include <cstdio>

// using namespace std;


// int main(){
//     // 资源的获取
//     // Infer infer;
//     // // 初始化
//     // infer.load_model("a");
//     // infer.load_model("b");

//     // infer.forward();
//     auto infer = create_infer("a");
//     if (infer == nullptr) {
//         printf("failed. \n");
//         return -1;
//     }
//     infer->forward();
//     return 0;
// }


#include "infer.hpp"

int main(){
    auto infer = create_infer("a");
    if(infer == nullptr){
        printf("failed. \n");
        return -1;
    }

    // 串行
    // auto fa = infer->forward("A").get();
    // auto fb = infer->forward("B").get();
    // auto fc = infer->forward("C").get();
    // printf("%s \n", fa.c_str());
    // printf("%s \n", fb.c_str());
    // printf("%s \n", fc.c_str());

    // 并行
    auto fa = infer->forward("A");
    auto fb = infer->forward("B");
    auto fc = infer->forward("C");
    printf("%s \n", fa.get().c_str());
    printf("%s \n", fb.get().c_str());
    printf("%s \n", fc.get().c_str());
    return 0;
}

