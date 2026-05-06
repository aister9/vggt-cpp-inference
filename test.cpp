#include <torch/torch.h>
#include <iostream>
#include <string>

#include <filesystem>

#include "OnnxVGGT.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

torch::Tensor load_image(const std::string &path){
    int width, height, channels;

    unsigned char* img = stbi_load(path.c_str(),
        &width,
        &height,
        &channels,
        STBI_rgb
    );

    if(!img){
        throw std::runtime_error(
            "Failed to load image"
        );
    }

    torch::Tensor tensor= torch::from_blob(img, {height, width, 3}, torch::kUInt8);
    tensor = tensor.permute({2,0,1}).to(torch::kFloat32)/255.0f;
    tensor = tensor.unsqueeze(0);

    stbi_image_free(img);

    return tensor.clone();
}

std::vector<torch::Tensor> load_images_from_folder(const std::string &path){
    std::filesystem::path directoryFolder;
    
    return {};
}

int main(){
    DataType type = FP32;
    std::filesystem::path onnxPath = (type==FP32)?"D:/vggt_onnx/vggt.onnx":"D:/vggt_onnx/vggt_fp16.onnx";


    std::cout << "Hello libTorch!" << std::endl;
    if(torch::cuda::is_available()){
        std::cout << "CUDA is okay!" << std::endl;
    }else{
        std::cout << "CUDA is failed!" << std::endl;
    }

    torch::Tensor x = torch::rand({2,3}).cuda();
    std::cout << x << std::endl;

    OnnxVGGT models;
    if(models.build()){
        std::cout << "model load successfully!" << std::endl;
        models.printInfo();
    }
}