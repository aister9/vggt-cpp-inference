#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <array>

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
    std::filesystem::path directoryFolder(path);
    if (!std::filesystem::exists(directoryFolder) || !std::filesystem::is_directory(directoryFolder)) {
        throw std::runtime_error("Invalid image folder path");
    }

    std::vector<std::filesystem::path> imagePaths;
    for (const auto& entry : std::filesystem::directory_iterator(directoryFolder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
            extension == ".bmp" || extension == ".webp") {
            imagePaths.push_back(entry.path());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());

    std::vector<torch::Tensor> images;
    images.reserve(imagePaths.size());

    for (const auto& imagePath : imagePaths) {
        images.push_back(load_image(imagePath.string()));
    }

    return images;
}

//1, C, H, W ==> 1, 3, 518, 518
std::vector<torch::Tensor> makeInputImages(const std::vector<torch::Tensor>& ori_images){
    constexpr int kBatchSize = 4;
    constexpr int kChannels = 3;
    constexpr int kImageSize = 518;

    if (ori_images.size() != kBatchSize) {
        throw std::runtime_error("makeInputImages expects exactly 4 images");
    }

    std::vector<torch::Tensor> inputImages;
    inputImages.reserve(ori_images.size());

    for (const auto& oriImage : ori_images) {
        torch::Tensor image = oriImage;

        if (image.dim() == 4) {
            if (image.size(0) != 1) {
                throw std::runtime_error("Input image batch dimension must be 1");
            }
            image = image.squeeze(0);
        }

        if (image.dim() != 3) {
            throw std::runtime_error("Input image must have shape [1,C,H,W] or [C,H,W]");
        }

        if (image.size(0) != kChannels) {
            throw std::runtime_error("Input image must have 3 channels");
        }

        image = image.to(torch::kFloat32).contiguous();

        image = torch::nn::functional::interpolate(
            image.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{kImageSize, kImageSize})
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze(0);

        image = image.clamp(0.0f, 1.0f).contiguous();
        inputImages.push_back(image);
    }

    return inputImages;
}

int main(){
    DataType type = FP32;
    std::filesystem::path onnxPath = (type==FP32)?"D:/vggt_onnx/vggt.onnx":"D:/vggt_onnx/vggt_fp16.onnx";

    const auto datasetPath = std::filesystem::current_path().parent_path().parent_path() / "dataset" / "fountain-p11";
    auto ori_images = load_images_from_folder(datasetPath.string());
    
    if (ori_images.size() < 4) {
        throw std::runtime_error("At least 4 images are required");
    }

    auto input_imgs = makeInputImages({ori_images[0], ori_images[1], ori_images[2], ori_images[3]});
    
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

    auto output_struct = models.infer(input_imgs);

    std::cout << output_struct.pose_enc << std::endl;
}
