#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>

#include <filesystem>

#include "OnnxVGGT.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

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
    int kBatchSize = ori_images.size();
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
                .size(std::vector<int64_t>{kImageSize, kImageSize}) // resize
                .mode(torch::kBilinear)
                .align_corners(false)
        ).squeeze(0);

        image = image.clamp(0.0f, 1.0f).contiguous();
        inputImages.push_back(image);
    }

    return inputImages;
}

void save_depth_maps(const torch::Tensor& depth, const std::filesystem::path& output_dir) {
    std::cout << "[save_depth_maps] start. output_dir=" << output_dir << std::endl;
    std::filesystem::create_directories(output_dir);

    torch::Tensor depth_cpu = depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::cout << "[save_depth_maps] depth moved to CPU. dim=" << depth_cpu.dim() << std::endl;

    if (depth_cpu.dim() == 5) {
        // Expected shape from VGGT: [1, N, H, W, 1]
        if (depth_cpu.size(0) != 1 || depth_cpu.size(4) != 1) {
            throw std::runtime_error("Depth tensor with 5 dims must have shape [1,N,H,W,1]");
        }
        depth_cpu = depth_cpu.squeeze(0).squeeze(-1); // [N,H,W]
    } else if (depth_cpu.dim() == 4) {
        if (depth_cpu.size(1) != 1) {
            throw std::runtime_error("Depth tensor with 4 dims must have shape [N,1,H,W]");
        }
        depth_cpu = depth_cpu.squeeze(1);
    } else if (depth_cpu.dim() == 2) {
        depth_cpu = depth_cpu.unsqueeze(0);
    } else if (depth_cpu.dim() != 3) {
        throw std::runtime_error("Unsupported depth tensor shape. Expected [N,1,H,W], [N,H,W], or [H,W]");
    }

    const int64_t batch = depth_cpu.size(0);
    const int64_t height = depth_cpu.size(1);
    const int64_t width = depth_cpu.size(2);
    std::cout << "[save_depth_maps] normalized shape [N,H,W] = ["
              << batch << ", " << height << ", " << width << "]" << std::endl;

    for (int64_t i = 0; i < batch; ++i) {
        std::cout << "[save_depth_maps] writing index " << i << std::endl;
        torch::Tensor single = depth_cpu[i].contiguous();

        const float min_val = single.min().item<float>();
        const float max_val = single.max().item<float>();
        std::cout << "[save_depth_maps] depth_" << i << " min=" << min_val << ", max=" << max_val << std::endl;
        torch::Tensor normalized;

        if (max_val > min_val) {
            normalized = (single - min_val) / (max_val - min_val);
        } else {
            normalized = torch::zeros_like(single);
        }

        torch::Tensor depth_u8 = (normalized * 255.0f).clamp(0.0f, 255.0f).to(torch::kUInt8).contiguous();

        std::vector<std::uint8_t> buffer(static_cast<size_t>(height * width));
        std::memcpy(buffer.data(), depth_u8.data_ptr<std::uint8_t>(), buffer.size() * sizeof(std::uint8_t));

        const auto output_path = output_dir / ("depth_" + std::to_string(i) + ".png");
        const int write_ok = stbi_write_png(
            output_path.string().c_str(),
            static_cast<int>(width),
            static_cast<int>(height),
            1,
            buffer.data(),
            static_cast<int>(width)
        );

        if (write_ok == 0) {
            throw std::runtime_error("Failed to write depth map: " + output_path.string());
        }
        std::cout << "[save_depth_maps] wrote: " << output_path << std::endl;
    }
    std::cout << "[save_depth_maps] done." << std::endl;
}

void save_point_cloud_pcd(
    const torch::Tensor& world_points,
    const torch::Tensor& images,
    const torch::Tensor& world_points_conf,
    const std::filesystem::path& pcd_path,
    float conf_threshold = 0.5f) {

    std::cout << "[save_point_cloud_pcd] start. path=" << pcd_path << std::endl;

    torch::Tensor points_cpu = world_points.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor images_cpu = images.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor conf_cpu = world_points_conf.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

    // points: [1,N,H,W,3] -> [N,H,W,3]
    if (points_cpu.dim() == 5) {
        if (points_cpu.size(0) != 1 || points_cpu.size(4) != 3) {
            throw std::runtime_error("world_points must have shape [1,N,H,W,3]");
        }
        points_cpu = points_cpu.squeeze(0);
    } else if (points_cpu.dim() != 4 || points_cpu.size(3) != 3) {
        throw std::runtime_error("world_points must have shape [N,H,W,3] or [1,N,H,W,3]");
    }

    // images: [1,N,3,H,W] -> [N,H,W,3]
    if (images_cpu.dim() == 5) {
        if (images_cpu.size(0) != 1 || images_cpu.size(2) != 3) {
            throw std::runtime_error("images must have shape [1,N,3,H,W]");
        }
        images_cpu = images_cpu.squeeze(0).permute({0, 2, 3, 1}).contiguous();
    } else if (images_cpu.dim() == 4 && images_cpu.size(1) == 3) {
        images_cpu = images_cpu.permute({0, 2, 3, 1}).contiguous();
    } else {
        throw std::runtime_error("images must have shape [N,3,H,W] or [1,N,3,H,W]");
    }

    // conf: [1,N,H,W] -> [N,H,W]
    if (conf_cpu.dim() == 4) {
        if (conf_cpu.size(0) == 1) {
            conf_cpu = conf_cpu.squeeze(0);
        }
    } else if (conf_cpu.dim() != 3) {
        throw std::runtime_error("world_points_conf must have shape [N,H,W] or [1,N,H,W]");
    }

    if (points_cpu.sizes().slice(0, 3) != images_cpu.sizes().slice(0, 3) ||
        points_cpu.sizes().slice(0, 3) != conf_cpu.sizes()) {
        throw std::runtime_error("Shape mismatch among world_points, images, and world_points_conf");
    }

    const int64_t n = points_cpu.size(0);
    const int64_t h = points_cpu.size(1);
    const int64_t w = points_cpu.size(2);

    std::vector<std::array<float, 4>> cloud_rows;
    cloud_rows.reserve(static_cast<size_t>(n * h * w));

    auto pts = points_cpu.accessor<float, 4>();
    auto imgs = images_cpu.accessor<float, 4>();
    auto conf = conf_cpu.accessor<float, 3>();

    for (int64_t vi = 0; vi < n; ++vi) {
        for (int64_t yi = 0; yi < h; ++yi) {
            for (int64_t xi = 0; xi < w; ++xi) {
                const float c = conf[vi][yi][xi];
                if (c < conf_threshold) {
                    continue;
                }

                const float x = pts[vi][yi][xi][0];
                const float y = pts[vi][yi][xi][1];
                const float z = pts[vi][yi][xi][2];
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    continue;
                }

                const int r = static_cast<int>(std::round(std::clamp(imgs[vi][yi][xi][0], 0.0f, 1.0f) * 255.0f));
                const int g = static_cast<int>(std::round(std::clamp(imgs[vi][yi][xi][1], 0.0f, 1.0f) * 255.0f));
                const int b = static_cast<int>(std::round(std::clamp(imgs[vi][yi][xi][2], 0.0f, 1.0f) * 255.0f));

                const std::uint32_t rgb_uint = (static_cast<std::uint32_t>(r) << 16) |
                                               (static_cast<std::uint32_t>(g) << 8) |
                                               static_cast<std::uint32_t>(b);

                float rgb_float = 0.0f;
                static_assert(sizeof(float) == sizeof(std::uint32_t), "float/u32 size mismatch");
                std::memcpy(&rgb_float, &rgb_uint, sizeof(float));

                cloud_rows.push_back({x, y, z, rgb_float});
            }
        }
    }

    std::filesystem::create_directories(pcd_path.parent_path());
    std::ofstream ofs(pcd_path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open PCD file for write: " + pcd_path.string());
    }

    ofs << "VERSION .7\n";
    ofs << "FIELDS x y z rgb\n";
    ofs << "SIZE 4 4 4 4\n";
    ofs << "TYPE F F F F\n";
    ofs << "COUNT 1 1 1 1\n";
    ofs << "WIDTH " << cloud_rows.size() << "\n";
    ofs << "HEIGHT 1\n";
    ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << cloud_rows.size() << "\n";
    ofs << "DATA ascii\n";

    for (const auto& row : cloud_rows) {
        ofs << row[0] << " " << row[1] << " " << row[2] << " " << row[3] << "\n";
    }

    ofs.close();
    std::cout << "[save_point_cloud_pcd] saved points=" << cloud_rows.size()
              << ", file=" << pcd_path << std::endl;
}

int main(){
    std::cout << "[main] start" << std::endl;
    DataType type = FP32;
    std::filesystem::path onnxPath = (type==FP32)?"D:/vggt_onnx/VGGT-1B-onnx/vggt.onnx":"D:/vggt_onnx/VGGT-1B-onnx/vggt_fp16.onnx";
    std::cout << "[main] onnxPath=" << onnxPath << std::endl;

    const auto datasetPath = std::filesystem::current_path().parent_path().parent_path() / "dataset" / "fountain-p11";
    std::cout << "[main] datasetPath=" << datasetPath << std::endl;
    auto ori_images = load_images_from_folder(datasetPath.string());
    std::cout << "[main] loaded images count=" << ori_images.size() << std::endl;
    
    if (ori_images.size() < 4) {
        throw std::runtime_error("At least 4 images are required");
    }

    auto input_imgs = makeInputImages(ori_images);
    std::cout << "[main] preprocessed images count=" << input_imgs.size() << std::endl;
    
    std::cout << "Hello libTorch!" << std::endl;
    if(torch::cuda::is_available()){
        std::cout << "CUDA is okay!" << std::endl;
    }else{
        std::cout << "CUDA is failed!" << std::endl;
    }

    torch::Tensor x = torch::rand({2,3}).cuda();
    std::cout << x << std::endl;

    OnnxVGGT models(type, onnxPath);
    std::cout << "[main] building model..." << std::endl;
    if(models.build()){
        std::cout << "model load successfully!" << std::endl;
        models.printInfo();
    }
    std::cout << "[main] infer start" << std::endl;

    auto output_struct = models.infer(input_imgs);
    std::cout << "[main] infer done" << std::endl;
    std::cout << "[main] depth dim=" << output_struct.depth.dim()
              << ", device=" << output_struct.depth.device()
              << ", dtype=" << output_struct.depth.dtype() << std::endl;
    std::cout << "[main] depth sizes=" << output_struct.depth.sizes() << std::endl;

    const auto outputPath = datasetPath / "output";
    std::cout << "[main] save depth to " << outputPath << std::endl;
    save_depth_maps(output_struct.depth, outputPath);
    std::cout << "Saved depth maps to: " << outputPath << std::endl;

    const auto pcdPath = outputPath / "point_cloud_rgb.pcd";
    std::cout << "[main] save point cloud to " << pcdPath << std::endl;
    save_point_cloud_pcd(
        output_struct.world_points,
        output_struct.images,
        output_struct.world_points_conf,
        pcdPath,
        0.5f);

    std::cout << output_struct.pose_enc << std::endl;
}
