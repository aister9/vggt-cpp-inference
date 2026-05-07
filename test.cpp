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
#include <unordered_set>

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

void save_depth_maps(
    const torch::Tensor& depth,
    const std::filesystem::path& output_dir,
    const std::vector<std::pair<int64_t, int64_t>>& original_sizes) {

    std::cout << "[save_depth_maps] start (resize to original). output_dir=" << output_dir << std::endl;
    std::filesystem::create_directories(output_dir);

    torch::Tensor depth_cpu = depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

    if (depth_cpu.dim() == 5) {
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

    const int64_t n = depth_cpu.size(0);
    if (static_cast<int64_t>(original_sizes.size()) != n) {
        throw std::runtime_error("original_sizes count must match depth view count");
    }

    for (int64_t i = 0; i < n; ++i) {
        const int64_t out_h = original_sizes[static_cast<size_t>(i)].first;
        const int64_t out_w = original_sizes[static_cast<size_t>(i)].second;

        torch::Tensor single = depth_cpu[i].unsqueeze(0).unsqueeze(0); // [1,1,H,W]
        single = torch::nn::functional::interpolate(
            single,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));
        single = single.squeeze(0).squeeze(0).contiguous(); // [H,W]

        const float min_val = single.min().item<float>();
        const float max_val = single.max().item<float>();
        torch::Tensor normalized = (max_val > min_val) ? (single - min_val) / (max_val - min_val) : torch::zeros_like(single);
        torch::Tensor depth_u8 = (normalized * 255.0f).clamp(0.0f, 255.0f).to(torch::kUInt8).contiguous();

        std::vector<std::uint8_t> buffer(static_cast<size_t>(out_h * out_w));
        std::memcpy(buffer.data(), depth_u8.data_ptr<std::uint8_t>(), buffer.size() * sizeof(std::uint8_t));

        const auto output_path = output_dir / ("depth_" + std::to_string(i) + ".png");
        const int write_ok = stbi_write_png(
            output_path.string().c_str(),
            static_cast<int>(out_w),
            static_cast<int>(out_h),
            1,
            buffer.data(),
            static_cast<int>(out_w));

        if (write_ok == 0) {
            throw std::runtime_error("Failed to write depth map: " + output_path.string());
        }
    }
}

void save_point_cloud_pcd(
    const torch::Tensor& world_points,
    const torch::Tensor& images,
    const torch::Tensor& world_points_conf,
    const std::filesystem::path& pcd_path,
    const std::vector<std::pair<int64_t, int64_t>>& original_sizes,
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

    if (static_cast<int64_t>(original_sizes.size()) != n) {
        throw std::runtime_error("original_sizes count must match point view count");
    }

    std::vector<std::array<float, 4>> cloud_rows;
    cloud_rows.reserve(static_cast<size_t>(n * h * w));

    for (int64_t vi = 0; vi < n; ++vi) {
        const int64_t out_h = original_sizes[static_cast<size_t>(vi)].first;
        const int64_t out_w = original_sizes[static_cast<size_t>(vi)].second;

        torch::Tensor pts_view = points_cpu[vi].permute({2, 0, 1}).unsqueeze(0); // [1,3,H,W]
        torch::Tensor img_view = images_cpu[vi].permute({2, 0, 1}).unsqueeze(0);  // [1,3,H,W]
        torch::Tensor conf_view = conf_cpu[vi].unsqueeze(0).unsqueeze(0);          // [1,1,H,W]

        pts_view = torch::nn::functional::interpolate(
            pts_view,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));
        img_view = torch::nn::functional::interpolate(
            img_view,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));
        conf_view = torch::nn::functional::interpolate(
            conf_view,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));

        torch::Tensor pts_hw3 = pts_view.squeeze(0).permute({1, 2, 0}).contiguous(); // [H,W,3]
        torch::Tensor img_hw3 = img_view.squeeze(0).permute({1, 2, 0}).contiguous(); // [H,W,3]
        torch::Tensor conf_hw = conf_view.squeeze(0).squeeze(0).contiguous();         // [H,W]

        auto pts = pts_hw3.accessor<float, 3>();
        auto imgs = img_hw3.accessor<float, 3>();
        auto conf = conf_hw.accessor<float, 2>();

        for (int64_t yi = 0; yi < out_h; ++yi) {
            for (int64_t xi = 0; xi < out_w; ++xi) {
                const float c = conf[yi][xi];
                if (c < conf_threshold) {
                    continue;
                }

                const float x = pts[yi][xi][0];
                const float y = pts[yi][xi][1];
                const float z = pts[yi][xi][2];
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    continue;
                }

                const int r = static_cast<int>(std::round(std::clamp(imgs[yi][xi][0], 0.0f, 1.0f) * 255.0f));
                const int g = static_cast<int>(std::round(std::clamp(imgs[yi][xi][1], 0.0f, 1.0f) * 255.0f));
                const int b = static_cast<int>(std::round(std::clamp(imgs[yi][xi][2], 0.0f, 1.0f) * 255.0f));

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
    std::ofstream ofs(pcd_path, std::ios::out | std::ios::trunc | std::ios::binary);
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
    ofs << "DATA binary\n";

    for (const auto& row : cloud_rows) {
        ofs.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(sizeof(float) * 4));
    }

    ofs.close();
    std::cout << "[save_point_cloud_pcd] saved points=" << cloud_rows.size()
              << ", file=" << pcd_path << std::endl;
}

void save_point_cloud_from_depth_unprojection(
    const torch::Tensor& depth,
    const torch::Tensor& images,
    const torch::Tensor& depth_conf,
    const torch::Tensor& pose_enc,
    const std::filesystem::path& pcd_path,
    const std::vector<std::pair<int64_t, int64_t>>& original_sizes,
    float conf_threshold = 0.5f) {

    std::cout << "[save_point_cloud_from_depth_unprojection] start. path=" << pcd_path << std::endl;

    torch::Tensor depth_cpu = depth.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor images_cpu = images.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor conf_cpu = depth_conf.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor pose_cpu = pose_enc.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();

    if (depth_cpu.dim() == 5) {
        if (depth_cpu.size(0) != 1 || depth_cpu.size(4) != 1) {
            throw std::runtime_error("depth must have shape [1,N,H,W,1]");
        }
        depth_cpu = depth_cpu.squeeze(0).squeeze(-1); // [N,H,W]
    } else if (depth_cpu.dim() == 4 && depth_cpu.size(1) == 1) {
        depth_cpu = depth_cpu.squeeze(1);
    } else if (depth_cpu.dim() != 3) {
        throw std::runtime_error("depth must have shape [N,H,W] or [1,N,H,W,1]");
    }

    if (images_cpu.dim() == 5) {
        if (images_cpu.size(0) != 1 || images_cpu.size(2) != 3) {
            throw std::runtime_error("images must have shape [1,N,3,H,W]");
        }
        images_cpu = images_cpu.squeeze(0).permute({0, 2, 3, 1}).contiguous(); // [N,H,W,3]
    } else if (images_cpu.dim() == 4 && images_cpu.size(1) == 3) {
        images_cpu = images_cpu.permute({0, 2, 3, 1}).contiguous(); // [N,H,W,3]
    } else {
        throw std::runtime_error("images must have shape [N,3,H,W] or [1,N,3,H,W]");
    }

    if (conf_cpu.dim() == 4) {
        if (conf_cpu.size(0) == 1) {
            conf_cpu = conf_cpu.squeeze(0);
        }
    } else if (conf_cpu.dim() != 3) {
        throw std::runtime_error("depth_conf must have shape [N,H,W] or [1,N,H,W]");
    }

    if (depth_cpu.sizes() != conf_cpu.sizes()) {
        throw std::runtime_error("depth and depth_conf shape mismatch");
    }
    if (depth_cpu.size(0) != images_cpu.size(0) ||
        depth_cpu.size(1) != images_cpu.size(1) ||
        depth_cpu.size(2) != images_cpu.size(2)) {
        throw std::runtime_error("depth and images shape mismatch");
    }

    if (pose_cpu.dim() != 3 || pose_cpu.size(0) != 1 || pose_cpu.size(2) != 9) {
        throw std::runtime_error("pose_enc must have shape [1,N,9]");
    }

    const int64_t n = depth_cpu.size(0);
    if (static_cast<int64_t>(original_sizes.size()) != n) {
        throw std::runtime_error("original_sizes count must match depth view count");
    }
    if (pose_cpu.size(1) != n) {
        throw std::runtime_error("pose_enc view count must match depth view count");
    }

    struct VoxelStat {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        double sum_r = 0.0;
        double sum_g = 0.0;
        double sum_b = 0.0;
        uint32_t sample_count = 0;
        uint32_t view_count = 0;
    };

    const float voxel_size = 0.0001f; // 1 cm
    const float inv_voxel = 1.0f / voxel_size;

    auto make_key = [inv_voxel](float x, float y, float z) -> std::string {
        const int32_t ix = static_cast<int32_t>(std::llround(x * inv_voxel));
        const int32_t iy = static_cast<int32_t>(std::llround(y * inv_voxel));
        const int32_t iz = static_cast<int32_t>(std::llround(z * inv_voxel));
        
        return std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(iz);
    };

    std::unordered_map<std::string, VoxelStat> voxels;
    voxels.reserve(static_cast<size_t>(n * depth_cpu.size(1) * depth_cpu.size(2) / 8));

    for (int64_t vi = 0; vi < n; ++vi) {
        const int64_t out_h = original_sizes[static_cast<size_t>(vi)].first;
        const int64_t out_w = original_sizes[static_cast<size_t>(vi)].second;
        std::unordered_set<std::string> seen_in_view;
        seen_in_view.reserve(static_cast<size_t>(out_h * out_w / 8));

        torch::Tensor d = depth_cpu[vi].unsqueeze(0).unsqueeze(0);          // [1,1,H,W]
        torch::Tensor c = conf_cpu[vi].unsqueeze(0).unsqueeze(0);           // [1,1,H,W]
        torch::Tensor rgb = images_cpu[vi].permute({2, 0, 1}).unsqueeze(0); // [1,3,H,W]

        d = torch::nn::functional::interpolate(
            d,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));
        c = torch::nn::functional::interpolate(
            c,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));
        rgb = torch::nn::functional::interpolate(
            rgb,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{out_h, out_w})
                .mode(torch::kBilinear)
                .align_corners(false));

        torch::Tensor d_hw = d.squeeze(0).squeeze(0).contiguous();
        torch::Tensor c_hw = c.squeeze(0).squeeze(0).contiguous();
        torch::Tensor rgb_hw3 = rgb.squeeze(0).permute({1, 2, 0}).contiguous();

        auto d_acc = d_hw.accessor<float, 2>();
        auto c_acc = c_hw.accessor<float, 2>();
        auto rgb_acc = rgb_hw3.accessor<float, 3>();

        // Decode pose encoding (absT_quaR_FoV) following VGGT:
        // pose[:3]=T, pose[3:7]=quat(x,y,z,w), pose[7]=fov_h, pose[8]=fov_w
        // Extrinsic is camera-from-world [R|t].
        auto p = pose_cpu[0][vi];
        const float tx = p[0].item<float>();
        const float ty = p[1].item<float>();
        const float tz = p[2].item<float>();
        float qx = p[3].item<float>();
        float qy = p[4].item<float>();
        float qz = p[5].item<float>();
        float qw = p[6].item<float>();
        const float fov_h = p[7].item<float>();
        const float fov_w = p[8].item<float>();

        const float qn = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (qn > 0.0f) {
            qx /= qn; qy /= qn; qz /= qn; qw /= qn;
        }

        const float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
        const float r01 = 2.0f * (qx * qy - qz * qw);
        const float r02 = 2.0f * (qx * qz + qy * qw);
        const float r10 = 2.0f * (qx * qy + qz * qw);
        const float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
        const float r12 = 2.0f * (qy * qz - qx * qw);
        const float r20 = 2.0f * (qx * qz - qy * qw);
        const float r21 = 2.0f * (qy * qz + qx * qw);
        const float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);

        // Intrinsics from FoV (VGGT pose_encoding_to_extri_intri)
        const float fy = (static_cast<float>(out_h) * 0.5f) / std::tan(fov_h * 0.5f);
        const float fx = (static_cast<float>(out_w) * 0.5f) / std::tan(fov_w * 0.5f);
        const float cx = static_cast<float>(out_w) * 0.5f;
        const float cy = static_cast<float>(out_h) * 0.5f;

        for (int64_t y = 0; y < out_h; ++y) {
            for (int64_t x = 0; x < out_w; ++x) {
                const float conf_val = c_acc[y][x];
                if (conf_val < conf_threshold) {
                    continue;
                }

                const float z = d_acc[y][x];
                if (!std::isfinite(z) || z <= 0.0f) {
                    continue;
                }

                // Camera point from depth
                const float xc = (static_cast<float>(x) - cx) * z / fx;
                const float yc = (static_cast<float>(y) - cy) * z / fy;
                const float zc = z;

                // Convert camera->world using inverse of [R|t]:
                // Xw = R^T * (Xc - t)
                const float v0 = xc - tx;
                const float v1 = yc - ty;
                const float v2 = zc - tz;
                const float px = r00 * v0 + r10 * v1 + r20 * v2;
                const float py = r01 * v0 + r11 * v1 + r21 * v2;
                const float pz = r02 * v0 + r12 * v1 + r22 * v2;

                const int r = static_cast<int>(std::round(std::clamp(rgb_acc[y][x][0], 0.0f, 1.0f) * 255.0f));
                const int g = static_cast<int>(std::round(std::clamp(rgb_acc[y][x][1], 0.0f, 1.0f) * 255.0f));
                const int b = static_cast<int>(std::round(std::clamp(rgb_acc[y][x][2], 0.0f, 1.0f) * 255.0f));
                const std::string key = make_key(px, py, pz);
                auto& stat = voxels[key];
                stat.sum_x += px;
                stat.sum_y += py;
                stat.sum_z += pz;
                stat.sum_r += static_cast<double>(r);
                stat.sum_g += static_cast<double>(g);
                stat.sum_b += static_cast<double>(b);
                stat.sample_count += 1;

                if (seen_in_view.insert(key).second) {
                    stat.view_count += 1;
                }
            }
        }
    }

    std::vector<std::array<float, 4>> cloud_rows;
    cloud_rows.reserve(voxels.size());
    for (const auto& kv : voxels) {
        const VoxelStat& stat = kv.second;
        if (stat.view_count < 2 || stat.sample_count == 0) {
            continue;
        }

        const float x = static_cast<float>(stat.sum_x / stat.sample_count);
        const float y = static_cast<float>(stat.sum_y / stat.sample_count);
        const float z = static_cast<float>(stat.sum_z / stat.sample_count);
        const int r = static_cast<int>(std::round(stat.sum_r / stat.sample_count));
        const int g = static_cast<int>(std::round(stat.sum_g / stat.sample_count));
        const int b = static_cast<int>(std::round(stat.sum_b / stat.sample_count));

        const std::uint32_t rgb_uint = (static_cast<std::uint32_t>(std::clamp(r, 0, 255)) << 16) |
                                       (static_cast<std::uint32_t>(std::clamp(g, 0, 255)) << 8) |
                                       static_cast<std::uint32_t>(std::clamp(b, 0, 255));
        float rgb_float = 0.0f;
        std::memcpy(&rgb_float, &rgb_uint, sizeof(float));

        cloud_rows.push_back({x, y, z, rgb_float});
    }

    std::filesystem::create_directories(pcd_path.parent_path());
    std::ofstream ofs(pcd_path, std::ios::out | std::ios::trunc | std::ios::binary);
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
    ofs << "DATA binary\n";
    for (const auto& row : cloud_rows) {
        ofs.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(sizeof(float) * 4));
    }
    ofs.close();

    std::cout << "[save_point_cloud_from_depth_unprojection] saved points=" << cloud_rows.size()
              << ", file=" << pcd_path << std::endl;
}

int main(){
    std::cout << "[main] start" << std::endl;
    DataType type = FP16;
    constexpr size_t kMaxInputViews = 24;
    std::filesystem::path onnxPath = (type==FP32)?"D:/vggt_onnx/VGGT-1B-onnx/vggt.onnx":"D:/vggt_onnx/VGGT-1B-onnx/vggt_fp16.onnx";
    std::cout << "[main] onnxPath=" << onnxPath << std::endl;

    //const auto datasetPath = std::filesystem::current_path().parent_path().parent_path() / "dataset" / "fountain-p11";
    std::filesystem::path datasetPath_dtu = "D:/MVSDataset/Testset/scan25";
    const auto datasetPath = datasetPath_dtu / "urd";
    std::cout << "[main] datasetPath=" << datasetPath << std::endl;
    auto ori_images = load_images_from_folder(datasetPath.string());
    std::cout << "[main] loaded images count=" << ori_images.size() << std::endl;
    
    if (ori_images.size() < 4) {
        throw std::runtime_error("At least 4 images are required");
    }

    if (ori_images.size() > kMaxInputViews) {
        std::cout << "[main] limiting input views from " << ori_images.size()
                  << " to " << kMaxInputViews << " to reduce GPU memory usage." << std::endl;
        ori_images.resize(kMaxInputViews);
    }

    std::vector<std::pair<int64_t, int64_t>> original_sizes;
    original_sizes.reserve(ori_images.size());
    for (const auto& img : ori_images) {
        torch::Tensor t = (img.dim() == 4) ? img.squeeze(0) : img;
        original_sizes.emplace_back(t.size(1), t.size(2));
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

    const auto outputPath = datasetPath_dtu / "vggt_output";
    std::cout << "[main] save depth to " << outputPath << std::endl;
    save_depth_maps(output_struct.depth, outputPath, original_sizes);
    std::cout << "Saved depth maps to: " << outputPath << std::endl;

    const auto pcdPath = outputPath / "point_cloud_rgb.pcd";
    std::cout << "[main] save point cloud to " << pcdPath << std::endl;
    save_point_cloud_pcd(
        output_struct.world_points,
        output_struct.images,
        output_struct.world_points_conf,
        pcdPath,
        original_sizes,
        0.95f);

    const auto pcdFromDepthPath = outputPath / "point_cloud_from_depth_rgb.pcd";
    std::cout << "[main] save depth-unprojected point cloud to " << pcdFromDepthPath << std::endl;
    save_point_cloud_from_depth_unprojection(
        output_struct.depth,
        output_struct.images,
        output_struct.depth_conf,
        output_struct.pose_enc,
        pcdFromDepthPath,
        original_sizes,
        0.95f);

    std::cout << output_struct.pose_enc << std::endl;
}
