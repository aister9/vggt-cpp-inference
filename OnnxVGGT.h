#pragma once
#include <torch/torch.h>

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <unordered_map>

#include <NvInfer.h>
#include <NvOnnxParser.h>

enum DataType
{
    FP16,
    FP32
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

struct InferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        delete obj;
    }
};

static auto StreamDeleter = [](cudaStream_t *pStream)
{
    if (pStream)
    {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

inline torch::ScalarType toTorchScalarType(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        return torch::kFloat32;
    case nvinfer1::DataType::kHALF:
        return torch::kFloat16;
    case nvinfer1::DataType::kINT32:
        return torch::kInt32;
    case nvinfer1::DataType::kINT64:
        return torch::kInt64;
    case nvinfer1::DataType::kBOOL:
        return torch::kBool;
    default:
        throw std::runtime_error("Unsupported TensorRT tensor datatype");
    }
}

class OnnxVGGT
{
public:
    struct VGGTOutput
    {
        torch::Tensor pose_enc;
        torch::Tensor depth;
        torch::Tensor depth_conf;
        torch::Tensor world_points;
        torch::Tensor world_points_conf;
        torch::Tensor images;
    };

    bool build();
    VGGTOutput infer(const std::vector<torch::Tensor> &input_images);

    void printInfo();

private:
    DataType type = FP16;
    std::filesystem::path onnxPath = (type == FP32) ? "D:/vggt_onnx/vggt.onnx" : "D:/vggt_onnx/vggt_fp16.onnx";

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    Logger logger;
};

bool OnnxVGGT::build()
{
    if (!std::filesystem::exists(onnxPath.string() + ".cache"))
    {

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
        if (!builder)
        {
            std::cout << "build failed" << std::endl;
            return false;
        }

        if (!std::filesystem::exists(onnxPath))
        {
            throw std::runtime_error(
                "onnx file not existed!");
        }
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        if (!network)
        {
            std::cout << "network failed" << std::endl;
            return false;
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            std::cout << "config failed" << std::endl;
            return false;
        }

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
        if (!parser)
        {
            std::cout << "parser failed" << std::endl;
            return false;
        }

        bool parsed = parser->parseFromFile(
            onnxPath.string().c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

        if (!parsed)
        {
            throw std::runtime_error(
                "ONNEX parse failed\n");
        }

        for (int i = 0; i < network->getNbInputs(); i++)
        {
            auto *input = network->getInput(i);

            auto dims = input->getDimensions();

            std::cout << "Input: "
                      << input->getName()
                      << "\n";

            for (int d = 0; d < dims.nbDims; d++)
            {
                std::cout << dims.d[d] << " ";
            }

            std::cout << "\n";
        }

        config->setBuilderOptimizationLevel(0);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE,
            2ULL << 30 // 예: 6GB
        );

        auto profile = builder->createOptimizationProfile();

        profile->setDimensions(
            "input_images",
            nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{4, 3, 518, 518});

        profile->setDimensions(
            "input_images",
            nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{4, 3, 518, 518});

        profile->setDimensions(
            "input_images",
            nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{4, 3, 518, 518});

        config->addOptimizationProfile(profile);
        std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            std::cout << "plan failed" << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                std::cout << parser->getError(i)->desc() << std::endl;
            }
            return false;
        }

        std::cout << "plan complete" << std::endl;

        mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!mRuntime)
        {
            std::cout << "mRuntime failed" << std::endl;
            return false;
        }
        std::cout << "mRuntime complete" << std::endl;

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
        if (!mEngine)
        {
            std::cout << "mEngine failed" << std::endl;
            return false;
        }
        std::cout << "mEngine complete" << std::endl;

        std::ofstream engineCache(onnxPath.string() + ".cache", std::ios::binary);
        engineCache.write(
            static_cast<const char *>(plan->data()),
            plan->size());

        engineCache.close();
    }
    else
    {
        std::ifstream engineCache(
            onnxPath.string() + ".cache",
            std::ios::binary | std::ios::ate);

        if (!engineCache)
        {
            std::cout << "engine cache open failed\n";
            return false;
        }

        std::streamsize size = engineCache.tellg();
        engineCache.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);

        if (!engineCache.read(buffer.data(), size))
        {
            std::cout << "engine cache read failed\n";
            return false;
        }

        mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger));

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()),
            InferDeleter());
        engineCache.close();
    }
    return true;
}

OnnxVGGT::VGGTOutput OnnxVGGT::infer(const std::vector<torch::Tensor> &input_images)
{
    constexpr int B = 4;
    constexpr int C = 3;
    constexpr int H = 518;
    constexpr int W = 518;

    auto types = (type == FP16) ? torch::kFloat16 : torch::kFloat32;

    auto context = std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter>(
        mEngine->createExecutionContext());

    if (!context)
        throw std::runtime_error("createExecutionContext failed");

    if (input_images.size() != B)
        throw std::runtime_error("infer expects exactly 4 input images");

    if (!context->setInputShape("input_images", nvinfer1::Dims4{B, C, H, W}))
        throw std::runtime_error("setInputShape failed");

    if (!context->allInputDimensionsSpecified())
        throw std::runtime_error("Not all input dimensions are specified");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const char *inputName = "input_images";

    std::unordered_map<std::string, torch::Tensor> outputTensors;

    std::vector<std::string> outputNames = {
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images"};

    for (const auto &name : outputNames)
    {
        auto dims = context->getTensorShape(name.c_str());
        if (dims.nbDims < 0)
        {
            cudaStreamDestroy(stream);
            throw std::runtime_error("Failed to resolve output tensor shape");
        }

        std::vector<int64_t> shape;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] < 0)
            {
                cudaStreamDestroy(stream);
                throw std::runtime_error("Output tensor has unresolved dynamic dimension");
            }
            shape.push_back(dims.d[i]);
        }

        const auto tensorType = toTorchScalarType(mEngine->getTensorDataType(name.c_str()));
        auto tensor = torch::empty(
            shape,
            torch::TensorOptions()
                .dtype(tensorType)
                .device(torch::kCUDA));

        outputTensors[name] = tensor;

        if (!context->setTensorAddress(name.c_str(), tensor.data_ptr()))
        {
            cudaStreamDestroy(stream);
            throw std::runtime_error(std::string("Failed to set output tensor address: ") + name);
        }
    }

    auto inputTorchType = toTorchScalarType(
        mEngine->getTensorDataType(inputName));

    torch::Tensor batch = torch::zeros(
        {B, C, H, W},
        torch::TensorOptions()
            .dtype(inputTorchType)
            .device(torch::kCUDA));

    for (int i = 0; i < input_images.size(); i++)
    {
        torch::Tensor img = input_images[i];

        if (img.sizes() != torch::IntArrayRef({C, H, W}))
        {
            cudaStreamDestroy(stream);
            throw std::runtime_error("Each input image must have shape [3, 518, 518]");
        }

        img = img.to(torch::kCUDA).to(inputTorchType).contiguous();
        batch[i].copy_(img);
    }

    if (!context->setTensorAddress(inputName, batch.data_ptr()))
    {
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to set input tensor address");
    }

    for (int i = 0; i < mEngine->getNbIOTensors(); ++i)
    {
        const char *name = mEngine->getIOTensorName(i);
        auto mode = mEngine->getTensorIOMode(name);
        auto dtype = mEngine->getTensorDataType(name);
        auto dims = context->getTensorShape(name);

        std::cout << i << " : " << name
                  << " mode=" << static_cast<int>(mode)
                  << " dtype=" << static_cast<int>(dtype)
                  << " shape=";

        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << " ";

        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    bool ok = context->enqueueV3(stream);
    if (!ok)
        throw std::runtime_error("TensorRT enqueueV3 failed");

    cudaStreamSynchronize(stream);

    VGGTOutput result;
    result.pose_enc = outputTensors["pose_enc"];
    result.depth = outputTensors["depth"];
    result.depth_conf = outputTensors["depth_conf"];
    result.world_points = outputTensors["world_points"];
    result.world_points_conf = outputTensors["world_points_conf"];
    result.images = outputTensors["images"];

    cudaStreamDestroy(stream);

    return result;
}

void OnnxVGGT::printInfo()
{
    int nbBindings = mEngine->getNbIOTensors();

    for (int i = 0; i < nbBindings; i++)
    {
        const char *name = mEngine->getIOTensorName(i);

        auto shape = mEngine->getTensorShape(name);

        std::cout << name << ", " << shape.nbDims << std::endl;
    }
}
