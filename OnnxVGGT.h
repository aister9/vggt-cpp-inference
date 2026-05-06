#pragma once
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

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

class OnnxVGGT
{
public:
    bool build();
    bool infer();

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

void OnnxVGGT::printInfo()
{
    int nbBindings = mEngine->getNbIOTensors();

    for (int i = 0; i < nbBindings; i++)
    {
        const char *name = mEngine->getIOTensorName(i);

        auto shape = mEngine->getTensorShape(name);

        std::cout << name << std::endl;
    }
}
