#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <set>
#include <stdexcept>
#include <fstream>
#include <cstring>
#include <iostream>

#ifdef NDEBUG
        const bool enablevalidationLayers = false;
        const bool verbose = false;
#else
        const bool enablevalidationLayers = true;
        const bool verbose = true;
#endif

// Checking for windows OS
#ifdef _WIN32
static std::vector<const char*> s_deviceExtensions = {};
static std::vector<const char*> s_validationLayers = {};

// Checking for mac OS
#elif __APPLE__
static std::vector<const char*> s_deviceExtensions = {"VK_KHR_portability_subset"};
static std::vector<const char*> s_validationLayers = {"VK_LAYER_KHRONOS_validation"};

// Checking for linux OS
#elif __linux__
static std::vector<const char*> s_deviceExtensions = {};
static std::vector<const char*> s_validationLayers = {};

#endif

struct QueueFamilyIndices
{
    uint32_t computeFamily = UINT32_MAX;
    uint32_t computeFamily_value() { return computeFamily % UINT32_MAX; }
    
    bool hasvalue()
    {
        if (computeFamily != UINT32_MAX)
                return true;
        return false;
    }
};

class UseGPU
{
    public:
        UseGPU();
        ~UseGPU();
        
        void program(std::vector<VkDeviceSize> input_sizes, std::vector<VkDeviceSize> output_sizes,
                        std::vector<void*> input_data, std::vector<void*> output_data, 
                        const char* binaryPath, void* pushConstant, VkDeviceSize pushConstantSize, 
                        uint32_t gx, uint32_t gy, uint32_t gz);

    private:
        bool validationCheck();
        void createInstance();
        void pickPhysicalDevice();
        bool isGPUsuitable(VkPhysicalDevice gpu);
        bool checkdeviceExtensions_upport(VkPhysicalDevice gpu);
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice gpu);
        void createLogicalDevice();

        VkShaderModule createShaderModule(const std::vector<char>& code);
        void createComputePipeline();
        void createCommandPool();
        void beginSingleTimeCommands();
        void endSingleTimeCommands();
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        void createDescriptorSetLayout();
        void createDescriptorSets();
        void createDescriptorPool();

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
        void createTensorBuffers(const std::vector<VkDeviceSize>& sizes);
        void uploadInputBuffers();
        void downloadOutputBuffers();
        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
        
        void uploadBarrier();
        void dispatch(uint32_t gx, uint32_t gy, uint32_t gz);
        void downloadBarrier();

        void cleanAfterProgram();

    private:
        VkInstance m_instance{};
        VkPhysicalDevice m_physicalDevice{};
        VkDevice m_device{};
        VkQueue m_computeQueue{};
        uint32_t m_queueFamily{};
        VkCommandPool m_commandPool{};
        VkDescriptorPool m_descriptorPool;
        VkDescriptorSet m_descriptorSet;

        VkDescriptorSetLayout m_computeDescriptorSetLayout;
        VkPipelineLayout m_computePipelineLayout;
        VkPipeline m_computePipeline;
        VkDescriptorSetLayout m_descriptorSetLayout;
        void* m_pushConstant;
        VkDeviceSize m_pushConstantSize;
        const char* m_shaderPath;

        std::vector<void*> m_inputs;
        std::vector<void*> m_outputs;
        std::vector<VkDeviceSize> m_input_sizes;
        std::vector<VkDeviceSize> m_output_sizes;

        std::vector<VkBuffer> m_buffers;
        std::vector<VkDeviceMemory> m_buffersMemory;

        std::vector<VkBuffer> m_input_stagingBuffers;
        std::vector<VkBuffer> m_output_stagingBuffers;

        std::vector<VkDeviceMemory> m_input_stagingBufferMemory;
        std::vector<VkDeviceMemory> m_output_stagingBufferMemory;
        VkCommandBuffer m_commandBuffer;

    public:
        static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
        static std::vector<char> readFile(const char* filename)
        {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open())
                throw std::runtime_error("failed to open spv file!");

            size_t fileSize = (size_t)file.tellg();
            std::vector<char> buffer(fileSize);

            file.seekg(0);
            file.read(buffer.data(), fileSize);
            file.close();

            return buffer;
        }
 
};