#include "../include/use_GPU.h"
#include <iostream>

UseGPU::UseGPU() {
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
}

UseGPU::~UseGPU() {
    vkDeviceWaitIdle(m_device);
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    vkDestroyDevice(m_device, nullptr);
    vkDestroyInstance(m_instance, nullptr);
}

bool UseGPU::validationCheck() // debug only
{
    // get avaliable layers
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    // check if req in avaliable
    for (const char* layerName : s_validationLayers)
    {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers)
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }

        if (!layerFound)
            return false;
    }

    return true;
}

void UseGPU::createInstance()
{
    // validation in debug
    if (enablevalidationLayers && !validationCheck())
        throw std::runtime_error("validation layers requested, but not available!");

    // begin
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "cpp_ml";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // if validation on in debug
    if (enablevalidationLayers) 
    {
        createInfo.enabledLayerCount = (uint32_t)s_validationLayers.size();
        createInfo.ppEnabledLayerNames = s_validationLayers.data();
    } 
    else createInfo.enabledLayerCount = 0;


    std::vector<const char*> extensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, // macOS stuff
        "VK_KHR_get_physical_device_properties2"       // macOS stuff
    };
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR; // macOS stuff

    // back to main...
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");

    // vk extensions
    uint32_t vk_extension_ct = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &vk_extension_ct, nullptr);
    std::vector<VkExtensionProperties> all_vk_Extensions(vk_extension_ct);
    vkEnumerateInstanceExtensionProperties(nullptr, &vk_extension_ct, all_vk_Extensions.data());
}

void UseGPU::pickPhysicalDevice()
{
    // get gpus
    uint32_t num_physicalDevices = 0;
    vkEnumeratePhysicalDevices(m_instance, &num_physicalDevices, nullptr);
    if (!num_physicalDevices)
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    if (verbose)
        std::cout << "[INFO ] number of gpus: " << num_physicalDevices << '\n';

    std::vector<VkPhysicalDevice> gpus(num_physicalDevices);
    vkEnumeratePhysicalDevices(m_instance, &num_physicalDevices, gpus.data());

    for (const auto& gpu : gpus)
        if (isGPUsuitable(gpu)) 
            { 
                m_physicalDevice = gpu;
                break;  // only get first gpu
            }

    if (m_physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("failed to find a suitable GPU!");
}

bool UseGPU::isGPUsuitable(VkPhysicalDevice gpu)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(gpu, &properties);

    VkPhysicalDeviceFeatures opt_features;
    vkGetPhysicalDeviceFeatures(gpu, &opt_features);

    if (verbose)
    {
        std::cout << "[INFO ] Using GPU: " << properties.deviceName << '\n';
        std::cout << "[INFO ] maxComputeWorkGroupCount: ["
                        << properties.limits.maxComputeWorkGroupCount[0] << ", "
                        << properties.limits.maxComputeWorkGroupCount[1] << ", "
                        << properties.limits.maxComputeWorkGroupCount[2] << "]\n";
        std::cout << "[INFO ] maxComputeWorkGroupSize: ["
                        << properties.limits.maxComputeWorkGroupSize[0] << ", "
                        << properties.limits.maxComputeWorkGroupSize[1] << ", "
                        << properties.limits.maxComputeWorkGroupSize[2] << "]\n";
        std::cout << "[INFO ] maxComputeWorkGroupInvocations: " << properties.limits.maxComputeWorkGroupInvocations << '\n';
    }

    bool good = true;

    // examples
    // good &= properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    // good &= opt_features.geometryShader;
    // good &= opt_features.samplerAnisotropy;

    QueueFamilyIndices indices = findQueueFamilies(gpu);
    good &= indices.hasvalue();

    bool extensionsSupported = checkdeviceExtensions_upport(gpu);
    good &= extensionsSupported;

    return good;
}

bool UseGPU::checkdeviceExtensions_upport(VkPhysicalDevice gpu)
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, availableExtensions.data());

    // check if req in avaliable
    for (uint32_t i = 0; i < (uint32_t)s_deviceExtensions.size(); i++)
    {
        bool extensionFound = false;
        for (const auto& extension : availableExtensions)
            {
                if (strcmp(s_deviceExtensions[i], extension.extensionName) == 0)
                { 
                    extensionFound = true;
                    if (verbose)
                        std::cout << "[INFO ] Extension found: " << extension.extensionName << '\n';
                }
                if (verbose)
                {
                    std::cout << "[INFO ] Extension available: " << extension.extensionName << '\n';
                }
            }
        
        if (!extensionFound)
            return false;
    }

    return true;

}

QueueFamilyIndices UseGPU::findQueueFamilies(VkPhysicalDevice gpu)
{
    QueueFamilyIndices indices;

    uint32_t num_queue_fam = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queue_fam, nullptr);
    std::vector<VkQueueFamilyProperties> queue_fam(num_queue_fam);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &num_queue_fam, queue_fam.data());

    uint32_t i = 0;
    for (const auto& queue : queue_fam)
    {
            if (queue.queueFlags & VK_QUEUE_COMPUTE_BIT) 
                {
                    indices.computeFamily = i;
                    if (verbose)
                        std::cout << "[INFO ] compute queue family " << i << '\n';
                }
            i++;
    }

    return indices;
}

void UseGPU::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.computeFamily_value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount = (uint32_t)(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = (uint32_t)(s_deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = s_deviceExtensions.data();

    if (enablevalidationLayers)
    {
        createInfo.enabledLayerCount = (uint32_t)(s_validationLayers.size());
        createInfo.ppEnabledLayerNames = s_validationLayers.data();
    }
    else
        createInfo.enabledLayerCount = 0;
    
    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device!");

    vkGetDeviceQueue(m_device, indices.computeFamily_value(), /*queue index*/0, &m_computeQueue);

}

VkShaderModule UseGPU::createShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module!");
    
    return shaderModule;
}

void UseGPU::createComputePipeline()
{
    auto computeShaderCode = readFile(m_shaderPath);
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = m_pushConstantSize;

    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_computePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline layout!");

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_computePipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline!");

    vkDestroyShaderModule(m_device, computeShaderModule, nullptr);
    
}

void UseGPU::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily_value();
    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create command pool!");
}

void UseGPU::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

}

void UseGPU::endSingleTimeCommands()
{
    vkEndCommandBuffer(m_commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;

    vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_computeQueue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_commandBuffer);
}

uint32_t UseGPU::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    throw std::runtime_error("failed to find suitable memory type!");

}

void  UseGPU::createDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings((uint32_t)m_buffers.size());
    for (int i = 0; i < (uint32_t)bindings.size(); i++)
    {
        bindings[i].binding = i; 
        bindings[i].descriptorCount = 1; 
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; 
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void UseGPU::createDescriptorSets()
{
    
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_descriptorSet) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor sets!");

    std::vector<VkDescriptorBufferInfo> bufferInfos((uint32_t)m_buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites((uint32_t)m_buffers.size());
    for (int i = 0; i < (uint32_t)m_buffers.size(); i++)
    {
       
        bufferInfos[i].buffer = m_buffers[i]; // A.buf, B.buf, C.buf
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = VK_WHOLE_SIZE;
        
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = m_descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(m_device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void UseGPU::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer!");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties); // prop = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate buffer memory!");

    vkBindBufferMemory(m_device, buffer, bufferMemory, 0);

    
}

void UseGPU::createTensorBuffers(const std::vector<VkDeviceSize>& sizes)
{
    for (const VkDeviceSize size : sizes)
    {
        VkDeviceSize bufferSize = size;
        VkBuffer buffer;
        VkDeviceMemory bufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            buffer, bufferMemory);

        m_buffers.push_back(buffer);
        m_buffersMemory.push_back(bufferMemory);
    }
}

void UseGPU::uploadInputBuffers()
{
    for (uint32_t i = 0; i < (uint32_t)m_inputs.size(); i++)
    {
        VkDeviceSize bufferSize = m_input_sizes[i];

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            m_input_stagingBuffers[i], m_input_stagingBufferMemory[i]);

        void* data;
        vkMapMemory(m_device, m_input_stagingBufferMemory[i], 0, bufferSize, 0, &data);
        memcpy(data, /*src=*/m_inputs[i], bufferSize);
        vkUnmapMemory(m_device, m_input_stagingBufferMemory[i]);

    }
}

void UseGPU::downloadOutputBuffers()
{
    uint32_t offset = (uint32_t)m_inputs.size(); // m_buffer is input+output so I add and offset
    for (uint32_t i = 0; i < (uint32_t)m_outputs.size(); i++)
    {
        
        VkDeviceSize bufferSize = m_output_sizes[i];
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            m_output_stagingBuffers[i], m_output_stagingBufferMemory[i]);

        copyBuffer(m_buffers[i + offset], m_output_stagingBuffers[i], bufferSize);
    }
}

void UseGPU::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(m_commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
}

void UseGPU::createDescriptorPool()
{
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = (uint32_t)m_buffers.size();  // TODO DONT USE 3

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");

}

void UseGPU::dispatch(uint32_t gx, uint32_t gy, uint32_t gz)
{
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

    if (m_pushConstantSize)
        vkCmdPushConstants(m_commandBuffer, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, m_pushConstantSize, m_pushConstant);
    vkCmdDispatch(m_commandBuffer, gx, gy, gz);
}

void UseGPU::uploadBarrier()
{
    std::vector<VkBufferMemoryBarrier> barriers(m_inputs.size());
    for (uint32_t i = 0; i < (uint32_t)m_inputs.size(); i++)
    {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].buffer = m_buffers[i];
        barriers[i].offset = 0;
        barriers[i].size = VK_WHOLE_SIZE;
    }
    
    vkCmdPipelineBarrier(
        m_commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr,
        (uint32_t)barriers.size(), barriers.data(),
        0, nullptr);
}

void UseGPU::downloadBarrier()
{
    uint32_t offset = (uint32_t)m_inputs.size();
    std::vector<VkBufferMemoryBarrier> barriers((uint32_t)m_output_sizes.size());

    for (uint32_t i = 0; i < (uint32_t)m_output_sizes.size(); i++)
    {
        barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[i].buffer = m_buffers[offset + i];
        barriers[i].offset = 0;
        barriers[i].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(
        m_commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr,
        (uint32_t)barriers.size(), barriers.data(),
        0, nullptr);
}

void UseGPU::cleanAfterProgram()
{
    for (uint32_t i = 0; i < (uint32_t)m_outputs.size(); i++)
    {
        VkDeviceSize bufferSize = m_output_sizes[i];
        void* data;
        vkMapMemory(m_device, m_output_stagingBufferMemory[i], 0, bufferSize, 0, &data);
        memcpy(m_outputs[i], /*src=*/data, bufferSize);
        vkUnmapMemory(m_device, m_output_stagingBufferMemory[i]);      
    }
    for (auto& buffer : m_input_stagingBuffers)
        vkDestroyBuffer(m_device, buffer, nullptr);
    for (auto& memory : m_input_stagingBufferMemory)
        vkFreeMemory(m_device, memory, nullptr);
    for (auto& buffer : m_output_stagingBuffers)
        vkDestroyBuffer(m_device, buffer, nullptr);
    for (auto& memory : m_output_stagingBufferMemory)
        vkFreeMemory(m_device, memory, nullptr);

    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_computePipelineLayout, nullptr);

    for (size_t i = 0; i < (uint32_t)m_buffers.size(); i++)
    {
        vkDestroyBuffer(m_device, m_buffers[i], nullptr);
        vkFreeMemory(m_device, m_buffersMemory[i], nullptr);
    }

    m_buffers.clear();
    m_buffersMemory.clear();
    m_inputs.clear();
    m_outputs.clear();
    m_input_sizes.clear();
    m_output_sizes.clear();

    m_input_stagingBuffers.clear();
    m_input_stagingBufferMemory.clear();

    m_output_stagingBuffers.clear();
    m_output_stagingBufferMemory.clear();

}

void UseGPU::program(std::vector<VkDeviceSize> input_sizes, std::vector<VkDeviceSize> output_sizes, 
                            std::vector<void*> input_data, std::vector<void*> output_data, 
                            const char* binaryPath, void* pushConstant, VkDeviceSize pushConstantSize, 
                            uint32_t gx, uint32_t gy, uint32_t gz)
{
    m_input_sizes = input_sizes;
    m_output_sizes = output_sizes;
    m_inputs = input_data;
    m_outputs = output_data;
    m_pushConstant = pushConstant;
    m_pushConstantSize = pushConstantSize;
    m_shaderPath = binaryPath;

    m_input_stagingBuffers.resize((uint32_t)m_inputs.size());
    m_input_stagingBufferMemory.resize((uint32_t)m_inputs.size());

    m_output_stagingBuffers.resize((uint32_t)m_outputs.size());
    m_output_stagingBufferMemory.resize((uint32_t)m_outputs.size());

    createTensorBuffers(m_input_sizes);
    createTensorBuffers(m_output_sizes);
    createDescriptorSetLayout(); // -> m_descriptorSetLayout
    createComputePipeline(); // -> m_computePipelineLayout, m_computePipeline
    createDescriptorPool(); // -> m_descriptorPool
    createDescriptorSets();  // -> m_descriptorSets

    uploadInputBuffers();
    beginSingleTimeCommands();
    for (uint32_t i = 0; i < (uint32_t)m_inputs.size(); i++)
        copyBuffer(m_input_stagingBuffers[i], m_buffers[i], m_input_sizes[i]);
    uploadBarrier();
    dispatch(gx, gy, gz);
    downloadBarrier();
    downloadOutputBuffers();
    endSingleTimeCommands();

    cleanAfterProgram();
}