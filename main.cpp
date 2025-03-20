#define GLFW_INCLUDE_VULKAN
#include <fstream>
#include <filesystem>
#include <sstream>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <iostream>
#include <unistd.h>
#include <limits.h>
#include <array>
#include <glm/glm.hpp>
#include <sys/stat.h>
#include "/Users/colinleary/Desktop/VulkanSDK/KTX-Software/include/ktx.h"
#include "/Users/colinleary/Desktop/VulkanSDK/KTX-Software/include/ktxvulkan.h"

struct Vertex {
    glm::vec2 pos;    // Position (x, y)
    glm::vec2 texCoord; // Texture Coordinates (u, v)
};

#include <glm/glm.hpp>

void printWorkingDirectory() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current Working Directory: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
    }
}
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_portability_subset"
};
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif
bool checkExtensionSupport(VkPhysicalDevice device, const char* extensionName) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    for (const auto& ext : availableExtensions) {
        if (strcmp(ext.extensionName, extensionName) == 0) {
            return true;
        }
    }
    return false;
}
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {
    
    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            std::cout << "🔍 Checking Memory Type " << i << ": Flags=" << memProperties.memoryTypes[i].propertyFlags << std::endl;
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("❌ ERROR: Failed to find suitable memory type!");
    }


    const std::string TEXTURE_PATH = "/Users/colinleary/Downloads/RoomONE.ktx2";
    void loadTexture() {
        std::cout << "🛠 Entering loadTexture()...\n";

        ktxTexture* texture;
        KTX_error_code result = ktxTexture_CreateFromNamedFile(TEXTURE_PATH.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);

        if (result != KTX_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to load KTX texture: " << TEXTURE_PATH << "\n";
            assert(false && "🔴 CRASH: Failed to load KTX texture!");
        }

        std::cout << "✅ KTX2 Texture loaded successfully!\n";
        std::cout << "🔍 Texture Dimensions: " << texture->baseWidth << " x " << texture->baseHeight << "\n";
        std::cout << "🔍 Mip Levels: " << texture->numLevels << "\n";

        // **Check BC7 format support before deciding transcoding format**
        std::cout << "🟡 Checking BC7 Format Support Before Loading Texture...\n";
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_BC7_UNORM_BLOCK, &formatProperties);

        bool supportsBC7 = formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if (!supportsBC7) {
            std::cerr << "❌ ERROR: GPU does NOT support BC7. Trying ASTC instead.\n";
        } else {
            std::cout << "✅ GPU supports BC7. Proceeding with BC7 format.\n";
        }

        // **Set transcoding format dynamically**
        ktx_transcode_fmt_e targetFormat = supportsBC7 ? KTX_TTF_BC7_RGBA : KTX_TTF_ASTC_4x4_RGBA;

        // **Handle VK_FORMAT_UNDEFINED case**
        VkFormat textureFormat = ktxTexture_GetVkFormat(texture);
        if (textureFormat == VK_FORMAT_UNDEFINED) {
            std::cerr << "❌ ERROR: KTX2 file has VK_FORMAT_UNDEFINED! Attempting to transcode...\n";
            
            if (ktxTexture2_TranscodeBasis(reinterpret_cast<ktxTexture2*>(texture), targetFormat, 0) != KTX_SUCCESS) {
                throw std::runtime_error("❌ ERROR: Transcoding failed!");
            }

            // **Map KTX format to Vulkan format**
            textureFormat = supportsBC7 ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
            std::cout << "✅ Transcoded Format: " << textureFormat << "\n";
        }

        createImage(texture->baseWidth, texture->baseHeight,
                    textureFormat,  // ✅ Using retrieved or transcoded format
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    textureImage, textureImageMemory);

        vkBindImageMemory(device, textureImage, textureImageMemory, 0);

        ktxTexture_Destroy(texture);
    }

    void createTextureImageView() {
        std::cout << "🛠 Entering createTextureImageView()..." << std::endl;
        std::cout << "🔍 textureImage: " << textureImage << std::endl;

        if (textureImage == VK_NULL_HANDLE) {
            std::cerr << "❌ ERROR: textureImage is NULL before creating textureImageView!" << std::endl;
            assert(false && "🔴 CRASH: textureImage is NULL!");
        }

        // Print the Vulkan memory properties of textureImage
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, textureImage, &memRequirements);
        std::cout << "🔍 Texture Image Memory Requirements: Size = " << memRequirements.size
                  << " | Alignment = " << memRequirements.alignment
                  << " | Memory Type Bits = " << memRequirements.memoryTypeBits << std::endl;

        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM);

        if (textureImageView == VK_NULL_HANDLE) {
            std::cerr << "❌ ERROR: textureImageView creation failed!" << std::endl;
            assert(false && "🔴 CRASH: textureImageView creation failed!");
        }

        std::cout << "✅ textureImageView created successfully!" << std::endl;
    }


    void createTextureSampler() {
        std::cout << "🛠 Creating Texture Sampler...\n";

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        VkResult result = vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler);

        if (result != VK_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to create Vulkan texture sampler! VkResult = " << result << "\n";
            assert(false && "🔴 CRASH: Failed to create Vulkan texture sampler!");
        }

        std::cout << "✅ Texture Sampler Created: " << textureSampler << "\n";
    }


    void createDescriptorSet() {
        std::cout << "🛠 Entering createDescriptorSet()..." << std::endl << std::flush;

        // 🔍 Step 1: Ensure descriptorPool is created
        std::cout << "🔍 Creating descriptor pool..." << std::endl << std::flush;
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;

        VkResult poolResult = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
        if (poolResult != VK_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to create descriptor pool! VkResult = " << poolResult << std::endl << std::flush;
            assert(false && "🔴 CRASH: Failed to create descriptor pool!");
        }

        std::cout << "✅ descriptorPool created successfully: " << descriptorPool << std::endl << std::flush;

        // 🔍 Step 2: Allocate Descriptor Set
        std::cout << "🔍 Allocating descriptor set..." << std::endl << std::flush;
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VkResult allocResult = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);
        if (allocResult != VK_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to allocate descriptor set! VkResult = " << allocResult << std::endl << std::flush;
            assert(false && "🔴 CRASH: Failed to allocate descriptor set!");
        }

        std::cout << "✅ descriptorSet allocated successfully: " << descriptorSet << std::endl << std::flush;

        // 🔍 Step 3: Validate Texture Image View & Sampler
        if (textureImageView == VK_NULL_HANDLE) {
            std::cerr << "❌ ERROR: textureImageView is NULL before writing descriptor!" << std::endl << std::flush;
            assert(false && "🔴 CRASH: textureImageView is NULL!");
        }
        if (textureSampler == VK_NULL_HANDLE) {
            std::cerr << "❌ ERROR: textureSampler is NULL before writing descriptor!" << std::endl << std::flush;
            assert(false && "🔴 CRASH: textureSampler is NULL!");
        }

        // 🔍 Step 4: Prepare Descriptor Write
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;  // Ensure this matches shader binding!
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pImageInfo = &imageInfo;

        // Debugging output for descriptor info
        std::cout << "🔍 descriptorSet: " << descriptorSet << std::endl;
        std::cout << "🔍 descriptorWrite.dstBinding: " << descriptorWrite.dstBinding << std::endl;
        std::cout << "🔍 descriptorWrite.descriptorType: " << descriptorWrite.descriptorType << std::endl;
        std::cout << "🔍 descriptorWrite.descriptorCount: " << descriptorWrite.descriptorCount << std::endl;
        std::cout << "🔍 descriptorWrite.pImageInfo: " << descriptorWrite.pImageInfo << std::endl;

        if (descriptorWrite.pImageInfo) {
            std::cout << "🔍 imageInfo.imageView: " << descriptorWrite.pImageInfo->imageView << std::endl;
            std::cout << "🔍 imageInfo.sampler: " << descriptorWrite.pImageInfo->sampler << std::endl;
            std::cout << "🔍 imageInfo.imageLayout: " << descriptorWrite.pImageInfo->imageLayout << std::endl;
        }

        // 🔍 Step 5: Update Descriptor Set
        std::cout << "🔍 Updating descriptor set..." << std::endl << std::flush;
        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        std::cout << "✅ Descriptor set updated successfully!" << std::endl << std::flush;

        // 🔍 Step 6: Additional Debugging After Update
        if (descriptorSet == VK_NULL_HANDLE) {
            std::cerr << "❌ ERROR: descriptorSet is NULL after vkUpdateDescriptorSets!" << std::endl << std::flush;
            assert(false && "🔴 CRASH: descriptorSet is NULL after update!");
        } else {
            std::cout << "✅ Post-update descriptorSet is valid: " << descriptorSet << std::endl << std::flush;
        }
    }

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS) {
            throw std::runtime_error("ERROR: Failed to create semaphores!");
        }
    }
    
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // ✅ Ensure vertexBuffer is valid before binding
        if (vertexBuffer == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: vertexBuffer is NULL before binding!");
        }

        std::cout << "🔍 Binding Vertex Buffer: " << vertexBuffer << std::endl;

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};

        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        // ✅ Ensure descriptorSet is valid before binding
        if (descriptorSet == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: descriptorSet is NULL before binding!");
        }

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        std::cout << "🟢 Calling vkCmdDraw()" << std::endl;
        vkCmdDraw(commandBuffer, 6, 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to record command buffer!");
        }
    }


    void drawFrame() {
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        if (result != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to acquire swap chain image!");
        }

        // 🛠 **Record the command buffer before submitting**
        recordCommandBuffer(commandBuffers[imageIndex], imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (graphicsQueue == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: graphicsQueue is NULL before submission!");
        }

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &imageIndex;

        vkQueuePresentKHR(presentQueue, &presentInfo);
    }


    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule;
        VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
        if (result != VK_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to create shader module! Error Code: " << result << std::endl;
            throw std::runtime_error("failed to create shader module!");
        }
        std::cout << "✅ Shader module created successfully!" << std::endl;
        return shaderModule;
    }
    std::vector<char> readFile(const std::string& filename) {
        // 🔹 Debug: Print the full file path
        std::cout << "Attempting to open file: " << filename << std::endl;

        // 🕒 Check last modified time before opening the file
        struct stat result;
        if (stat(filename.c_str(), &result) == 0) {
            std::cout << "🕒 Shader Last Modified: " << ctime(&result.st_mtime);
        } else {
            std::cerr << "❌ ERROR: Could not retrieve file modification time!" << std::endl;
        }

        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file: " + filename);
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    
    VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        // Position (vec2) -> Location 0
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        // Texture Coordinates (vec2) -> Location 1
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
    
public:
    
    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to begin command buffer!");
        }

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to end command buffer!");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    
    
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
private:
    VkDescriptorPool descriptorPool;  // Declare descriptor pool
    VkDescriptorSetLayout descriptorSetLayout;
    
    VkDescriptorSet descriptorSet;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkPipeline graphicsPipeline;
    VkPipelineLayout pipelineLayout;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    uint32_t textureWidth;
    uint32_t textureHeight;
    
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    VkImageView createImageView(VkImage image, VkFormat format);

    
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    
    void createFramebuffers() {
        std::cout << "🛠 Entering createFramebuffers()...\n";

        if (!swapChainFramebuffers.empty()) {
            std::cout << "🗑 Destroying existing framebuffers...\n";
            for (auto framebuffer : swapChainFramebuffers) {
                vkDestroyFramebuffer(device, framebuffer, nullptr);
            }
            swapChainFramebuffers.clear();
        }

        swapChainFramebuffers.resize(swapChainImageViews.size(), VK_NULL_HANDLE);
        std::cout << "🔍 Resized swapChainFramebuffers to " << swapChainFramebuffers.size() << " entries.\n";

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = { swapChainImageViews[i] };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            VkResult result = vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("❌ ERROR: Failed to create framebuffer " + std::to_string(i) + "!");
            }

            std::cout << "✅ Framebuffer " << i << " created successfully! Handle: " << swapChainFramebuffers[i] << std::endl;
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        if (!queueFamilyIndices.graphicsFamily.has_value()) {
            throw std::runtime_error("❌ ERROR: No valid graphics queue family found!");
        }

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
        
        if (result != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to create command pool! VkResult: " + std::to_string(result));
        }

        if (commandPool == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: commandPool is NULL after creation!");
        }

        std::cout << "✅ Command Pool Created Successfully!\n";
    }


    void createCommandBuffers(VkDevice device, VkCommandPool commandPool, std::vector<VkCommandBuffer>& commandBuffers, uint32_t bufferCount) {
        // Reset the command pool to allow command buffer reallocation
        if (vkResetCommandPool(device, commandPool, 0) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to reset command pool!");
        }

        // Allocate command buffers
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = bufferCount;

        commandBuffers.resize(bufferCount);
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to allocate command buffers!");
        }
    }




    VkRenderPass renderPass;
    GLFWwindow* window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }
    
    
    
    std::vector<Vertex> tileVertices; // Store tile vertices globally in the class
    void createVertexBuffer() {
        std::cout << "🛠 Entering createVertexBuffer()..." << std::endl;

        // Define a fullscreen quad with texture coordinates
        std::vector<Vertex> quadVertices = {
            {{-1.0f, -1.0f}, {0.0f, 1.0f}},  // Bottom-left
            {{1.0f, -1.0f}, {1.0f, 1.0f}},   // Bottom-right
            {{-1.0f, 1.0f}, {0.0f, 0.0f}},   // Top-left
            {{1.0f, 1.0f}, {1.0f, 0.0f}},    // Top-right
        };

        VkDeviceSize bufferSize = sizeof(quadVertices[0]) * quadVertices.size();

        // Create Vulkan buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult bufferResult = vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer);
        if (bufferResult != VK_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to create vertex buffer! VkResult: " << bufferResult << std::endl;
            throw std::runtime_error("❌ ERROR: Failed to create vertex buffer!");
        }

        std::cout << "✅ vertexBuffer Created: " << vertexBuffer << std::endl;

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkResult allocResult = vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory);
        if (allocResult != VK_SUCCESS) {
            std::cerr << "❌ ERROR: Failed to allocate vertex buffer memory! VkResult: " << allocResult << std::endl;
            throw std::runtime_error("❌ ERROR: Failed to allocate vertex buffer memory!");
        }

        std::cout << "✅ vertexBufferMemory Allocated: " << vertexBufferMemory << std::endl;

        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

        if (vertexBuffer == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: vertexBuffer is NULL after binding!");
        }

        std::cout << "✅ vertexBuffer Memory Bound Successfully!\n";

        // Copy vertex data to buffer memory
        void* data;
        vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, quadVertices.data(), (size_t) bufferSize);  // 🔹 Copy quad vertices into Vulkan memory
        vkUnmapMemory(device, vertexBufferMemory);

        std::cout << "✅ Vulkan Vertex Buffer Updated Successfully with Fullscreen Quad!\n";
    }

    void initVulkan() {
        std::cout << "🛠 Entering initVulkan..." << std::endl;

        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createPipelineLayout();
        createGraphicsPipeline();

        std::cout << "🟡 Checking BC7 Format Support Before Loading Texture..." << std::endl;

        // 🔍 Check if GPU supports BC7 before attempting to load texture
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_BC7_UNORM_BLOCK, &formatProperties);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            std::cerr << "❌ ERROR: GPU does NOT support BC7. Trying ASTC instead..." << std::endl;

            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, &formatProperties);
            if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
                throw std::runtime_error("❌ ERROR: GPU does NOT support BC7 or ASTC! Cannot load texture.");
            }

            std::cout << "✅ GPU supports ASTC. Using ASTC instead of BC7.\n";
        } else {
            std::cout << "✅ GPU supports BC7. Proceeding with BC7 format.\n";
        }

        std::cout << "🟡 Calling loadTexture() before descriptor set creation..." << std::endl;
        loadTexture();  // ✅ Now safe to load the texture

        std::cout << "🔍 Calling createTextureImageView()..." << std::endl;
        createTextureImageView();

        std::cout << "🔍 Calling createTextureSampler()..." << std::endl;
        createTextureSampler();

        std::cout << "🟡 BEFORE createDescriptorSet()" << std::endl;
        createDescriptorSet();

        std::cout << "🟡 Calling createVertexBuffer()..." << std::endl;
        createVertexBuffer();  // ✅ Ensures vertex buffer is created

        std::cout << "🔍 Calling createCommandBuffers()...\n";
        createFramebuffers();
        createCommandPool();

        if (commandPool == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: commandPool was not created! Did createCommandPool() run?");
        }

        std::cout << "🔍 Creating command buffers...\n";
        uint32_t bufferCount = 3; // Adjust based on the number of frames in flight
        createCommandBuffers(device, commandPool, commandBuffers, bufferCount);



        std::cout << "🔍 Creating synchronization objects...\n";
        createSyncObjects();

        std::cout << "✅ Vulkan initialization complete!\n";
    }




    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        VkResult renderPassResult = vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
        if (renderPassResult != VK_SUCCESS) {
            throw std::runtime_error("ERROR: Failed to create render pass!");
        }
        // **Debug Check**
        if (renderPass == VK_NULL_HANDLE) {
            throw std::runtime_error("ERROR: renderPass is NULL after creation!");
        } else {
            std::cout << "✅ Render Pass Created Successfully: " << renderPass << std::endl;
        }
    }
    
    void createPipelineLayout() {
        // Define the sampler descriptor layout binding
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 0;  // Matches the binding in the fragment shader
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        samplerLayoutBinding.pImmutableSamplers = nullptr;

        // Define descriptor set layout creation
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &samplerLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to create descriptor set layout!");
        }

        // Define pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;  // Use the descriptor set layout

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("❌ ERROR: Failed to create pipeline layout!");
        }
    }


    // Function to convert VkResult to a human-readable string
    std::string getVkResultString(VkResult result) {
        switch (result) {
            case VK_SUCCESS: return "VK_SUCCESS";
            case VK_NOT_READY: return "VK_NOT_READY";
            case VK_TIMEOUT: return "VK_TIMEOUT";
            case VK_EVENT_SET: return "VK_EVENT_SET";
            case VK_EVENT_RESET: return "VK_EVENT_RESET";
            case VK_INCOMPLETE: return "VK_INCOMPLETE";
            case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
            case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
            case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
            case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
            case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
            case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
            case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
            case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
            case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
            case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
            case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
            default: return "UNKNOWN_VK_RESULT";
        }
    }


    void createGraphicsPipeline() {
        VkVertexInputBindingDescription bindingDescription = getBindingDescription();
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Now valid
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        std::cout << "🛠 Entering createGraphicsPipeline()...\n";

        if (graphicsPipeline != VK_NULL_HANDLE) {
            std::cout << "🗑 Destroying old graphics pipeline before recreation...\n";
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            graphicsPipeline = VK_NULL_HANDLE;
        }

        std::string vertShaderPath = "/Users/colinleary/desktop/VulkanSDK/LittleVulkanEngine/shaders/triangle.vert.spv";
        std::string fragShaderPath = "/Users/colinleary/desktop/VulkanSDK/LittleVulkanEngine/shaders/triangle.frag.spv";

        std::cout << "🔍 Loading Vertex Shader: " << vertShaderPath << "\n";
        std::cout << "🔍 Loading Fragment Shader: " << fragShaderPath << "\n";

        auto vertShaderCode = readFile(vertShaderPath);
        auto fragShaderCode = readFile(fragShaderPath);

        std::cout << "Vertex Shader Size: " << vertShaderCode.size() << " bytes\n";
        std::cout << "Fragment Shader Size: " << fragShaderCode.size() << " bytes\n";

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        if (vertShaderModule == VK_NULL_HANDLE || fragShaderModule == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: Shader modules were not created successfully!");
        }

        VkPipelineShaderStageCreateInfo shaderStages[] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertShaderModule, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderModule, "main", nullptr}
        };


        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{0.0f, 0.0f, (float) swapChainExtent.width, (float) swapChainExtent.height, 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, swapChainExtent};

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        if (pipelineLayout == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: pipelineLayout is NULL before graphics pipeline creation!");
        }

        if (renderPass == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: renderPass is NULL before graphics pipeline creation!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        std::cout << "🟢 Debugging Graphics Pipeline Creation...\n";
        std::cout << "🟢 Device: " << device << "\n";
        std::cout << "🟢 Pipeline Layout: " << pipelineLayout << "\n";
        std::cout << "🟢 Render Pass: " << renderPass << "\n";
        std::cout << "🟢 Vertex Shader Module: " << vertShaderModule << "\n";
        std::cout << "🟢 Fragment Shader Module: " << fragShaderModule << "\n";

        VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
        if (result != VK_SUCCESS) {
            std::cerr << "❌ ERROR: vkCreateGraphicsPipelines failed! VkResult = " << result << "\n";
            throw std::runtime_error("❌ ERROR: Graphics pipeline creation failed!");
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);

        std::cout << "✅ Graphics Pipeline Fully Rebuilt!\n";
    }


    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
    }
    void cleanup() {
        // ✅ Ensure GPU has finished all work before destroying resources
        vkDeviceWaitIdle(device);
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
            commandPool = VK_NULL_HANDLE;
        }
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }
    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        auto extensions = getRequiredExtensions();
         extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME); //     Add portability enumeration extensions
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }
    
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }
    
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);
        // Bypass validation check for portability subset if necessary
        createInfo.messageSeverity &= ~VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }
    
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        for (const auto& device : devices) {
            VkPhysicalDeviceProperties deviceProperties;
            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            std::cout << "Detected GPU: " << deviceProperties.deviceName << std::endl;
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                std::cout << "Selected Vulkan GPU: " << deviceProperties.deviceName << std::endl;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("ERROR: Failed to find a suitable GPU!");
        }
    }
    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        float queuePriority = 1.0f;
        
        for (uint32_t queueFamily : uniqueQueueFamilies) {
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
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("ERROR: Failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

        if (graphicsQueue == VK_NULL_HANDLE || presentQueue == VK_NULL_HANDLE) {
            throw std::runtime_error("❌ ERROR: Failed to retrieve graphics or present queue!");
        }

        std::cout << "✅ graphicsQueue assigned: " << graphicsQueue << std::endl;
        std::cout << "✅ presentQueue assigned: " << presentQueue << std::endl;
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
    
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }
    bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        std::cout << "Checking Vulkan device: " << deviceProperties.deviceName << std::endl;
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool supportsPortabilitySubset = checkDeviceExtensionSupport(device) &&
                                         checkExtensionSupport(device, "VK_KHR_portability_subset");
        std::cout << "Queue Families Complete: " << indices.isComplete() << std::endl;
        std::cout << "Extensions Supported: " << extensionsSupported << std::endl;
        std::cout << "Portability Subset Supported: " << supportsPortabilitySubset << std::endl;
        return indices.isComplete() && extensionsSupported && supportsPortabilitySubset;
    }
    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        std::cout << "Checking required extensions: " << std::endl;
        for (const auto& ext : requiredExtensions) {
            std::cout << "\t" << ext << std::endl;
        }
        std::cout << "Matching required extensions with available ones..." << std::endl;
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        if (!requiredExtensions.empty()) {
            std::cout << "Missing extensions: " << std::endl;
            for (const auto& missing : requiredExtensions) {
                std::cout << "\t" << missing << std::endl;
            }
        }
        return requiredExtensions.empty();
    }
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }
            i++;
        }
        if (!indices.isComplete()) {
            throw std::runtime_error("ERROR: Failed to find required queue families!");
        }
        return indices;
    }
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }
    
    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                return false;
            }
        }
        return true;
    }
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
    
    
};
int main() {
    printWorkingDirectory();
    HelloTriangleApplication app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// Implementation of missing functions

void HelloTriangleApplication::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                 VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {

    if (format == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error("❌ ERROR: Trying to create an image with VK_FORMAT_UNDEFINED!");
    }

    std::cout << "🛠 Creating Vulkan Image: " << width << "x" << height << "\n";

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // ✅ Ensure format mutability for BC7 compatibility
    imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

    // ✅ Ensure proper texture usage flags
    imageInfo.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VkResult result = vkCreateImage(device, &imageInfo, nullptr, &image);
    if (result != VK_SUCCESS) {
        std::cerr << "❌ ERROR: Failed to create Vulkan Image! VkResult = " << result << "\n";
        throw std::runtime_error("❌ Vulkan Image creation failed!");
    }

    std::cout << "✅ Vulkan Image Created: " << image << "\n";
}


void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("❌ ERROR: Unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}


void HelloTriangleApplication::copyBufferToImage(VkBuffer buffer, VkImage image,
                                                 uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

VkImageView HelloTriangleApplication::createImageView(VkImage image, VkFormat format) {
    if (image == VK_NULL_HANDLE) {
        throw std::runtime_error("❌ ERROR: Attempting to create image view for NULL image!");
    }

    std::cout << "🔍 Creating Image View for Vulkan Image: " << image << std::endl;

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

    // ✅ Use the **actual** transcoded format (BC7) to avoid incompatibility
    viewInfo.format = format;

    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    // ✅ Ensure proper format compatibility
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    VkImageView imageView;
    VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &imageView);
    if (result != VK_SUCCESS) {
        std::cerr << "❌ ERROR: Failed to create Image View! VkResult = "
                  << result << " (" << getVkResultString(result) << ")" << std::endl;
        throw std::runtime_error("❌ ERROR: Image View creation failed!");
    }

    std::cout << "✅ Image View Created: " << imageView << std::endl;
    return imageView;
}


bool checkFormatSupport(VkPhysicalDevice physicalDevice, VkFormat format) {
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
        std::cerr << "❌ ERROR: GPU does NOT support the required format: " << format << "\n";
        return false;
    }
    return true;
}

void recordAndSubmitCommandBuffer(VkCommandBuffer commandBuffer, VkQueue graphicsQueue) {
    // End command buffer recording before submission
    VkResult res = vkEndCommandBuffer(commandBuffer);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("❌ ERROR: Failed to record command buffer!");
    }

    // Prepare the submit info
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Submit the command buffer to the queue
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("❌ ERROR: Failed to submit draw command buffer!");
    }

    // Ensure the queue finishes execution
    vkQueueWaitIdle(graphicsQueue);
}

void createVulkanInstance(VkInstance& instance) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Little Vulkan Engine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Enable debug validation layers
    const char* validationLayers[] = { "VK_LAYER_KHRONOS_validation" };
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = validationLayers;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("❌ ERROR: Failed to create Vulkan instance!");
    }
}

