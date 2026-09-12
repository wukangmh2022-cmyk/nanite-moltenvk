// Feasibility probe for route B: hand-written MSL submitted straight to MoltenVK
// through its private kMVKMagicNumberMSLSourceCode path, so a 64-bit
// atomic_max_explicit can be reached without SPIRV-Cross refusing to translate it.
//
// vkSetWorkgroupSizeMVK - which MSL compute modules need, since MSL carries no
// workgroup size - is documented as unavailable through the Vulkan loader, so it is
// resolved with dlsym instead of linked. That makes the probe work both linked
// against libMoltenVK directly and going through the loader, which is the
// configuration the renderer itself uses:
//
//   direct: clang++ -std=c++17 -I/opt/homebrew/include probe.cpp -o probe_direct \
//               /opt/homebrew/lib/libMoltenVK.dylib
//   loader: clang++ -std=c++17 -I/opt/homebrew/include probe.cpp -o probe_loader \
//               /opt/homebrew/lib/libvulkan.dylib
//
// Run with MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=0. With argument buffers on,
// descriptor set 0 arrives as a Metal argument buffer at [[buffer(0)]] and no plain
// device pointer parameter can ever see the storage buffer.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <dlfcn.h>

#include <vulkan/vulkan.h>
#include <MoltenVK/mvk_private_api.h>

using PFN_SetWorkgroupSizeMVK = void(VKAPI_PTR*)(VkShaderModule, uint32_t, uint32_t, uint32_t);

// The loader does not forward this one, and RTLD_DEFAULT only finds it when
// MoltenVK happens to be in the global namespace. Asking dlopen for the ICD by name
// returns the already-loaded image rather than a second copy, so the handle refers
// to the same MoltenVK that owns the shader module.
static PFN_SetWorkgroupSizeMVK ResolveSetWorkgroupSize(const char** whereFrom)
{
    if (void* sym = dlsym(RTLD_DEFAULT, "vkSetWorkgroupSizeMVK"))
    {
        *whereFrom = "RTLD_DEFAULT";
        return reinterpret_cast<PFN_SetWorkgroupSizeMVK>(sym);
    }
    for (const char* path : {"libMoltenVK.dylib", "/opt/homebrew/lib/libMoltenVK.dylib"})
    {
        void* handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr)
            continue;
        if (void* sym = dlsym(handle, "vkSetWorkgroupSizeMVK"))
        {
            *whereFrom = path;
            return reinterpret_cast<PFN_SetWorkgroupSizeMVK>(sym);
        }
    }
    *whereFrom = "not found";
    return nullptr;
}

#define CHECK(expr)                                                                 \
    do {                                                                            \
        const VkResult result_ = (expr);                                             \
        if (result_ != VK_SUCCESS) {                                                 \
            std::printf("FAIL %s -> VkResult %d\n", #expr, static_cast<int>(result_));\
            return 1;                                                                \
        }                                                                            \
    } while (false)

// 64-bit atomic max, the operation SPIRV-Cross throws on. Two of them: a plain
// max over a shared slot, and a pack of (depth, id) in one 64-bit word, which is
// the shape the visibility buffer actually needs.
static const char* kMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void main0(device atomic_ulong* dst [[buffer(%d)]],
                  uint tid [[thread_position_in_grid]])
{
    // Slot 0: plain 64-bit max over the thread index.
    atomic_max_explicit(&dst[0], (ulong)tid, memory_order_relaxed);

    // Slot 1: the visibility-buffer shape. High 32 bits are a key that must win by
    // being larger, low 32 bits are the payload dragged along with it. Threads
    // contribute keys in a jumbled order so a 32-bit-only max would give a
    // different answer.
    const uint key = (tid * 2654435761u) >> 8;
    const ulong packed = ((ulong)key << 32) | (ulong)(tid + 1000u);
    atomic_max_explicit(&dst[1], packed, memory_order_relaxed);
}
)MSL";

int main(int argc, char** argv)
{
    // "spv" runs an equivalent SPIR-V shader instead, so MoltenVK's own generated
    // MSL can be compared against the hand-written one - in particular the Metal
    // buffer index it picks for descriptor set 0 binding 0.
    const bool useSpirv = argc > 1 && std::strcmp(argv[1], "spv") == 0;
    // "msl-buffer=N" overrides the [[buffer(N)]] index the hand-written MSL uses.
    int mslBufferIndex = 0;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "msl-buffer=", 11) == 0)
            mslBufferIndex = std::atoi(argv[i] + 11);
    std::printf("mode: %s", useSpirv ? "spirv\n" : "msl");
    if (!useSpirv) std::printf(" buffer(%d)\n", mslBufferIndex);

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceCI{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceCI.pApplicationInfo = &appInfo;
    // MoltenVK is a portability driver, so the loader hides it unless enumeration is
    // asked for explicitly. Harmless when linked straight against MoltenVK, which
    // reports the extension and ignores the flag.
    const char* instanceExtensions[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
    instanceCI.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instanceCI.enabledExtensionCount = 1;
    instanceCI.ppEnabledExtensionNames = instanceExtensions;
    VkInstance instance = VK_NULL_HANDLE;
    CHECK(vkCreateInstance(&instanceCI, nullptr, &instance));

    uint32_t physicalCount = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr));
    if (physicalCount == 0) { std::printf("FAIL no physical devices\n"); return 1; }
    std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
    CHECK(vkEnumeratePhysicalDevices(instance, &physicalCount, physicalDevices.data()));
    VkPhysicalDevice physical = physicalDevices[0];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    std::printf("device: %s\n", props.deviceName);

    // What Vulkan claims about 64-bit atomics, for the record. It says no, because
    // Vulkan demands the full add/exchange/CAS/min/max set and Metal only has
    // min/max. The MSL below uses only what Metal does have.
    VkPhysicalDeviceShaderAtomicInt64Features atomic64{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &atomic64;
    vkGetPhysicalDeviceFeatures2(physical, &features2);
    std::printf("shaderBufferInt64Atomics reported by Vulkan: %u\n",
                atomic64.shaderBufferInt64Atomics);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &queueFamilyCount, queueFamilies.data());
    uint32_t computeFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i)
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeFamily = i; break; }
    if (computeFamily == UINT32_MAX) { std::printf("FAIL no compute queue\n"); return 1; }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCI{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCI.queueFamilyIndex = computeFamily;
    queueCI.queueCount = 1;
    queueCI.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCI{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCI.queueCreateInfoCount = 1;
    deviceCI.pQueueCreateInfos = &queueCI;
    VkDevice device = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(physical, &deviceCI, nullptr, &device));

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeFamily, 0, &queue);

    // Two 64-bit slots, host visible so the result can just be read back.
    constexpr VkDeviceSize kBufferSize = 2 * sizeof(uint64_t);
    VkBufferCreateInfo bufferCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCI.size = kBufferSize;
    bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    CHECK(vkCreateBuffer(device, &bufferCI, nullptr, &buffer));

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(device, buffer, &memReq);
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps);
    uint32_t memType = UINT32_MAX;
    const VkMemoryPropertyFlags wanted =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((memReq.memoryTypeBits & (1u << i)) == 0) continue;
        if ((memProps.memoryTypes[i].propertyFlags & wanted) != wanted) continue;
        memType = i;
        break;
    }
    if (memType == UINT32_MAX) { std::printf("FAIL no host visible memory\n"); return 1; }

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memory));
    CHECK(vkBindBufferMemory(device, buffer, memory, 0));

    void* mapped = nullptr;
    CHECK(vkMapMemory(device, memory, 0, kBufferSize, 0, &mapped));
    std::memset(mapped, 0, kBufferSize);

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo setLayoutCI{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutCI.bindingCount = 1;
    setLayoutCI.pBindings = &binding;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorSetLayout(device, &setLayoutCI, nullptr, &setLayout));

    VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts = &setLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout));

    // The whole point: pCode is not SPIR-V. It is the magic number followed by
    // null-terminated MSL source, and codeSize covers both.
    char mslSource[4096];
    std::snprintf(mslSource, sizeof(mslSource), kMSL, mslBufferIndex);
    const size_t mslLength = std::strlen(mslSource) + 1;
    std::vector<uint8_t> moduleBlob(sizeof(MVKMSLSPIRVHeader) + mslLength);
    const MVKMSLSPIRVHeader magic = kMVKMagicNumberMSLSourceCode;
    std::memcpy(moduleBlob.data(), &magic, sizeof(magic));
    std::memcpy(moduleBlob.data() + sizeof(magic), mslSource, mslLength);

    if (useSpirv)
    {
        std::FILE* spv = std::fopen("ref.spv", "rb");
        if (spv == nullptr) { std::printf("FAIL cannot open ref.spv\n"); return 1; }
        std::fseek(spv, 0, SEEK_END);
        const long spvSize = std::ftell(spv);
        std::fseek(spv, 0, SEEK_SET);
        moduleBlob.assign(static_cast<size_t>(spvSize), 0);
        std::fread(moduleBlob.data(), 1, static_cast<size_t>(spvSize), spv);
        std::fclose(spv);
    }

    VkShaderModuleCreateInfo moduleCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleCI.codeSize = moduleBlob.size();
    moduleCI.pCode = reinterpret_cast<const uint32_t*>(moduleBlob.data());
    VkShaderModule module = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(device, &moduleCI, nullptr, &module));
    std::printf("vkCreateShaderModule: ok\n");

    // MSL has no equivalent of SPIR-V's LocalSize execution mode, so the workgroup
    // size has to be handed over separately.
    if (!useSpirv)
    {
        const char* resolvedFrom = nullptr;
        const PFN_SetWorkgroupSizeMVK setWorkgroupSize = ResolveSetWorkgroupSize(&resolvedFrom);
        std::printf("vkSetWorkgroupSizeMVK resolved from: %s\n", resolvedFrom);
        if (setWorkgroupSize == nullptr)
        {
            std::printf("FAIL vkSetWorkgroupSizeMVK unavailable\n");
            return 1;
        }
        setWorkgroupSize(module, 64, 1, 1);
    }

    VkComputePipelineCreateInfo pipelineCI{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCI.stage.module = module;
    // For MSL source this is the Metal function name; for SPIR-V it is the entry point.
    pipelineCI.stage.pName = useSpirv ? "main" : "main0";
    pipelineCI.layout = pipelineLayout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline));
    std::printf("vkCreateComputePipelines with 64-bit atomic_max_explicit: ok\n");

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes = &poolSize;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorPool(device, &poolCI, nullptr, &pool));

    VkDescriptorSetAllocateInfo setAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setAlloc.descriptorPool = pool;
    setAlloc.descriptorSetCount = 1;
    setAlloc.pSetLayouts = &setLayout;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    CHECK(vkAllocateDescriptorSets(device, &setAlloc, &descriptorSet));

    VkDescriptorBufferInfo bufferInfo{buffer, 0, kBufferSize};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    VkCommandPoolCreateInfo cmdPoolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdPoolCI.queueFamilyIndex = computeFamily;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cmdAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAlloc.commandPool = cmdPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(device, &cmdAlloc, &cmd));

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                            0, 1, &descriptorSet, 0, nullptr);
    constexpr uint32_t kGroups = 16;   // 16 * 64 = 1024 threads
    vkCmdDispatch(cmd, kGroups, 1, 1);
    CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));

    const uint64_t* results = static_cast<const uint64_t*>(mapped);
    constexpr uint32_t kThreads = kGroups * 64;

    if (useSpirv)
    {
        // The reference shader writes 32-bit maxima into words 0 and 2, purely to
        // prove the descriptor reaches the shader at all under SPIR-V.
        const uint32_t* words = static_cast<const uint32_t*>(mapped);
        uint32_t expectedKey = 0;
        for (uint32_t tid = 0; tid < kThreads; ++tid)
        {
            const uint32_t key = static_cast<uint32_t>(tid * 2654435761u) >> 8;
            if (key > expectedKey) expectedKey = key;
        }
        std::printf("spirv word0 got %u expected %u %s\n", words[0], kThreads - 1,
                    words[0] == kThreads - 1 ? "OK" : "MISMATCH");
        std::printf("spirv word2 got %u expected %u %s\n", words[2], expectedKey,
                    words[2] == expectedKey ? "OK" : "MISMATCH");
        return (words[0] == kThreads - 1 && words[2] == expectedKey) ? 0 : 1;
    }

    // Expected slot 0: the largest thread index.
    const uint64_t expected0 = kThreads - 1;

    // Expected slot 1: recompute the same packing on the CPU and take the max.
    uint64_t expected1 = 0;
    for (uint32_t tid = 0; tid < kThreads; ++tid)
    {
        const uint32_t key = static_cast<uint32_t>(tid * 2654435761u) >> 8;
        const uint64_t packed = (static_cast<uint64_t>(key) << 32) | (tid + 1000u);
        if (packed > expected1) expected1 = packed;
    }

    std::printf("slot0 got %llu expected %llu %s\n",
                (unsigned long long)results[0], (unsigned long long)expected0,
                results[0] == expected0 ? "OK" : "MISMATCH");
    std::printf("slot1 got 0x%016llx expected 0x%016llx %s\n",
                (unsigned long long)results[1], (unsigned long long)expected1,
                results[1] == expected1 ? "OK" : "MISMATCH");
    const bool pass = results[0] == expected0 && results[1] == expected1;
    std::printf("%s\n", pass ? "PASS: real 64-bit atomic max on MoltenVK" : "FAIL");
    return pass ? 0 : 1;
}
