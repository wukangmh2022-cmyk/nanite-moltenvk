// Compiles an MSL file with the Metal runtime compiler and prints the diagnostics.
//
// MoltenVK reports a failed shader compile as "Compute shader function could not be
// compiled into pipeline. See previous logged error." and the previous logged error
// does not always reach stderr, which leaves nothing to act on. The Metal toolchain
// (xcrun metal) is a separate Xcode download, but Metal.framework can compile from
// source at runtime, and its NSError carries the full diagnostics.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//       msl_compile.mm -o msl_compile
// Run:
//   ./msl_compile shader.metal [function_name]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: %s <file.metal> [function]\n", argv[0]);
        return 2;
    }

    @autoreleasepool
    {
        NSString* path = [NSString stringWithUTF8String:argv[1]];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
        if (source == nil)
        {
            std::printf("cannot read %s: %s\n", argv[1],
                        error.localizedDescription.UTF8String);
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil)
        {
            std::printf("no Metal device\n");
            return 1;
        }
        std::printf("device: %s\n", device.name.UTF8String);

        MTLCompileOptions* options = [MTLCompileOptions new];
        // MoltenVK 1.4.2 on this machine reports Metal Shading Language 3.2.
        options.languageVersion = MTLLanguageVersion3_2;

        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                     options:options
                                                       error:&error];
        if (library == nil)
        {
            std::printf("COMPILE FAILED:\n%s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        std::printf("compile ok; functions: %s\n",
                    library.functionNames.description.UTF8String);

        if (argc >= 3)
        {
            NSString* name = [NSString stringWithUTF8String:argv[2]];
            id<MTLFunction> fn = [library newFunctionWithName:name];
            if (fn == nil)
            {
                std::printf("FAIL: no function named '%s'\n", argv[2]);
                return 1;
            }
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn
                                                                                   error:&error];
            if (pso == nil)
            {
                std::printf("PIPELINE FAILED:\n%s\n", error.localizedDescription.UTF8String);
                return 1;
            }
            std::printf("compute pipeline ok; maxThreadsPerThreadgroup=%lu\n",
                        (unsigned long)pso.maxTotalThreadsPerThreadgroup);
        }
        return 0;
    }
}
