# Patches against the vendored DiligentCore

`../DiligentCore` is a working copy of DiligentCore that this demo links as a
prebuilt static library. It is outside this repository, so any change made in it
is invisible to git here and would be lost by a re-clone. Every such change is
exported into this directory as well, and this file is the record of why.

Base commit: `b402aefa3`.

## 0001-vulkan-msl-shader-module-override.patch

Adds an MSL shader-module override to the Vulkan backend, active only on Apple
platforms and only when `DILIGENT_MSL_OVERRIDE_DIR` is set. For each shader
stage it looks for `<shader name with non-alphanumerics replaced by _>.metal` in
that directory and, if found, hands MoltenVK that source instead of the shader's
SPIR-V, through MoltenVK's `kMVKMagicNumberMSLSourceCode` path.

The point is to reach Metal instructions SPIRV-Cross will not emit. Concretely:
64-bit `atomic_max_explicit`, which is what a visibility buffer needs and which
SPIRV-Cross rejects with "MSL currently does not support 64-bit atomics" - while
Vulkan reports `shaderBufferInt64Atomics = 0` on this device because it requires
the full add/exchange/CAS/min/max set and Metal has only min and max. The
hardware can do it; only the translation layer cannot.

Reflection still comes from the SPIR-V, so the override replaces module bytes
only and must keep the same interface. Two things it must respect:

- `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=0` is required. With argument buffers
  on, descriptor set 0 arrives as one Metal argument buffer at `[[buffer(0)]]`
  and individual bindings have no buffer index of their own, so the indices a
  hand-written kernel declares cannot match. The demo sets this value
  automatically when `NANITE_NATIVE_64BIT_VISIBILITY` is enabled; standalone
  users of the patch must set it themselves. Measured cost of turning it off in
  this app: none.
- The `[[buffer(n)]]`/`[[texture(n)]]` indices have to be the ones MoltenVK
  would have chosen. `MVK_CONFIG_SHADER_DUMP_DIR=<dir>` dumps its own generated
  `.metal` per shader, which is both the way to read those indices off and the
  practical way to author an override: start from the dump and edit it.

Two directives are read from comments at the top of the file:

    // MVK_WORKGROUP_SIZE 64 1 1
    // MVK_ENTRY_POINT main0

The workgroup size is mandatory for compute, because MSL has no equivalent of
SPIR-V's LocalSize execution mode; it is forwarded via `vkSetWorkgroupSizeMVK`.
That entry point is not exported through the Vulkan loader, so it is resolved
with `dlsym(RTLD_DEFAULT)` and then `dlopen` of libMoltenVK, which returns the
image the loader has already brought in. The entry point name defaults to
`main0` because MSL cannot have a function called `main`.

Users of this facility in the demo: `Shaders/Nanite/msl/`.

## Runtime queue-family workaround

The shader override above is a DiligentCore source patch because it changes how
shader modules are created. Queue-family specialization does not require a
second DiligentCore patch: MoltenVK exposes the setting as the documented
`MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES` runtime parameter. The demo does not
enable it automatically: queue-family specialization is a startup experiment
because it can hurt performance on a unified GPU. Set it to `1` before launch to
expose the compute-only family, and add `NANITE_ASYNC_RASTER=1` to use it by
default. The Control panel can toggle async submission after launch; changing
the family layout requires a restart. Set it to `0` to force MoltenVK's default
general-purpose layout.
