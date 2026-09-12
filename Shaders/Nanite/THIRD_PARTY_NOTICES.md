# Nanite Shader References

The Nanite shader work in this directory is an independent HLSL adaptation.
It uses the following MIT-licensed projects as technical references:

- SimNanite, Copyright (c) 2024 ShawnTSH1229.
  - Persistent work queues and indirect draw argument layouts.
  - Source: https://github.com/ShawnTSH1229/SimNanite
- Nyx, Copyright (c) 2026 moonlovelj.
  - DAG node/group layouts, screen-space error evaluation, conservative HZB
    visibility testing, and meshlet culling structure.
  - Source: https://github.com/moonlovelj/Nyx

The original repositories remain separate checkouts. Their licenses are
preserved here because the adapted algorithms are derived from their shader
implementations.

The offline mesh builder also uses meshoptimizer and its clusterlod companion:

- meshoptimizer, Copyright (c) 2016-2026 Arseny Kapoulkine.
  - Source: https://github.com/zeux/meshoptimizer
  - License: MIT
