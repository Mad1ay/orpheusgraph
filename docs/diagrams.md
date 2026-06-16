# orpheusgraph — Architecture Diagrams

## 1. High-Level Overview (Low Detail)

```mermaid
graph TB
    subgraph External["External Systems"]
        DB[("PostgreSQL")]
        Redis[("Redis L2")]
        PGV[("pgvector")]
    end

    subgraph OrpheusGraph["orpheusgraph Rust Crate"]
        Builder["builder.rs"]
        Graph["graph.rs - Immutable DiGraph"]
        Traversal["traversal.rs"]
        Scoring["scoring.rs"]
        Overlay["overlay.rs"]
        Serialization["serialization.rs - rkyv"]
        Edges["outgoing/incoming_edges"]
    end

    subgraph Python["Python Layer (host application)"]
        Cache["graph_cache.py - 3-Tier"]
        Tools["agent_tools.py"]
        Context["DynamicContext"]
        Format["format_for_llm - Markdown"]
    end

    subgraph LLM["LLM Pipeline"]
        Agent["RAG agent"]
        Prompt["Markdown context 2-3K tokens"]
    end

    DB -->|"SQL rows"| Builder
    Builder -->|"petgraph"| Graph
    Graph -->|"rkyv bytes"| Serialization
    Serialization -->|"lz4 + rkyv"| Redis
    Redis -->|"lz4 + rkyv"| Serialization

    Cache -->|"L1 hit"| Graph
    Cache -->|"L2 miss"| Redis
    Cache -->|"L3 miss"| DB

    PGV -->|"cosine sim"| Context
    Context -->|"ephemeral"| Overlay
    Context -->|"w coefficients"| Scoring

    Tools -->|"get_graph"| Cache
    Tools -->|"DynamicContext"| Traversal
    Traversal -->|"Top-K NodeResult"| Format
    Edges -->|"in/out EdgeResult"| Format
    Scoring -.->|"lazy eval"| Traversal
    Overlay -.->|"chain iterator"| Traversal

    Format -->|"Markdown with arrows"| Prompt
    Prompt --> Agent

    style OrpheusGraph fill:#8b5cf6,color:#fff
    style Python fill:#3b82f6,color:#fff
    style External fill:#22c55e,color:#fff
    style LLM fill:#f97316,color:#fff
```

---

## 2. 3-Tier Cache Flow (Detailed)

```mermaid
flowchart TD
    REQ["get_graph request"] --> GEN{"Check generation"}

    GEN --> L1{"L1: In-process Rust memory"}
    L1 -->|"HIT + gen match"| DONE["Return graph"]
    L1 -->|"MISS or stale gen"| L2

    L2{"L2: Redis lz4+rkyv"} -->|"HIT ~5ms"| DECOMP["lz4.decompress + from_rkyv"]
    DECOMP --> WARM_L1["Warm L1 cache"] --> DONE

    L2 -->|"MISS"| LOCK{"redis.set lock nx=True"}

    LOCK -->|"GOT LOCK"| BUILD["L3: load_from_db + build_graph"]
    BUILD --> COMPRESS["lz4.compress to_rkyv"]
    COMPRESS --> STORE["redis.set key data"]
    STORE --> SIGNAL["redis.lpush signal ready"]
    SIGNAL --> RELEASE["redis.delete lock"]
    RELEASE --> WARM2["Warm L1"] --> DONE

    LOCK -->|"LOCK EXISTS"| WAIT["redis.blpop signal timeout=30"]
    WAIT -->|"SIGNAL"| L2
    WAIT -->|"TIMEOUT"| FORCE["Force rebuild delete lock"] --> BUILD

    BUILD -.->|"EXCEPTION"| ERROR["redis.set build_error TTL=60s"]
    BUILD -.->|"SIGKILL"| TTL["Lock TTL 30s expires"]

    style DONE fill:#22c55e,color:#fff
    style ERROR fill:#dc2626,color:#fff
    style TTL fill:#f97316,color:#fff
```

---

## 3. Beam Search Traversal Pipeline (Detailed)

```mermaid
flowchart LR
    START["Start node: sale.order"] --> LEVEL1

    subgraph LEVEL1["Level 1 depth=1"]
        N1["Get neighbors base+overlay"]
        F1["Filter max_fan_out + pagerank"]
        S1["Score: W = base+sem+ovr x 1-noise"]
        T1["Sort Top-K=5"]
    end

    N1 --> F1 --> S1 --> T1

    T1 --> LEVEL2

    subgraph LEVEL2["Level 2 depth=2"]
        N2["Neighbors of Top-5"]
        F2["Filter"]
        S2["Score lazy"]
        T2["Top-K=5"]
    end

    N2 --> F2 --> S2 --> T2

    T2 --> LEVEL3

    subgraph LEVEL3["Level 3 depth=3"]
        N3["Neighbors of Top-5"]
        F3["Filter"]
        S3["Score lazy"]
        T3["Top-K=5"]
    end

    N3 --> F3 --> S3 --> T3

    T3 --> RESULT["Result: 15 nodes 2-3K tokens"]

    style RESULT fill:#22c55e,color:#fff
```

---

## 4. DynamicContext Composition (Detailed)

```mermaid
flowchart TB
    subgraph Sources["Data Sources Python"]
        VEC["pgvector search - semantic_boosts top-200"]
        DUMP["Structure Dump - overlay_nodes, overlay_edges"]
        STATS["Project usage stats - weight_overrides"]
        CONFIG["Pipeline config - w_base, w_semantic, noise_tags, max_fan_out"]
    end

    subgraph CTX["DynamicContext Rust pyclass"]
        SB["semantic_boosts: HashMap"]
        ON["overlay_nodes: Vec"]
        OE["overlay_edges: Vec"]
        WO["weight_overrides: HashMap"]
        WC["w_base / w_semantic / w_noise / w_override"]
        NT["noise_tags: HashSet"]
        MF["max_fan_out: Option usize"]
    end

    VEC --> SB
    DUMP --> ON
    DUMP --> OE
    STATS --> WO
    CONFIG --> WC
    CONFIG --> NT
    CONFIG --> MF

    CTX -->|"per-request ephemeral"| RUST["Rust: beam_traverse"]

    RUST --> RESULT["NodeResult + explain_score"]

    style CTX fill:#8b5cf6,color:#fff
    style Sources fill:#3b82f6,color:#fff
```

---

## 5. Graph Build Pipeline (Detailed)

```mermaid
flowchart TD
    subgraph Parse["KB Parse Python"]
        ENT["ERPKnowledgeEntity - nodes kind=model"]
        FLD["ERPKnowledgeField - nodes kind=field, edges kind=contains"]
        MOD["ERPKnowledgeModule - nodes kind=module, edges kind=depends_on"]
        DOC["ERPDocChunk - nodes kind=doc, edges kind=describes"]
        INH["Inherit JSONB - edges kind=inherits"]
        REL["field.relation - edges kind=relates_to"]
    end

    subgraph Normalize["Normalize Python"]
        NRM["base_weight to 0-1, noise_penalty to 0-1, pagerank compute"]
    end

    subgraph Build["Build Rust"]
        BG["orpheusgraph.build_graph nodes edges"]
        PG["petgraph DiGraph Immutable"]
        PR["Compute pagerank_weight"]
    end

    subgraph CachePipe["Cache Pipeline"]
        SER["graph.to_rkyv"]
        LZ4["lz4.compress"]
        RED["Redis SET key=v1:x86_64:odoo:18.0"]
    end

    ENT --> NRM
    FLD --> NRM
    MOD --> NRM
    DOC --> NRM
    INH --> NRM
    REL --> NRM

    NRM -->|"nodes and edges"| BG
    BG --> PG
    PG --> PR
    PR --> SER
    SER --> LZ4
    LZ4 --> RED

    style Build fill:#8b5cf6,color:#fff
    style CachePipe fill:#22c55e,color:#fff
```

---

## 6. Invalidation & Generation Flow

```mermaid
sequenceDiagram
    participant Parser as KB Parser
    participant Redis as Redis
    participant WB as Worker B
    participant WC as Worker C

    Parser->>Redis: invalidate odoo 18.0
    Redis->>Redis: DELETE graph key
    Redis->>Redis: INCR graph_gen gen=5

    Note over WB,WC: Workers check gen on next request

    WB->>Redis: GET graph_gen = 5
    WB->>WB: L1 gen=4 stale
    WB->>WB: old.close free Rust memory
    WB->>Redis: GET graph key
    Redis-->>WB: MISS

    WB->>Redis: SET lock nx=True OK
    WB->>WB: load_from_db + build_graph

    WC->>Redis: GET graph_gen = 5 L1 stale
    WC->>Redis: SET lock nx=True FAIL
    WC->>Redis: BLPOP graph_signal waiting

    WB->>Redis: SET graph lz4+rkyv
    WB->>Redis: LPUSH graph_signal ready
    Redis-->>WC: BLPOP returns ready
    WC->>Redis: GET graph HIT
    WC->>WC: from_rkyv warm L1
```

---

## 7. Flyweight: Tenant Isolation

```mermaid
graph TB
    subgraph Shared["Shared Immutable"]
        BASE["Base Graph: 50K nodes, 180K edges, 15MB Rust memory"]
    end

    subgraph TenantA["Tenant A: Warehouse Project"]
        CTX_A["Context A: stock boosted, x_custom overlay, technical noise"]
        RES_A["Result A: stock.picking, stock.move, x_custom_field"]
    end

    subgraph TenantB["Tenant B: HR Module"]
        CTX_B["Context B: hr boosted, x_hr_skill overlay, technical+stock noise"]
        RES_B["Result B: hr.employee, hr.department, x_hr_skill"]
    end

    BASE -->|"+ Context A ephemeral"| RES_A
    BASE -->|"+ Context B ephemeral"| RES_B

    style Shared fill:#8b5cf6,color:#fff
    style TenantA fill:#3b82f6,color:#fff
    style TenantB fill:#f97316,color:#fff
```
