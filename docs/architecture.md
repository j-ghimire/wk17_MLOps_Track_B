# Part B Architecture

```mermaid
flowchart LR
    User --> UI[W15 Streamlit UI]
    UI --> API[W15 FastAPI service]
    API --> Agent[W16 bounded agent loop]
    Agent --> Retrieve[Curated support retrieval tool]
    Retrieve --> KB[(Support knowledge base)]
    Agent --> Verify{Evidence sufficient?}
    Verify -->|yes| Answer[Grounded answer]
    Verify -->|no| Retry[Refined retrieval or clarification]
    Retry --> Agent
    Provider[Optional OpenAI-compatible provider] -. bounded, evidence-grounded .-> Agent

    Eval[Configuration evaluation harness] --> Agent
    Eval --> MLflow[MLflow runs and artifacts]
    Eval --> Compare[Comparison and query CSVs]
    Compare --> Evidently[Evidently drift report]
    MLflow --> RegistryUI[MLflow tracking UI]
```

The serving path uses a deterministic local knowledge base by default. The optional provider is disabled unless explicitly configured with environment settings and a key. The evaluation harness runs the same labeled query suite against multiple bounded retrieval configurations, logs results to MLflow, exports comparison artifacts, and compares evaluation telemetry with Evidently.
