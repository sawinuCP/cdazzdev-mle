# Supply-Chain Anomaly Analyst — Use Case

## Problem statement

Operations teams receive thousands of machine-generated event feeds (one per
warehouse/SKU-day). Human analysts read each feed and write a structured
assessment: what happened, how bad it is, what evidence supports that, and what
to do about it. This project fine-tunes a small open-weight language model to
perform that first-pass analysis automatically and reliably.

- **Input (what the model receives):** one JSON event feed object with the
  fields `product_category`, `region`, `order_volume_vs_forecast_pct`,
  `inventory_days_on_hand`, `on_time_delivery_pct`, `supplier_status`,
  `transit_days_normal`, `transit_days_observed`, `port_congestion_index`,
  `weather_event`, `labor_document_note`, `unit_cost_vs_last_quarter_pct`,
  `notes` (free text).
- **Output (what the model must produce):** one valid JSON object —
  `{is_anomaly: bool, anomaly_class, severity: 1-5, root_cause: 2-4 sentences,
  evidence_fields: [input field names], corrective_actions: 3-5 strings,
  confidence: 0-1}`.
- **Correct vs incorrect:** correct = schema-valid, class supported by the
  cited evidence fields, severity consistent with the magnitudes in the feed,
  actions actionable and tied to the cause. Partially correct = schema-valid but
  one element off (a plausible-but-secondary class, severity off by 2+, weak
  causality). Hallucinated = invalid JSON, contradicts the input values, cites
  an event or field absent from the input, or a class outside the taxonomy.

## Taxonomy (`anomaly_class`)

| Class | Typical feed signature |
|---|---|
| `demand_spike` | order volume far above forecast, inventory drawn down |
| `demand_collapse` | order volume far below forecast, inventory piling up |
| `supplier_shipment_delay` | supplier status delayed, observed transit > normal |
| `port_logistics_congestion` | high port congestion index, transit above normal |
| `weather_disruption` | specific weather event in the feed plus transit/OTD impact |
| `quality_hold` | supplier/quality hold note, affected on-time delivery |
| `labor_strike` | labour note documents a stoppage plus throughput impact |
| `freight_inflation` | unit cost jump without a matching operational disruption |
| `inventory_obsolescence` | aging inventory with soft demand |
| `warehouse_capacity_breach` | inventory days far above storage norms |
| `none` | benign fluctuations only (control cases), `is_anomaly=false`, severity 1 |

## Severity guide

1 trivial · 2 minor · 3 material · 4 severe · 5 critical. Scenario tuples carry
an allowed band: mild → {1,2}, severe → {3,4}, critical → {4,5}.

## Why this use case is a strong choice

- Domain-specific and operationally meaningful — not a generic chat task.
- Every output is objectively checkable: schema, taxonomy membership, evidence
  subset, severity band, and keyword-level grounding rules are all mechanical.
- Gold labels are derivable from the generator, so base-vs-fine-tuned
  comparisons are like-for-like on identical prompts and decoding.
