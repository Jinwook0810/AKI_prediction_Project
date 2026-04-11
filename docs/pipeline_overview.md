# Pipeline Overview

```mermaid
flowchart TD
    A["Raw ICU Event Data<br/>released_df.csv(.gz)"] --> B["Urine Duplicate Cleaning<br/>keep smallest positive value per exact timestamp conflict"]
    B --> C["Hourly Wide Aggregation<br/>Urine=sum, Creatinine=min, others=last"]
    C --> D["Broad Value Cleaning<br/>set impossible values to NaN"]
    D --> E["Forward Fill Policy<br/>ffill most variables<br/>do not ffill Urine or Creatinine"]
    E --> F["AKI Label Generation"]
    F --> G["Early AKI Exclusion<br/>remove stays with AKI in hours 0-23"]
    G --> H["Cleaned Filtered Cohort"]
    H --> I["EDA"]
    H --> J["Tabular Features<br/>v1 / v2 / compact"]
    H --> K["Sequence Inputs<br/>24 hourly steps"]
    H --> L["Observed Mask<br/>pre-ffill observation indicators"]
    J --> M["Tabular Models<br/>LogReg / RF / CatBoost"]
    K --> N["Value-only LSTM"]
    K --> O["Masked Sequence Models"]
    L --> O
    O --> P["Masked LSTM / Transformer"]
```

## Current Canonical Workflow

1. run preprocessing
2. run EDA
3. build tabular or sequence inputs from the same cleaned cohort
4. train baselines using the provided split
5. select threshold on validation data
6. report final test metrics
