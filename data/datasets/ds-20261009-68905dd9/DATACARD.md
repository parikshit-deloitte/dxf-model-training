# Data card — ds-20261009-68905dd9

- dataset_sha256: `68905dd9b75efce7655cf4424f3cf3ed40a880566b4490df4d9d044855a050f6`
- prompt_sha256: `e5e67f5e6d0eb7d38e6981567ec2f8bc2623961181d7d29c04a52e2fa6825de6` (prompts/classify_system.v1.txt + contracts/categories.v1.json)
- git: `HEAD+dirty` · extractor 1.1.0 · created 2026-10-09T08:14:19.447857
- Real data only from drawings; augmented rows are train-only and counted separately. No synthetic files.

## Files

Real DXF files extracted: 53 · tiers: {'UNLABELED': 19, 'NEAR_CONFORMING': 23, 'CONFORMING': 11} · usable for labels (plot rule passed, conforming/near): 20

## Per class (real examples, no_colour variant; augmented in brackets)

| class | train | val | test | total real | learnable (≥30 real) |
|---|---:|---:|---:|---:|---|
| unknown | 1311 [1311] | 4582 | 609 | 6502 | yes |
| BLK_n_FLR_n_STAIR_n_FLIGHT_n | 63 [63] | 24 | 39 | 126 | yes |
| BLK_n_FLR_n_BLT_UP_AREA_EXISTING | 19 [19] | 74 | 1 | 94 | yes |
| BLK_n_SETBACK_DIM | 58 [58] | 19 | 13 | 90 | yes |
| BLK_n_COVERED_AREA | 19 [19] | 54 | 5 | 78 | yes |
| BLK_n_LVL_n_BLDG_FOOT_PRINT | 19 [19] | 55 | 3 | 77 | yes |
| BLK_n_FLR_n_ROOF_AREA | 70 [70] | 0 | 0 | 70 | yes |
| BLK_n_FLR_n_STAIR_n_LANDING_n | 33 [33] | 18 | 0 | 51 | yes |
| BLK_n_FLR_n_BLT_UP_AREA | 24 [24] | 18 | 7 | 49 | yes |
| BLK_n_FLR_n_FLOOR_HEIGHT | 29 [29] | 9 | 4 | 42 | yes |
| GUARD_ROOM | 36 [36] | 1 | 0 | 37 | yes |
| BLK_n_HT_OF_BLDG | 20 [20] | 5 | 4 | 29 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_WATER_CLOSET | 28 [28] | 0 | 0 | 28 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_FRONT_SETBACK | 18 [18] | 7 | 3 | 28 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_REAR_SETBACK | 18 [18] | 7 | 3 | 28 | **NOT LEARNABLE YET** |
| OPEN_PARKING | 16 [16] | 7 | 5 | 28 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_SIDE_SETBACK1 | 18 [18] | 6 | 3 | 27 | **NOT LEARNABLE YET** |
| NORTH_DIRECTION | 18 [18] | 5 | 4 | 27 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_BLT_UP_AREA_DEDUCT | 9 [9] | 17 | 0 | 26 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_SIDE_SETBACK2 | 17 [17] | 6 | 3 | 26 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_STAIR_n | 22 [22] | 3 | 0 | 25 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_WC_BATH | 13 [13] | 8 | 0 | 21 | **NOT LEARNABLE YET** |
| PLAN_INFO | 12 [12] | 4 | 4 | 20 | **NOT LEARNABLE YET** |
| PLOT_BOUNDARY | 12 [12] | 4 | 4 | 20 | **NOT LEARNABLE YET** |
| DIST_BETWEEN_BLK_n_BLK_n | 13 [13] | 6 | 0 | 19 | **NOT LEARNABLE YET** |
| MAIN_GATE | 12 [12] | 4 | 2 | 18 | **NOT LEARNABLE YET** |
| BLK_n_PLINTH_HEIGHT | 9 [9] | 2 | 2 | 13 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_LIFT_n | 10 [10] | 0 | 0 | 10 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_BLT_UP_AREA_DEDUCT_EXISTING | 7 [7] | 0 | 0 | 7 | **NOT LEARNABLE YET** |
| RWH | 4 [4] | 0 | 0 | 4 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_DEM_AREA | 3 [3] | 0 | 0 | 3 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_ROOM_n_MEZ_AREA_n | 3 [3] | 0 | 0 | 3 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_CHAJJA | 2 [2] | 0 | 0 | 2 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_COVERED_PARKING | 2 [2] | 0 | 0 | 2 | **NOT LEARNABLE YET** |
| BLK_n_ROOF_WATER_TANK_HT | 2 [2] | 0 | 0 | 2 | **NOT LEARNABLE YET** |
| BLK_n_STAIR_HEADROOM | 2 [2] | 0 | 0 | 2 | **NOT LEARNABLE YET** |

Classes with ZERO real examples (19): AMENITY_n, BLK_n_COVERED_AREA_DEDUCT, BLK_n_FLR_n_BASEMENT_HT_ABOVE_GL, BLK_n_FLR_n_ROOM_n_LIGHT_VENTILATION_n, BLK_n_FLR_n_SERVICE_FLOOR, BLK_n_FLR_n_STILT_FLOOR, BLK_n_FLR_n_VEHICLERAMP_n, BLK_n_LVL_n_TEMP_STRUCTURE_n, BLK_n_MUMTY_HT, BLK_n_OTHER_ROOF_FEATURE_HT_n, BLK_n_ROOF_SERV_EQUIP_HT_n, BLK_n_TEMP_STRUCTURE_HT, COMPOUND_WALL, DIST_EXIT, HILL_CUTTING_AREA, PLANTATION_TREE_COUNT, SOLAR_PANEL, SPECIAL_PARKING, STACK_PARKING

## Per project (real objects, no_colour)

| project | split | files | objects | labelled (non-unknown) |
|---|---|---:|---:|---:|
| P-389c9b1fdd | test | 1 | 6 | 5 |
| P-c866f377f1 | test | 2 | 642 | 40 |
| P-ce24a2f002 | test | 1 | 70 | 64 |
| P-0744c38d12 | train | 1 | 74 | 30 |
| P-6073c2a6ba | train | 1 | 64 | 64 |
| P-75e7b3e5bc | train | 2 | 752 | 188 |
| P-78592d8181 | train | 1 | 188 | 47 |
| P-7949c7daba | train | 1 | 118 | 32 |
| P-81fa43b6a3 | train | 1 | 68 | 38 |
| P-9c988ff6a8 | train | 1 | 63 | 61 |
| P-a0fc981ea6 | train | 1 | 164 | 41 |
| P-b0767d4e87 | train | 1 | 240 | 60 |
| P-bf5f11283e | train | 1 | 188 | 47 |
| P-c160064eda | train | 1 | 52 | 52 |
| P-0599b89d73 | val | 1 | 328 | 82 |
| P-3cf5591a59 | val | 1 | 3711 | 179 |
| P-73bec27336 | val | 1 | 347 | 26 |
| P-975c4d649c | val | 1 | 559 | 76 |

## Dropped objects (not labelled unknown — dropped)

- _text_redacted_not_dropped: 2
- duplicate_of_labelled: 26
- near_miss: 191
- other_layer_in_near_conforming: 16713
- train_unknown_capped: 3499
- type_mismatch: 140

## Rows per file in this dataset

- test.no_colour.jsonl: 718 rows {'real': 718}
- test.with_colour.jsonl: 718 rows {'real': 718}
- train.no_colour.jsonl: 3942 rows {'real': 1971, 'augmented': 1971}
- train.with_colour.jsonl: 3942 rows {'real': 1971, 'augmented': 1971}
- val.no_colour.jsonl: 4945 rows {'real': 4945}
- val.with_colour.jsonl: 4945 rows {'real': 4945}
