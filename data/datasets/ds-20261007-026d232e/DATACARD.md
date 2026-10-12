# Data card — ds-20261007-026d232e

- dataset_sha256: `026d232ea2ee6694fc1b601ca4987dde3d7c8f9e2cf48c06ed0b455542387a11`
- prompt_sha256: `e5e67f5e6d0eb7d38e6981567ec2f8bc2623961181d7d29c04a52e2fa6825de6` (prompts/classify_system.v1.txt + contracts/categories.v1.json)
- git: `HEAD+dirty` · extractor 1.0.0 · created 2026-10-07T14:59:57.335483
- Real data only from drawings; augmented rows are train-only and counted separately. No synthetic files.

## Files

Real DXF files extracted: 40 · tiers: {'NEAR_CONFORMING': 20, 'UNLABELED': 9, 'CONFORMING': 11} · usable for labels (plot rule passed, conforming/near): 14

## Per class (real examples, no_colour variant; augmented in brackets)

| class | train | val | test | total real | learnable (≥30 real) |
|---|---:|---:|---:|---:|---|
| unknown | 1213 [1213] | 815 | 609 | 2637 | yes |
| BLK_n_FLR_n_BLT_UP_AREA_EXISTING | 76 [76] | 10 | 1 | 87 | yes |
| BLK_n_FLR_n_STAIR_n_FLIGHT_n | 10 [10] | 38 | 39 | 87 | yes |
| BLK_n_SETBACK_DIM | 43 [43] | 17 | 13 | 73 | yes |
| BLK_n_LVL_n_BLDG_FOOT_PRINT | 60 [60] | 7 | 3 | 70 | yes |
| BLK_n_COVERED_AREA | 57 [57] | 7 | 5 | 69 | yes |
| BLK_n_FLR_n_ROOF_AREA | 35 [35] | 0 | 0 | 35 | yes |
| BLK_n_FLR_n_BLT_UP_AREA | 9 [9] | 16 | 7 | 32 | yes |
| BLK_n_FLR_n_FLOOR_HEIGHT | 17 [17] | 8 | 4 | 29 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_FRONT_SETBACK | 13 [13] | 4 | 3 | 20 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_REAR_SETBACK | 13 [13] | 4 | 3 | 20 | **NOT LEARNABLE YET** |
| OPEN_PARKING | 7 [7] | 8 | 5 | 20 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_STAIR_n_LANDING_n | 1 [1] | 18 | 0 | 19 | **NOT LEARNABLE YET** |
| BLK_n_HT_OF_BLDG | 10 [10] | 5 | 4 | 19 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_SIDE_SETBACK1 | 12 [12] | 4 | 3 | 19 | **NOT LEARNABLE YET** |
| BLK_n_LVL_n_SIDE_SETBACK2 | 12 [12] | 4 | 3 | 19 | **NOT LEARNABLE YET** |
| GUARD_ROOM | 18 [18] | 1 | 0 | 19 | **NOT LEARNABLE YET** |
| NORTH_DIRECTION | 12 [12] | 3 | 4 | 19 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_BLT_UP_AREA_DEDUCT | 3 [3] | 15 | 0 | 18 | **NOT LEARNABLE YET** |
| MAIN_GATE | 11 [11] | 2 | 2 | 15 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_WC_BATH | 11 [11] | 3 | 0 | 14 | **NOT LEARNABLE YET** |
| PLAN_INFO | 7 [7] | 3 | 4 | 14 | **NOT LEARNABLE YET** |
| PLOT_BOUNDARY | 7 [7] | 3 | 4 | 14 | **NOT LEARNABLE YET** |
| DIST_BETWEEN_BLK_n_BLK_n | 7 [7] | 6 | 0 | 13 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_STAIR_n | 7 [7] | 3 | 0 | 10 | **NOT LEARNABLE YET** |
| BLK_n_PLINTH_HEIGHT | 4 [4] | 1 | 2 | 7 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_WATER_CLOSET | 6 [6] | 0 | 0 | 6 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_BLT_UP_AREA_DEDUCT_EXISTING | 4 [4] | 0 | 0 | 4 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_ROOM_n_MEZ_AREA_n | 3 [3] | 0 | 0 | 3 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_DEM_AREA | 2 [2] | 0 | 0 | 2 | **NOT LEARNABLE YET** |
| BLK_n_FLR_n_LIFT_n | 2 [2] | 0 | 0 | 2 | **NOT LEARNABLE YET** |
| BLK_n_STAIR_HEADROOM | 1 [1] | 0 | 0 | 1 | **NOT LEARNABLE YET** |

Classes with ZERO real examples (23): AMENITY_n, BLK_n_COVERED_AREA_DEDUCT, BLK_n_FLR_n_BASEMENT_HT_ABOVE_GL, BLK_n_FLR_n_CHAJJA, BLK_n_FLR_n_COVERED_PARKING, BLK_n_FLR_n_ROOM_n_LIGHT_VENTILATION_n, BLK_n_FLR_n_SERVICE_FLOOR, BLK_n_FLR_n_STILT_FLOOR, BLK_n_FLR_n_VEHICLERAMP_n, BLK_n_LVL_n_TEMP_STRUCTURE_n, BLK_n_MUMTY_HT, BLK_n_OTHER_ROOF_FEATURE_HT_n, BLK_n_ROOF_SERV_EQUIP_HT_n, BLK_n_ROOF_WATER_TANK_HT, BLK_n_TEMP_STRUCTURE_HT, COMPOUND_WALL, DIST_EXIT, HILL_CUTTING_AREA, PLANTATION_TREE_COUNT, RWH, SOLAR_PANEL, SPECIAL_PARKING, STACK_PARKING

## Per project (real objects, no_colour)

| project | split | files | objects | labelled (non-unknown) |
|---|---|---:|---:|---:|
| P-389c9b1fdd | test | 1 | 6 | 5 |
| P-c866f377f1 | test | 2 | 642 | 40 |
| P-ce24a2f002 | test | 1 | 70 | 64 |
| P-0744c38d12 | train | 1 | 74 | 30 |
| P-3cf5591a59 | train | 1 | 716 | 179 |
| P-73bec27336 | train | 1 | 104 | 26 |
| P-75e7b3e5bc | train | 1 | 384 | 96 |
| P-78592d8181 | train | 1 | 188 | 47 |
| P-9c988ff6a8 | train | 1 | 63 | 61 |
| P-a0fc981ea6 | train | 1 | 164 | 41 |
| P-0599b89d73 | val | 1 | 328 | 82 |
| P-7949c7daba | val | 1 | 118 | 32 |
| P-975c4d649c | val | 1 | 559 | 76 |

## Dropped objects (not labelled unknown — dropped)

- duplicate_of_labelled: 7
- near_miss: 78
- other_layer_in_near_conforming: 14371
- train_unknown_capped: 4647
- type_mismatch: 70

## Rows per file in this dataset

- test.no_colour.jsonl: 718 rows {'real': 718}
- test.with_colour.jsonl: 718 rows {'real': 718}
- train.no_colour.jsonl: 3386 rows {'real': 1693, 'augmented': 1693}
- train.with_colour.jsonl: 3386 rows {'real': 1693, 'augmented': 1693}
- val.no_colour.jsonl: 1005 rows {'real': 1005}
- val.with_colour.jsonl: 1005 rows {'real': 1005}
