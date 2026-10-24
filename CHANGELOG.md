# CHANGELOG

<!-- version list -->

## v6.0.0 (2026-10-24)

### Features

- Complete public API, forecasting preparation and scientific docs
  ([`96b5615`](https://github.com/El3ssar/ResDAG/commit/96b561584c34e8b57392ddaae5e75ba4a9fb04a0))

### Breaking Changes

- Generators default to normalize=None; request normalize="standard" explicitly to reproduce old
  generator scaling. Use advanced HPO helpers from rd.hpo rather than duplicate root aliases. Raw
  load_full requires trusted=True. Internal memory-context builders, cell validators and worker
  adapters are private. Migration guidance is included.


## v5.0.2 (2026-10-24)

### Bug Fixes

- Support minimum Torch pipeline capture and replay final evidence
  ([#467](https://github.com/El3ssar/ResDAG/pull/467),
  [`ca40fe4`](https://github.com/El3ssar/ResDAG/commit/ca40fe4b9806a97ec11813a791b6f1cb11c58d0f))

- Validate PyTorch pipelines and streamline parallel forecasting
  ([#467](https://github.com/El3ssar/ResDAG/pull/467),
  [`ca40fe4`](https://github.com/El3ssar/ResDAG/commit/ca40fe4b9806a97ec11813a791b6f1cb11c58d0f))

- Verify PyTorch pipelines and improve parallel forecasting
  ([#467](https://github.com/El3ssar/ResDAG/pull/467),
  [`ca40fe4`](https://github.com/El3ssar/ResDAG/commit/ca40fe4b9806a97ec11813a791b6f1cb11c58d0f))

### Documentation

- Record verified PyTorch closeout and experiment readiness
  ([#467](https://github.com/El3ssar/ResDAG/pull/467),
  [`ca40fe4`](https://github.com/El3ssar/ResDAG/commit/ca40fe4b9806a97ec11813a791b6f1cb11c58d0f))


## v5.0.1 (2026-10-23)

### Bug Fixes

- **ci**: Select evidence checks for every CPU result record
  ([#466](https://github.com/El3ssar/ResDAG/pull/466),
  [`3b34005`](https://github.com/El3ssar/ResDAG/commit/3b34005efd93a2f6573ecd96c9a5b23daecdf547))

### Performance Improvements

- **cpu**: Remove repeated sequence views and solve diagnostics
  ([#466](https://github.com/El3ssar/ResDAG/pull/466),
  [`3b34005`](https://github.com/El3ssar/ResDAG/commit/3b34005efd93a2f6573ecd96c9a5b23daecdf547))

- **cpu**: Reuse diagnostics and normalize sparse cycles analytically
  ([#466](https://github.com/El3ssar/ResDAG/pull/466),
  [`3b34005`](https://github.com/El3ssar/ResDAG/commit/3b34005efd93a2f6573ecd96c9a5b23daecdf547))

- **cpu**: Reuse readout diagnostics and normalize sparse cycles analytically
  ([#466](https://github.com/El3ssar/ResDAG/pull/466),
  [`3b34005`](https://github.com/El3ssar/ResDAG/commit/3b34005efd93a2f6573ecd96c9a5b23daecdf547))


## v5.0.0 (2026-10-23)

### Bug Fixes

- Isolate initialization roles and verify native benchmark limits
  ([#465](https://github.com/El3ssar/ResDAG/pull/465),
  [`18f78c8`](https://github.com/El3ssar/ResDAG/commit/18f78c83f37469fb941da0fd48cc223b7ea24b9c))

### Features

- Add compact reservoirs and verified forecast execution
  ([#465](https://github.com/El3ssar/ResDAG/pull/465),
  [`18f78c8`](https://github.com/El3ssar/ResDAG/commit/18f78c83f37469fb941da0fd48cc223b7ea24b9c))

- Add compact reservoirs, LSMR and verified forecast hooks
  ([#465](https://github.com/El3ssar/ResDAG/pull/465),
  [`18f78c8`](https://github.com/El3ssar/ResDAG/commit/18f78c83f37469fb941da0fd48cc223b7ea24b9c))

### Testing

- Preserve fixed reservoir quality anchors across seed changes
  ([#465](https://github.com/El3ssar/ResDAG/pull/465),
  [`18f78c8`](https://github.com/El3ssar/ResDAG/commit/18f78c83f37469fb941da0fd48cc223b7ea24b9c))


## v4.1.0 (2026-10-23)

### Bug Fixes

- **benchmarks**: Validate held-out forecasts and tuned RC accuracy
  ([#464](https://github.com/El3ssar/ResDAG/pull/464),
  [`041db61`](https://github.com/El3ssar/ResDAG/commit/041db61765573c936af0a5f69843d0da64b45ae1))

- **ci**: Run tuning guide for documentation changes
  ([#464](https://github.com/El3ssar/ResDAG/pull/464),
  [`041db61`](https://github.com/El3ssar/ResDAG/commit/041db61765573c936af0a5f69843d0da64b45ae1))

### Documentation

- Keep tuning parameter names readable ([#464](https://github.com/El3ssar/ResDAG/pull/464),
  [`041db61`](https://github.com/El3ssar/ResDAG/commit/041db61765573c936af0a5f69843d0da64b45ae1))

### Features

- **hpo**: Broaden built-in forecast tuning defaults
  ([#464](https://github.com/El3ssar/ResDAG/pull/464),
  [`041db61`](https://github.com/El3ssar/ResDAG/commit/041db61765573c936af0a5f69843d0da64b45ae1))

- **hpo**: Broaden preset tuning and validate held-out RC accuracy
  ([#464](https://github.com/El3ssar/ResDAG/pull/464),
  [`041db61`](https://github.com/El3ssar/ResDAG/commit/041db61765573c936af0a5f69843d0da64b45ae1))


## v4.0.0 (2026-10-23)

### Features

- **core**: Add complete artifacts and single-pass DAG training
  ([`0b546d6`](https://github.com/El3ssar/ResDAG/commit/0b546d6725ab9ce945ec0ec580500c6590a54ab7))

### Breaking Changes

- **core**: Batch fitting evaluates the complete warmup/training prefix in one call; temporal-layer
  behavior and stochastic draw order can change. Reused readout invocations and custom outer
  ESNModel.forward bodies fail clearly. Diagram exports infer suffixes and return paths without
  opening a viewer.


## v3.0.0 (2026-10-23)

### Features

- **hpo**: Complete streaming pruning and resource-aware tuning
  ([#462](https://github.com/El3ssar/ResDAG/pull/462),
  [`f8d7637`](https://github.com/El3ssar/ResDAG/commit/f8d7637efbe34aeedc788bf044c05a449dfeafde))

### Breaking Changes

- **hpo**: Raw HPO counts new attempts per call, reevaluates transferred settings by default, and
  keeps sampling, storage, pruning and callbacks in the parent process.


## v2.1.0 (2026-10-22)

### Features

- **hpo**: Add common model tuning and automatic readout solves
  ([#461](https://github.com/El3ssar/ResDAG/pull/461),
  [`ad0f8c9`](https://github.com/El3ssar/ResDAG/commit/ad0f8c90d30621fa85aabf030075e42e2ea7d4e7))


## v2.0.0 (2026-10-22)

### Features

- **core**: Unify PyTorch training and named forecasting
  ([#460](https://github.com/El3ssar/ResDAG/pull/460),
  [`4735542`](https://github.com/El3ssar/ResDAG/commit/47355420b689b7742b570a87d1a39bd6aa69f494))

### Breaking Changes

- **core**: ESN uses the tensor-based common workflow; custom readouts implement solve; raw graphs
  use .graph; Model and ESN checkpoint metadata uses version 2 and rejects older schemas. No legacy
  wrappers or conversion layer are provided.


## v1.2.0 (2026-10-22)

### Features

- **core**: Add named PyTorch model workflow ([#459](https://github.com/El3ssar/ResDAG/pull/459),
  [`d63905d`](https://github.com/El3ssar/ResDAG/commit/d63905d9a05f825465a5b1d00b45b5bf5f8954f0))


## v1.1.1 (2026-10-22)

### Bug Fixes

- Apply live cell settings and refresh sparse weight caches
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Complete phase-one correctness work and track roadmap
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Correct initializer mathematics and reservoir diagnostics
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Exclude failed tuning trials from successful results
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Fit recurrent DAG stages and preserve inference contracts
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Harden capacity scoring and qualify benchmark reports
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Honor sequential execution order in staged fitting
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Stabilize readout fitting and preserve solver precision
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

### Continuous Integration

- Refresh documentation after successful releases
  ([#457](https://github.com/El3ssar/ResDAG/pull/457),
  [`58c06e1`](https://github.com/El3ssar/ResDAG/commit/58c06e1d0cdab0831fed3c15c90c5468ef820433))

### Documentation

- Mark phase one verified and record CI evidence
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))

- Track delivery roadmap and strengthen phase-one verification
  ([#458](https://github.com/El3ssar/ResDAG/pull/458),
  [`41a1744`](https://github.com/El3ssar/ResDAG/commit/41a174491ce9f97d5e5ab8da277ba6a982800133))


## v1.1.0 (2026-10-21)

### Features

- Expand reservoir initialization, dynamics and calibration
  ([#456](https://github.com/El3ssar/ResDAG/pull/456),
  [`4024202`](https://github.com/El3ssar/ResDAG/commit/402420247b2ab5e2c58f18836e3f769242892251))


## v1.0.1 (2026-10-21)

### Bug Fixes

- **release**: Advance past reserved PyPI version 1.0.0
  ([#455](https://github.com/El3ssar/ResDAG/pull/455),
  [`d3cd0c3`](https://github.com/El3ssar/ResDAG/commit/d3cd0c3cc97ba0684a9efa0e99f8baa2cce89f17))


## v1.0.0 (2026-10-21)

### Features

- Unify forecast timing and composable reservoir pipelines
  ([#454](https://github.com/El3ssar/ResDAG/pull/454),
  [`ded8c22`](https://github.com/El3ssar/ResDAG/commit/ded8c22c04008af4197678130305c06495286558))

### Breaking Changes

- Low-level forecasts now start at the immediate unseen sample and return_warmup has W-1+H frames.
  Use alignment="legacy" in preparation and forecasting to reproduce older core/ensemble
  trajectories.


## v0.11.0 (2026-10-21)

### Bug Fixes

- Improve reservoir pipeline integration and forecast correctness
  ([#453](https://github.com/El3ssar/ResDAG/pull/453),
  [`0355ef0`](https://github.com/El3ssar/ResDAG/commit/0355ef0f5c62be9df26eff3bc42854a1dc5a2002))

### Features

- **training**: State-noise/LMNT regularizer + observables metrics (phase 15)
  ([#452](https://github.com/El3ssar/ResDAG/pull/452),
  [`072c1fc`](https://github.com/El3ssar/ResDAG/commit/072c1fc22ef136648fded9a9050836421fda69aa))


## v0.10.0 (2026-07-06)

### Features

- **scale**: Parallel local-state reservoirs + sparse CSR reservoir (phase 14)
  ([#451](https://github.com/El3ssar/ResDAG/pull/451),
  [`9717358`](https://github.com/El3ssar/ResDAG/commit/971735826ac204de485a200a8237b38163fee7ab))


## v0.9.0 (2026-07-05)

### Features

- **training**: Online learning — RLS readout + FORCE (phase 13)
  ([#450](https://github.com/El3ssar/ResDAG/pull/450),
  [`d24f773`](https://github.com/El3ssar/ResDAG/commit/d24f773bbbb98e07572479975f049c7524e2e01a))


## v0.8.0 (2026-07-05)

### Chores

- **release**: Let version bumps cross 1.0 (major_on_zero=true)
  ([`7ddcd8c`](https://github.com/El3ssar/ResDAG/commit/7ddcd8cb8a066192e746b66ec75dd3d64ce720dc))

### Features

- **api**: Coherent layer names + consolidated data namespace + duplicate-readout guard (phase 12)
  ([#449](https://github.com/El3ssar/ResDAG/pull/449),
  [`9abcaf8`](https://github.com/El3ssar/ResDAG/commit/9abcaf82b59e795d5cc12753ace53cb78afd73ec))

### Testing

- **benchmarks**: Full cross-library sweep on 0.7.0 + new-cell profile + reproducible traces
  ([`b76f576`](https://github.com/El3ssar/ResDAG/commit/b76f576de81a80a2d9a1c54f529ea4d0a08b5c2d))


## v0.7.0 (2026-07-05)

### Bug Fixes

- Coerce/validate device+dtype in coupled ensemble fit and forecast (#229)
  ([#424](https://github.com/El3ssar/ResDAG/pull/424),
  [`f111e3c`](https://github.com/El3ssar/ResDAG/commit/f111e3c07d53866daa95f474ab68a98330c7a405))

- Construct initializer RNG inside initialize() so seeded inits are pure functions of (seed, shape)
  ([#329](https://github.com/El3ssar/ResDAG/pull/329),
  [`18ac0b2`](https://github.com/El3ssar/ResDAG/commit/18ac0b26f7e358cfbe26969e8577fa81125a2ed9))

- Drop phantom chord_dendrocycle_graph from graphs.__all__
  ([#332](https://github.com/El3ssar/ResDAG/pull/332),
  [`a2a2fe6`](https://github.com/El3ssar/ResDAG/commit/a2a2fe633d6b3fa57508a464c80b35a93d90483c))

- Interpolated median (torch.quantile 0.5) for even-N coupled ensembles
  ([#408](https://github.com/El3ssar/ResDAG/pull/408),
  [`0ff1471`](https://github.com/El3ssar/ResDAG/commit/0ff1471f2a7cc2b7de25ac57d0f5517bb4902bff))

- Make esp_index cell-agnostic and route state through public API
  ([#330](https://github.com/El3ssar/ResDAG/pull/330),
  [`db9e994`](https://github.com/El3ssar/ResDAG/commit/db9e99457a57f4f63517fd22b3bf314a9d056741))

- Normalize Chebyshev feedback weights per-column (#190)
  ([#414](https://github.com/El3ssar/ResDAG/pull/414),
  [`497e669`](https://github.com/El3ssar/ResDAG/commit/497e669cfea9bf48128095045ec2ac8dc2b9959d))

- Preserve (timesteps, features) for 1-row CSV and document 1D loads
  ([#327](https://github.com/El3ssar/ResDAG/pull/327),
  [`cdf0b5a`](https://github.com/El3ssar/ResDAG/commit/cdf0b5a965d4eb8bd390955c060e58b1f0db6bce))

- Register reservoir state as a non-persistent buffer (#132)
  ([#323](https://github.com/El3ssar/ResDAG/pull/323),
  [`de50e9b`](https://github.com/El3ssar/ResDAG/commit/de50e9bfa44b4c65a354786733d7d928af49603b))

- Sample m distinct nodes in barabasi_albert generator (#138)
  ([#337](https://github.com/El3ssar/ResDAG/pull/337),
  [`301b4a1`](https://github.com/El3ssar/ResDAG/commit/301b4a188931e62a0bff51ba94964ee742a287db))

- Sign-preserving mode for Power + exponent guidance for power_augmented (#208)
  ([#410](https://github.com/El3ssar/ResDAG/pull/410),
  [`42c3744`](https://github.com/El3ssar/ResDAG/commit/42c37440b7a941468ec61311e49667d994017349))

- Tolerate partial reservoir states in load(load_states=True)
  ([#413](https://github.com/El3ssar/ResDAG/pull/413),
  [`eae3291`](https://github.com/El3ssar/ResDAG/commit/eae3291da800e1ddc4cb1a0b8f472eaa54de51bf))

- Validate FeaturePartitioner config at construction (#211)
  ([#420](https://github.com/El3ssar/ResDAG/pull/420),
  [`7d4ed26`](https://github.com/El3ssar/ResDAG/commit/7d4ed267ca2a7b0cc13965ae40dd8933975a394e))

- Validate NG-RC input_dim and reject driving inputs (#149)
  ([#406](https://github.com/El3ssar/ResDAG/pull/406),
  [`7d4aa92`](https://github.com/El3ssar/ResDAG/commit/7d4aa92bd8025d825c6f91bb0f50e66be5ccc1d6))

- Validate step counts in prepare_esn_data and warn on empty val split
  ([#409](https://github.com/El3ssar/ResDAG/pull/409),
  [`ba04a61`](https://github.com/El3ssar/ResDAG/commit/ba04a614a7683e0153c5138299ba6c49a8bd7e49))

- **ci**: Replace removed lychee --exclude-mail with --exclude=mailto:
  ([`cb67d75`](https://github.com/El3ssar/ResDAG/commit/cb67d75f2b705022e6787a3f377ce5fe5d9b960a))

- **core**: Validate and coerce states in set_reservoir_states
  ([#334](https://github.com/El3ssar/ResDAG/pull/334),
  [`5b4d7ce`](https://github.com/El3ssar/ResDAG/commit/5b4d7ce9ae0cd139afb3dc3c2a1849967c1b78d9))

- **ensemble**: Make CoupledEnsembleESNModel.forecast genuinely autoregressive
  ([#324](https://github.com/El3ssar/ResDAG/pull/324),
  [`4b1d9b2`](https://github.com/El3ssar/ResDAG/commit/4b1d9b2543fe425a6fe8775c700eded422b7b626))

- **ensemble**: Stop global-RNG clobber in coupled_ensemble_esn and guard sub-model output dim
  ([#351](https://github.com/El3ssar/ResDAG/pull/351),
  [`b4ae13c`](https://github.com/El3ssar/ResDAG/commit/b4ae13cff5e82c14b8eefdf396adb85a6e134864))

- **ensemble**: Use robust median/MAD z-score in OutliersFilteredMean
  ([#339](https://github.com/El3ssar/ResDAG/pull/339),
  [`33bfead`](https://github.com/El3ssar/ResDAG/commit/33bfeadb8f1db1a90617df2abe3695f42ecfbfe2))

- **ensemble**: Validate OutliersFilteredMean input rank
  ([#431](https://github.com/El3ssar/ResDAG/pull/431),
  [`0907af7`](https://github.com/El3ssar/ResDAG/commit/0907af7d0c3a596ac79b20f1a607ce03bf1f031e))

- **ensemble**: Wrap CoupledEnsembleESNModel.forecast/warmup in torch.no_grad()
  ([#378](https://github.com/El3ssar/ResDAG/pull/378),
  [`4fdd4d4`](https://github.com/El3ssar/ResDAG/commit/4fdd4d48b689a3d907ce6e8327929147a63a8c9d))

- **examples**: Forecast() driver alignment to horizon-step contract (#383)
  ([#418](https://github.com/El3ssar/ResDAG/pull/418),
  [`8c083fa`](https://github.com/El3ssar/ResDAG/commit/8c083fa3d51626b76802fd46574f2e3f8df33b9b))

- **hpo**: Cover multi-output forecast tuple guard in the objective (#274)
  ([#365](https://github.com/El3ssar/ResDAG/pull/365),
  [`72d2186`](https://github.com/El3ssar/ResDAG/commit/72d21862c3590f71a793caf9c0ccd409395024f5))

- **hpo**: Error on mismatched/missing drivers_keys instead of silently dropping
  ([#429](https://github.com/El3ssar/ResDAG/pull/429),
  [`dc2fafc`](https://github.com/El3ssar/ResDAG/commit/dc2fafcab0090b67ebf459c93bd338f3b8922f8c))

- **hpo**: Make make_study_name robust to partials, lambdas, callables (#272)
  ([#364](https://github.com/El3ssar/ResDAG/pull/364),
  [`8d71b99`](https://github.com/El3ssar/ResDAG/commit/8d71b99e12a74eb3550ce52adf218926149f1429))

- **hpo**: Penalize non-finite loss instead of failing the trial (#271)
  ([#354](https://github.com/El3ssar/ResDAG/pull/354),
  [`ac31ad1`](https://github.com/El3ssar/ResDAG/commit/ac31ad1858f5c98d0741b2ed7feb4cdc43e46d74))

- **hpo**: Portable multiprocess backend — spawn fallback, bounded budget, cooperative stop (#276)
  ([#371](https://github.com/El3ssar/ResDAG/pull/371),
  [`2d780a3`](https://github.com/El3ssar/ResDAG/commit/2d780a315abe7bbb7985e796e81c898155b330ea))

- **hpo**: Scale-safe loss defaults, zero-safe aggregation, no global side effects
  ([#367](https://github.com/El3ssar/ResDAG/pull/367),
  [`669c464`](https://github.com/El3ssar/ResDAG/commit/669c4643730fd2058b823c0f38eb5238b032f858))

- **hpo**: Thread per-trial seed into reservoir topology/initializer RNG (#273)
  ([#386](https://github.com/El3ssar/ResDAG/pull/386),
  [`1399afc`](https://github.com/El3ssar/ResDAG/commit/1399afcefe90c03a1172957716b2e9461cb88d54))

- **init**: Connected_* retry threads a shared RNG and raises on exhaustion
  ([#373](https://github.com/El3ssar/ResDAG/pull/373),
  [`63a32e8`](https://github.com/El3ssar/ResDAG/commit/63a32e87b2e93fdaba9185e9a9167d81f37c43e9))

- **init**: Honor per-call initialize(**kwargs) in class initializers
  ([#436](https://github.com/El3ssar/ResDAG/pull/436),
  [`44763b2`](https://github.com/El3ssar/ResDAG/commit/44763b2baffa0e47d0bb6813eba56572e9dd7ca0))

- **init**: Infer chain_of_neurons_input features, fix bare-name crash (#194)
  ([#425](https://github.com/El3ssar/ResDAG/pull/425),
  [`f837406`](https://github.com/El3ssar/ResDAG/commit/f8374065003d380c70bf91f88ce6ada8f9e9729c))

- **init**: Make graph topologies reproducible under torch.manual_seed + add seed kwarg
  ([#331](https://github.com/El3ssar/ResDAG/pull/331),
  [`2194c46`](https://github.com/El3ssar/ResDAG/commit/2194c466a4163c5c1e623cc4127c64b41a767489))

- **init**: Preserve float64 precision in structured initializers
  ([#341](https://github.com/El3ssar/ResDAG/pull/341),
  [`fb35b7f`](https://github.com/El3ssar/ResDAG/commit/fb35b7fb1a04f07676c5910a72d31fa9142dafd3))

- **init**: Spread OppositeAnchors over full ring to avoid duplicate columns
  ([#333](https://github.com/El3ssar/ResDAG/pull/333),
  [`616a897`](https://github.com/El3ssar/ResDAG/commit/616a89764d6167b00550f1c70e0351b30775ff0a))

- **init**: Stop double-rescaling baked-in spectral structure (#140)
  ([#352](https://github.com/El3ssar/ResDAG/pull/352),
  [`6c11902`](https://github.com/El3ssar/ResDAG/commit/6c11902062163384bdb5e27fb17fb6ecab0e604b))

- **models**: Correct power_augmented docs, sign-safe default, and Power fractional-exponent NaN
  ([#433](https://github.com/El3ssar/ResDAG/pull/433),
  [`06ebcc9`](https://github.com/El3ssar/ResDAG/commit/06ebcc9005035f1bd99c702a9d3ef50139e4ed1c))

- **models**: Forbid activation override in linear_esn (#233)
  ([#419](https://github.com/El3ssar/ResDAG/pull/419),
  [`c88e88c`](https://github.com/El3ssar/ResDAG/commit/c88e88c9ed4d98e6b7c9e21545e64a58e88febe9))

- **numerics**: Trustworthy defaults — spectral radius, ridge readout, seeds, MG
  ([#438](https://github.com/El3ssar/ResDAG/pull/438),
  [`72d0dd3`](https://github.com/El3ssar/ResDAG/commit/72d0dd37ce0023024578e6371459b8d26c803cfb))

- **readouts**: Guard CG ridge solver against per-column 0/0 NaNs
  ([#326](https://github.com/El3ssar/ResDAG/pull/326),
  [`890071a`](https://github.com/El3ssar/ResDAG/commit/890071a1633ef9b42dcfa651e0ad416590c5338c))

- **readouts**: Reject non-finite fit inputs with a clear ValueError
  ([#417](https://github.com/El3ssar/ResDAG/pull/417),
  [`56f9b51`](https://github.com/El3ssar/ResDAG/commit/56f9b5150b23f23b0da647ab233b10acdcc4182b))

- **reservoirs**: Detach restored state from autograd graph in set_state (#146)
  ([#392](https://github.com/El3ssar/ResDAG/pull/392),
  [`42db20f`](https://github.com/El3ssar/ResDAG/commit/42db20fe75c068b4e2f11f8cac7b56f6966df3e3))

- **reservoirs**: Drop duplicate degree-1 monomial block in NGCell (p=1)
  ([#328](https://github.com/El3ssar/ResDAG/pull/328),
  [`5a243f4`](https://github.com/El3ssar/ResDAG/commit/5a243f4dbad46c719b715441d2a6417ae01c0d7f))

- **reservoirs**: Error on set_state batch mismatch instead of silent zero-reinit
  ([#372](https://github.com/El3ssar/ResDAG/pull/372),
  [`f2d85f3`](https://github.com/El3ssar/ResDAG/commit/f2d85f33c85caea94c8e25d3f497b04f97879be2))

- **reservoirs**: Validate leak_rate in (0,1] and unify forward()/step() leak semantics
  ([#390](https://github.com/El3ssar/ResDAG/pull/390),
  [`4f5fd0b`](https://github.com/El3ssar/ResDAG/commit/4f5fd0b3dae5293024e5649f9110a63940fcd222))

- **state**: Exact forecast alignment, state lifecycle, and pipeline correctness
  ([`54ea230`](https://github.com/El3ssar/ResDAG/commit/54ea230294793a6d8920d910f6ff454ffbf837cd))

- **transforms**: Stop SelectiveExponentiation NaN gradients and NaN forward
  ([#322](https://github.com/El3ssar/ResDAG/pull/322),
  [`b456400`](https://github.com/El3ssar/ResDAG/commit/b4564007267c00fd6c49faa57e0bba79fd15437f))

- **typing**: Make the whole package mypy-clean and delete the baseline (#295)
  ([#370](https://github.com/El3ssar/ResDAG/pull/370),
  [`cf55a5d`](https://github.com/El3ssar/ResDAG/commit/cf55a5d6db3f8a627020aa772423dfc038940c0e))

### Chores

- Reconcile divergent dev-deps and unify dev-setup docs
  ([#423](https://github.com/El3ssar/ResDAG/pull/423),
  [`af8b0eb`](https://github.com/El3ssar/ResDAG/commit/af8b0eb3e8982f212148e09f68052a4570c64cf4))

- **ci**: Pin numpy<2.5 in the lint env to unbreak mypy
  ([#415](https://github.com/El3ssar/ResDAG/pull/415),
  [`919da02`](https://github.com/El3ssar/ResDAG/commit/919da024e91b722e0ecbdcca3259a7a511b3b69c))

- **ci**: Reusable _test.yml, py3.11-3.14 matrix, coverage floor, benchmark lane
  ([#369](https://github.com/El3ssar/ResDAG/pull/369),
  [`4b0b6af`](https://github.com/El3ssar/ResDAG/commit/4b0b6affda3d920ab4a53fdd4d5f3531f53ed6b3))

### Continuous Integration

- Freeze releases until 1.0 and adopt worktree-based parallel-dev protocol
  ([`dce0e4c`](https://github.com/El3ssar/ResDAG/commit/dce0e4c44e8c0ecc1d23456285b706df680abc73))

- Lift the release freeze and fix the release repo-name guard
  ([`b66cb70`](https://github.com/El3ssar/ResDAG/commit/b66cb709ee46d2d4aabf8901793cdcb418b06f68))

- Run only PR-affected tests via a static import-graph selector
  ([#362](https://github.com/El3ssar/ResDAG/pull/362),
  [`eec9670`](https://github.com/El3ssar/ResDAG/commit/eec9670aae5f6e3e7d349707446dce0baf85e1b3))

### Documentation

- Add LICENSE + CITATION.cff and fix GitHub Pages URL casing
  ([#116](https://github.com/El3ssar/ResDAG/pull/116),
  [`f0aaf36`](https://github.com/El3ssar/ResDAG/commit/f0aaf36e8be3ec2a105f9e85e49f8288a1c1f820))

- Align forecast.md Anatomy section with the #254 autoregressive contract
  ([#427](https://github.com/El3ssar/ResDAG/pull/427),
  [`86c068c`](https://github.com/El3ssar/ResDAG/commit/86c068ccf638f281b4051a291ef0a27361632569))

- Name competitors in related-work, fix first-forecast reservoir size (#309)
  ([#422](https://github.com/El3ssar/ResDAG/pull/422),
  [`a4e1232`](https://github.com/El3ssar/ResDAG/commit/a4e12328debf2c86da14b0d42e0624981e2b62bb))

- NG-RC tutorial + reservoirpy/RC.jl migration page + streaming/DataLoader workflow (#403)
  ([#426](https://github.com/El3ssar/ResDAG/pull/426),
  [`b8795dd`](https://github.com/El3ssar/ResDAG/commit/b8795dddfdc6017ddae7907432a8aa7b5a66cb8f))

- Sklearn + pure-PyTorch/SGD onboarding paths and the README pipeline story (#308)
  ([#421](https://github.com/El3ssar/ResDAG/pull/421),
  [`bf0a6cd`](https://github.com/El3ssar/ResDAG/commit/bf0a6cdf0c3accfee5937d479d762763e5ebd906))

- **init**: Correct resolve_* docstring format count (five vs six)
  ([#435](https://github.com/El3ssar/ResDAG/pull/435),
  [`79ddaed`](https://github.com/El3ssar/ResDAG/commit/79ddaedf089d9eab7d574a13c77f3e75264ce956))

- **reservoirs**: Fix base_reservoir state-shape docstring drift for NG-RC 3-D state
  ([#434](https://github.com/El3ssar/ResDAG/pull/434),
  [`eecdd7c`](https://github.com/El3ssar/ResDAG/commit/eecdd7c581394cde5da32265873bbd6062332d7b))

### Features

- Esp_index tail/window statistic and relative-distance normalization
  ([#407](https://github.com/El3ssar/ResDAG/pull/407),
  [`e08806c`](https://github.com/El3ssar/ResDAG/commit/e08806ce75911109e44fc1e2819cc85577fae7d0))

- **api**: Add resdag.ESN native facade for one-liner fit/forecast
  ([#325](https://github.com/El3ssar/ResDAG/pull/325),
  [`46b5bd9`](https://github.com/El3ssar/ResDAG/commit/46b5bd9d689a1033638f89eac5adc2ac1beb5b8f))

- **api**: Export esp_index from resdag.utils and top-level resdag
  ([#430](https://github.com/El3ssar/ResDAG/pull/430),
  [`2fd073c`](https://github.com/El3ssar/ResDAG/commit/2fd073cdf5de537c5d47416d1d9b66470b6814b6))

- **cells**: Continuous-time reservoirs — ContinuousReservoirCell + EuSN
  ([`3c82ef6`](https://github.com/El3ssar/ResDAG/commit/3c82ef632ed5fc1964f607de94d7dc0aa550af7f))

- **cells**: Non-ESN cell zoo — ES2N, ResESN, gated reservoir
  ([`2d9962a`](https://github.com/El3ssar/ResDAG/commit/2d9962a2f4bd21e4fcd26e17f752ca5f6714908e))

- **cells**: Spiking Liquid State Machine — SpikingReservoirCell + LIF
  ([`22e6fde`](https://github.com/El3ssar/ResDAG/commit/22e6fdef8efc9bd4c5069d6d881b8d215540259d))

- **core**: ESNModel.windowed_forecast — gap-filling reconstruction (#338)
  ([#384](https://github.com/El3ssar/ResDAG/pull/384),
  [`8712149`](https://github.com/El3ssar/ResDAG/commit/8712149c679e4cca2a10e1a79718fe5cf74a439e))

- **core**: Optionally-differentiable forecast/warmup/windowed_forecast (#264)
  ([#391](https://github.com/El3ssar/ResDAG/pull/391),
  [`bc77fba`](https://github.com/El3ssar/ResDAG/commit/bc77fbac1b88de4d9ed556c0823745e6acf70b52))

- **core**: ReservoirFeatureExtractor — nn.Sequential-friendly drop-in adapter (#266)
  ([#368](https://github.com/El3ssar/ResDAG/pull/368),
  [`12ebecc`](https://github.com/El3ssar/ResDAG/commit/12ebecc47f2ac7eae57b139ab471796e36136795))

- **data**: Canonical RC dataset generators (lorenz, mackey_glass, narma, ...)
  ([#346](https://github.com/El3ssar/ResDAG/pull/346),
  [`d7a397d`](https://github.com/El3ssar/ResDAG/commit/d7a397dc810fad0f827454b81791e672affdcdc0))

- **data**: Public denormalize_data + prepare_esn_data returns fitted stats
  ([#349](https://github.com/El3ssar/ResDAG/pull/349),
  [`9adb5ea`](https://github.com/El3ssar/ResDAG/commit/9adb5ea118bbf1ed6f6338eac29a0576ce229edd))

- **data**: TimeSeriesWindowDataset + make_dataloader for the SGD/streaming path
  ([#388](https://github.com/El3ssar/ResDAG/pull/388),
  [`80ee5b9`](https://github.com/El3ssar/ResDAG/commit/80ee5b913033cfb28274ac34e145a4dda47edb80))

- **hpo**: First-class Optuna pruners (ASHA/Hyperband/median) fed by incremental-horizon reporting
  ([#393](https://github.com/El3ssar/ResDAG/pull/393),
  [`430c5f3`](https://github.com/El3ssar/ResDAG/commit/430c5f3353514c67b59afecb15477ca8f28fc2a8))

- **hpo**: Warm-start + transfer (enqueue_trial / add_trial) and export_best_config
  ([#381](https://github.com/El3ssar/ResDAG/pull/381),
  [`2d74d89`](https://github.com/El3ssar/ResDAG/commit/2d74d894c844d86059f57f5841c30b8646c9bcc3))

- **init**: Device-native, torch.Generator-aware seed for random initializers (#188)
  ([#375](https://github.com/El3ssar/ResDAG/pull/375),
  [`b09b439`](https://github.com/El3ssar/ResDAG/commit/b09b4391ce7429c1c6889279be276c868fe4fab5))

- **init**: Normal/uniform/bernoulli input initializers with connectivity
  ([#382](https://github.com/El3ssar/ResDAG/pull/382),
  [`c6e15d2`](https://github.com/El3ssar/ResDAG/commit/c6e15d2ebe0b6456e2e29a6089c436e1da16caf3))

- **init**: Thread a torch/Generator seed through ESNLayer to fix the whole reservoir
  ([#336](https://github.com/El3ssar/ResDAG/pull/336),
  [`124f2f5`](https://github.com/El3ssar/ResDAG/commit/124f2f50561ef9f243bc2ef6e8d0e70c48a4581a))

- **metrics**: Resdag.metrics — forecast horizons in Lyapunov times
  ([`44b99b0`](https://github.com/El3ssar/ResDAG/commit/44b99b08a7d1ab71dbef95f8d5a1b9ead04ecc8b))

- **models**: Driving-input support in classic/ott/power_augmented factories
  ([#380](https://github.com/El3ssar/ResDAG/pull/380),
  [`2f045eb`](https://github.com/El3ssar/ResDAG/commit/2f045ebc1b6863313ebbab3e6843c0b5b09989e2))

- **models**: Per-neuron leak, DeepESN + HybridESN, FISTA readout
  ([`c244c82`](https://github.com/El3ssar/ResDAG/commit/c244c82cdc905535ac7856040abd6e8dba212fd9))

- **ngrc**: Document exact-degree-p convention + opt-in cumulative-degree mode
  ([#387](https://github.com/El3ssar/ResDAG/pull/387),
  [`aa35f94`](https://github.com/El3ssar/ResDAG/commit/aa35f9486fe407c62be4dbf39d9a9624da71bb45))

- **packaging**: Ship py.typed marker and gate mypy on the whole package
  ([#360](https://github.com/El3ssar/ResDAG/pull/360),
  [`115d1bc`](https://github.com/El3ssar/ResDAG/commit/115d1bcf2d6df8feeefb2536071a87b5a26c2f80))

- **parity**: RC.jl state-mod transforms + minimal/structured initializers
  ([`188f16a`](https://github.com/El3ssar/ResDAG/commit/188f16a9c798ab98fd09b9a6900101e7752c64d1))

- **readouts**: Add direct-solve readouts (Ridge/SVD/Pinv) + shared _solve_utils
  ([#335](https://github.com/El3ssar/ResDAG/pull/335),
  [`9570d4e`](https://github.com/El3ssar/ResDAG/commit/9570d4e16492ce786e568cf52cd52fddbadd79c9))

- **readouts**: IncrementalRidgeReadout + CholeskyReadoutLayer for the streaming / DataLoader path
  ([#374](https://github.com/El3ssar/ResDAG/pull/374),
  [`0402fa2`](https://github.com/El3ssar/ResDAG/commit/0402fa218c1e6df3e229da11ce4860693bb1fee2))

- **reservoirs**: Add optional train-mode state-noise injection to ESNCell
  ([#340](https://github.com/El3ssar/ResDAG/pull/340),
  [`34090a7`](https://github.com/El3ssar/ResDAG/commit/34090a76209d6de0b043ff4792f4fa68c9ec1b69))

- **transforms**: Add Standardize per-feature z-score transform (#215)
  ([#348](https://github.com/El3ssar/ResDAG/pull/348),
  [`4f55319`](https://github.com/El3ssar/ResDAG/commit/4f55319df293654b85c4d3029d8be662c31e7a68))

- **utils**: Torch-aware seed_everything + resolve_device helpers (#243)
  ([#355](https://github.com/El3ssar/ResDAG/pull/355),
  [`1b2441f`](https://github.com/El3ssar/ResDAG/commit/1b2441fe13b95e645757b7d17f1b9d93e59ec62e))

### Performance Improvements

- Vectorize dense graph generators in NumPy, skip NetworkX on hot path
  ([#416](https://github.com/El3ssar/ResDAG/pull/416),
  [`ba3c505`](https://github.com/El3ssar/ResDAG/commit/ba3c5054c6d2bd8a65adb103889fe123c36fda69))

- **core**: Flattened single-step forecast engine (eager + opt-in compile) (#254)
  ([#363](https://github.com/El3ssar/ResDAG/pull/363),
  [`3465101`](https://github.com/El3ssar/ResDAG/commit/3465101f05118a674fd95be8c916ed7e7d6ecfc9))

- **engine+init**: Compiled-native speedups, all bit-exact or opt-in
  ([`1f72922`](https://github.com/El3ssar/ResDAG/commit/1f72922dbad3e3206e555017c1fc2c52dd4d1f32))

- **init**: Power-iteration/sparse-eigs spectral-radius scaling + FSI initializer
  ([#347](https://github.com/El3ssar/ResDAG/pull/347),
  [`47350b0`](https://github.com/El3ssar/ResDAG/commit/47350b058e1fbf16061d11c6e7a012cb0d2558de))

- **reservoirs**: Vectorize NG-RC feature construction (whole-sequence)
  ([#343](https://github.com/El3ssar/ResDAG/pull/343),
  [`2ad0942`](https://github.com/El3ssar/ResDAG/commit/2ad094270ce76ed8b802086a865d0196863936c2))

### Refactoring

- **api**: Restrict tab-completion to the curated public API (PEP 562 __dir__)
  ([#353](https://github.com/El3ssar/ResDAG/pull/353),
  [`09f89ee`](https://github.com/El3ssar/ResDAG/commit/09f89ee89aa2898a4775f13977e2f5b8d0772a7e))

- **hpo**: Picklable TrialRunner with horizon-checkpoint pruning
  ([#345](https://github.com/El3ssar/ResDAG/pull/345),
  [`81923b0`](https://github.com/El3ssar/ResDAG/commit/81923b065af4857a8301df5dbb2e2168e74fd8ea))

- **init**: Unify input_scaling/connectivity contract on initializer base (#191)
  ([#350](https://github.com/El3ssar/ResDAG/pull/350),
  [`3b4310a`](https://github.com/El3ssar/ResDAG/commit/3b4310a2731b876264d7938a5705592ae0756381))

- **models**: Shared _esn_builder behind the premade factories
  ([#344](https://github.com/El3ssar/ResDAG/pull/344),
  [`eec6bb6`](https://github.com/El3ssar/ResDAG/commit/eec6bb6f42867dadd522ae5b9b667894fec0a843))

- **reservoirs**: Drop legacy ESNLayer._initialized flag
  ([#437](https://github.com/El3ssar/ResDAG/pull/437),
  [`5ed348c`](https://github.com/El3ssar/ResDAG/commit/5ed348c8401cf880cfa1cacb1f0cfa06a1bc7acf))

- **reservoirs**: Pure forward_stateless + stack-based loop on BaseReservoirLayer (#253)
  ([#361](https://github.com/El3ssar/ResDAG/pull/361),
  [`d57adc6`](https://github.com/El3ssar/ResDAG/commit/d57adc69d58514095b04580545ab7d21d94eef44))

- **structure**: Extract forecast rollout, clean up HPO, drop dead weight
  ([`69d0914`](https://github.com/El3ssar/ResDAG/commit/69d09143bf494d2d4e6c7090c65ffba88bdc7f01))

- **training**: Hoist warnings import and simplify redundant target loop
  ([#432](https://github.com/El3ssar/ResDAG/pull/432),
  [`4962a32`](https://github.com/El3ssar/ResDAG/commit/4962a32774ed21da9fdd334aab3bef2f39b7fc12))

### Testing

- **bench**: Performance-regression harness with committed targets, wired into CI (#263)
  ([#394](https://github.com/El3ssar/ResDAG/pull/394),
  [`59d7d8d`](https://github.com/El3ssar/ResDAG/commit/59d7d8df8cf9a3a178c6824051118ed850f4abfb))

- **benchmarks**: Drop flaky wall-clock flat-vs-graph forecast gate
  ([`ec5d4ec`](https://github.com/El3ssar/ResDAG/commit/ec5d4ecbb3e014c2bf58a9abaf3b9f8b708265c5))

- **core**: NG-RC end-to-end in ESNModel + ESNTrainer + forecast (#317)
  ([#366](https://github.com/El3ssar/ResDAG/pull/366),
  [`72ffdbb`](https://github.com/El3ssar/ResDAG/commit/72ffdbbaa47dae0cd43d348ac3f335a6885f51f5))

- **data**: Cover io/normalize/load_and_prepare (#251)
  ([#357](https://github.com/El3ssar/ResDAG/pull/357),
  [`98d0e81`](https://github.com/El3ssar/ResDAG/commit/98d0e81b2577e2a66dfbaba2c6cdd8b9a463e5b6))

- **ensemble**: Add n_models=1 fast-path tests for coupled ensemble
  ([#428](https://github.com/El3ssar/ResDAG/pull/428),
  [`1c60c09`](https://github.com/El3ssar/ResDAG/commit/1c60c0983671e0917bb98b1271a64bbc710de52e))

- **hpo**: Clip/prune+raw_loss, monitor remap, drivers, per-trial seeding
  ([#412](https://github.com/El3ssar/ResDAG/pull/412),
  [`40e5671`](https://github.com/El3ssar/ResDAG/commit/40e5671c13779e2cce05b88706efad1ef327c786))

- **init**: Registry-driven topology + initializer coverage sweep (#207)
  ([#359](https://github.com/El3ssar/ResDAG/pull/359),
  [`850be4b`](https://github.com/El3ssar/ResDAG/commit/850be4beab26ac9d99384cabdfb5699497722c06))

- **models**: Golden-trajectory forecast regression per premade model
  ([#405](https://github.com/El3ssar/ResDAG/pull/405),
  [`91904e3`](https://github.com/El3ssar/ResDAG/commit/91904e3d575a2965390de66e2663263612d17736))

- **readouts**: NaN/degenerate-input coverage for CG ridge solver + trainer (#184)
  ([#411](https://github.com/El3ssar/ResDAG/pull/411),
  [`4e2cb47`](https://github.com/El3ssar/ResDAG/commit/4e2cb47b6927e7a633fcf3a8f0cfb2f30dcddeb3))

- **training**: Add learnable-target reconstruction checks + seeding to ESNTrainer tests
  ([#342](https://github.com/El3ssar/ResDAG/pull/342),
  [`3885eb5`](https://github.com/El3ssar/ResDAG/commit/3885eb5fff08d89040bf45f70c2f0653ffe56f27))

- **transforms**: Backward/gradient tests for all transforms + aggregator
  ([#379](https://github.com/El3ssar/ResDAG/pull/379),
  [`b447393`](https://github.com/El3ssar/ResDAG/commit/b447393d01e195512e530c4a69aa1193cbbe3e66))

- **utils**: Consolidated io/normalize/esp_index regression suite (#250)
  ([#385](https://github.com/El3ssar/ResDAG/pull/385),
  [`56f8f2e`](https://github.com/El3ssar/ResDAG/commit/56f8f2e5f09b468eb2312d6cb21ef1f5f0fa236d))

- **utils**: Cover esp_index ESP diagnostic (#252)
  ([#358](https://github.com/El3ssar/ResDAG/pull/358),
  [`7f9526b`](https://github.com/El3ssar/ResDAG/commit/7f9526bc0d744fb40a7475f831adfc956e937fcd))


## v0.6.2 (2026-06-15)

### Bug Fixes

- Realign prepare_esn_data val with the autoregressive forecast
  ([#115](https://github.com/El3ssar/ResDAG/pull/115),
  [`1b700f0`](https://github.com/El3ssar/ResDAG/commit/1b700f019813ca0533fa2cfc39686cf7dbd332c2))

- Restore lazy composition/hpo imports so `import resdag` is warning-free
  ([#113](https://github.com/El3ssar/ResDAG/pull/113),
  [`94bb10d`](https://github.com/El3ssar/ResDAG/commit/94bb10db80da6e1f6c43dd9ced918c318020c1de))


## v0.6.1 (2026-06-15)

### Bug Fixes

- Forecast() emits genuinely autoregressive steps (slot 0, horizon, typing)
  ([#111](https://github.com/El3ssar/ResDAG/pull/111),
  [`fbc2baa`](https://github.com/El3ssar/ResDAG/commit/fbc2baa01338fd4e826e0dd9b925c532a7298868))

### Documentation

- Parallel-development protocol + roadmap pointer
  ([#110](https://github.com/El3ssar/ResDAG/pull/110),
  [`e1960d3`](https://github.com/El3ssar/ResDAG/commit/e1960d317b2074e97854b51f7ab62f46c35337a5))

- Reservoir-computing benchmark suite and comparison page
  ([#12](https://github.com/El3ssar/ResDAG/pull/12),
  [`65bf14f`](https://github.com/El3ssar/ResDAG/commit/65bf14f624d8cdae5c2d59a13e486377fbc99173))


## v0.6.0 (2026-06-14)

### Features

- Support pytorch-symbolic 1.2 and add whole-model save_full/load_full
  ([#11](https://github.com/El3ssar/ResDAG/pull/11),
  [`8fe8cdb`](https://github.com/El3ssar/ResDAG/commit/8fe8cdbf3819966450208f9c7ff54fe52af7d476))


## v0.5.0 (2026-06-12)

### Bug Fixes

- Invalid changelog config aborted the release ([#10](https://github.com/El3ssar/ResDAG/pull/10),
  [`cf96bb8`](https://github.com/El3ssar/ResDAG/commit/cf96bb83a1362c066d534925e824a764d8c5e054))

- Release pipeline detached HEAD, lean README ([#9](https://github.com/El3ssar/ResDAG/pull/9),
  [`1b99071`](https://github.com/El3ssar/ResDAG/commit/1b99071576ac911f224ceafda07b4bbbfc47188b))

### Features

- GPU performance, generalized initialization, docs platform, automated releases (0.5)
  ([#8](https://github.com/El3ssar/ResDAG/pull/8),
  [`3c40fe9`](https://github.com/El3ssar/ResDAG/commit/3c40fe9426f6213a61c341b0eed55d22f28fd254))

### Refactoring

- Rename ReadoutLayer import and introduce base_readout module
  ([#6](https://github.com/El3ssar/ResDAG/pull/6),
  [`1d39211`](https://github.com/El3ssar/ResDAG/commit/1d3921147fcb939388e02fe527b5a014d53ee7ca))

- Reorganize layout — composition→core, custom→transforms
  ([#6](https://github.com/El3ssar/ResDAG/pull/6),
  [`1d39211`](https://github.com/El3ssar/ResDAG/commit/1d3921147fcb939388e02fe527b5a014d53ee7ca))


## v0.4.0 (2026-05-17)


## v0.3.1 (2026-04-16)


## v0.3.0 (2026-03-02)


## v0.2.0 (2026-01-18)


## v0.1.3 (2026-01-14)


## v0.1.2 (2026-01-14)


## v0.1.1 (2026-01-14)

- Initial Release
