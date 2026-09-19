# multinode transport — which fabric protocol, and what changes here

Decision record for extending `distBackend` / `PartitionedDevices` past one node.
Companion to [`backend/README.md`](README.md) (how one mat-vec is partitioned) and
[`docs/dip_operator_memory.md`](../docs/dip_operator_memory.md) (why the satellite is matrix-free).

## Verdict

| | |
|---|---|
| **Data plane** | **NCCL** — `ncclGroupStart` + per-peer `ncclSend`/`ncclRecv`, GPUDirect RDMA over IB |
| **Reductions** | `ncclAllGather` of partials + **serial sum in ascending rank order**. Never `ncclAllReduce`. |
| **Bootstrap** | plain Go TCP, or the 128-byte `ncclUniqueId` written to GPFS. **No MPI needed.** |
| **Control plane** (rank registry, checkpoint coordination, health) | Go RPC/gRPC — fine, and only here |
| Rejected | MPI (Go ergonomics), RPC for bulk data (host staging) |

NCCL uses the **same NVLink P2P intra-node that `PeerCopier` uses today**, so single-node
behaviour and single-node timings are unchanged. Only inter-rank pairs take the IB path.

## This is not a bandwidth decision — comm is ~1000x below compute

Production triplet, figures from `backend/README.md` and `dip/matfree_dist.go`:

- slab per device per chunk = `n*w*8` = 14,766,249 x 64 x 8 = **7.56 GB**
- **26 chunks** per mat-vec (`ceil(b/w)`, b = main = 1653, w = `SatChunkCols` = 64)
- block-0 apply measured **40h15m** at nd=8 (job 14158038)

At 2 nodes x 8 GPUs (R=16) each device's slab is half off-node:

- **30 GB IB ingress per node per chunk** (8 devices x 3.78 GB) -> 786 GB per node per mat-vec
- compute per chunk at R=16 ≈ 46 min
- **required bandwidth ≈ 11 MB/s.** One 100 Gb HDR port is ~12.5 GB/s -> **~1100x headroom**

A 100x kernel speedup from the Go/CUDA work still leaves ~10x. Sync count is equally free:
2 fences per chunk x 26 = **52 global barriers per mat-vec**.

**Consequence:** pick the transport for ergonomics and determinism, not throughput. Do not spend
effort tuning IB. Do spend it on load balance (below).

## Mapping onto this package

| today | multinode |
|---|---|
| `PeerCopier.PeerCopy2D`, nd^2 transfers per chunk (`dip.gatherSlabs`) | one `ncclGroupStart` / `ncclSend`+`ncclRecv` per peer / `ncclGroupEnd`; NCCL routes NVLink vs IB per pair |
| `enablePeers` / `AllPeered` (`distributed.go:180`, `:790`) | `ncclCommInitRank` once at startup; predicate becomes "comm exists" |
| `distBackend.bound` over local devices | same array, over **global ranks** |
| `Gemm(transA=true)` all-reduce (`distributed.go:605`) | `ncclAllGather` of the per-rank partials, then the existing ascending-order serial sum |
| `Download`/`Upload` host round-trip inside that reduce | device-resident; drops the host staging entirely |

- cgo surface is ~8 NCCL entry points. The build already does cgo + CUDA.
- `gatherSlabs` is a strided 2D gather (src `ld=rows`, dst `ld=n`). NCCL has no `allgatherv`;
  grouped `ncclSend`/`ncclRecv` reproduces `PeerCopy2D` semantics exactly and fuses the whole
  nd^2 loop into one op.

### Hard dependency: NCCL requires streams

Every NCCL call is stream-ordered. The current design is blocking null-stream round-trips through
one owning goroutine per GPU. **The stream/event work must land before the multinode work** —
these two items are not independent, and the proposal should say so.

## Determinism rule (non-negotiable)

Results must stay bit-reproducible run to run, validated against theADCcode.

- **Gather/send-recv do no arithmetic** -> bit-exact by construction, regardless of completion
  order. This is the property `gatherSlabs` already documents; NCCL preserves it.
- **The only cross-rank reduction is `Gemm(transA=true)`.** It is `main x main` (1653^2 x 8 =
  21.9 MB per rank) and runs ~6x per Lanczos iteration under Mode B — alpha, Gram, and the
  two-pass CGS2 against the two live blocks (`lanczos/lowmem.go:217`). At R=16 that is ~2 GB per
  rank per mat-vec against 786 GB of slab traffic: irrelevant to wall time, so buy determinism
  with it.
- **Allgather + fixed-rank-order sum is strictly stronger than pinning `NCCL_ALGO=Ring`**: the
  reduction order depends only on rank index, not on NCCL's internal algorithm selection, node
  count, or transfer timing.

For the proposal: *"our only cross-rank reduction is main x main; we allgather it and sum in fixed
rank order, so the distributed result is bit-identical to the single-node one by construction."*

## Rejected

**MPI.** Performance would be fine (see headroom). The objections are Go-specific:

- every MPI call must sit on a locked OS thread (`runtime.LockOSThread`), fighting the
  goroutine-per-device design and `goDevices`' fault-to-solver unwinding
- leans hard on `MPI_THREAD_MULTIPLE`, whose quality is uneven
- `mpirun` launch model sits awkwardly with the Go runtime and current SLURM usage
- CUDA-aware OpenMPI delegates RDMA to UCX anyway — a layer on top of NCCL's own substrate

Keeps two advantages: reviewers expect it, and it is the only route if CPU-side collectives are
ever needed. Common hybrid: MPI *only* to broadcast `ncclUniqueId`. Not required here.

**RPC/gRPC for bulk data.** IPoIB -> kernel TCP -> pageable host memory -> `Download`/`Upload`
round-trips. That is `newSatelliteMatFreeDistributed`, the fallback written to avoid exactly this,
and the host-staging churn behind the 733 GB OOM kills (jobs 14040959 / 14075367). It would meet
11 MB/s. Do not.

## Later, not first: NVSHMEM

Semantically the better fit — the gather *is* a one-sided read of a peer's row band
(`nvshmem_getmem`), issuable **from inside the kernel**, which would fuse gather into the operator
fill and delete the 2 fences per chunk. Costs: kernels get rewritten, thinner Go story, and it buys
latency the numbers above say is not needed. Phase 2, conditional on the stream work landing.

(Unrelated aside: **SMC-R** / `AF_SMC` lets an unmodified Go `net.Conn` ride RDMA — useful for
checkpoint traffic and control plane. No GPUDirect, so not for slabs.)

## HELIX facts and open checks

- IB present: `mlx5_2` = 100 Gb/s HDR (`/sys/class/infiniband/`). Also two 40 GbE + a 10 GbE bond.
- **NCCL is not a module** and is not in the CUDA toolkit dirs (`devel/cuda/{11.6..13.0}`).
  Install it (the `nvidia-nccl-cu12` wheel is the least-friction route).
- OpenMPI 4.1 exists only as a dependency of `devel/scorep` / `lib/hdf5`.
- **OPEN — verify before committing to multinode:** `sinfo` is permission-denied from the login
  node, so it is unconfirmed that a partition exists that can allocate >=2 GPU nodes, and what its
  MaxTime is. `gpu-single` caps at 5 days. This is the one assumption underneath the whole plan.

## The actual risk: load imbalance, not the network

The applier is bulk-synchronous — 2 fences per chunk, so the slowest rank sets the pace — and
`active[d]` already marks partitions that legitimately do no work (empty satellite row band, or no
block in any batch). Going 8 -> 32 ranks makes stragglers, not bandwidth, the thing that bites.
Partition-boundary quality (`dip.PartitionBounds`) is the lever.
