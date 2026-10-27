CUTLASS is pinned to NVIDIA v3.5.1, commit
`f7b19de32c5d1f3cedfc735c2849f12b537522ee`.

Run `bash scripts/setup_dependencies.sh` on the login node. The GitCode mirror
is used to stage the exact pinned source; the source headers retain NVIDIA's
BSD license. The checkout is excluded from this repository and no patches are
applied to it. CMake accepts `-DCUTLASS_ROOT=/path/to/another/v3.5.1/checkout`.
