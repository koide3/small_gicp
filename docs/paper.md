---
title: 'small_gicp: Efficient and parallel algorithms for point cloud registration'
tags:
  - C++
  - Python
  - Point cloud registration
authors:
  - name: Kenji Koide
    orcid: 0000-0001-5361-1428
    corresponding: true
    affiliation: "1"
affiliations:
 - name: National Institute of Advanced Industrial Science and Technology (AIST), Japan
   index: 1
date: 22 June 2024
bibliography: paper.bib
---

# Summary

Point cloud registration is a task of aligning two point clouds measured by 3D ranging
sensors, for example, LiDARs and range cameras. Iterative point cloud registration,
also known as fine registration or local registration, is particularly crucial. This
process iteratively performs proximity-based point correspondence search and minimizes
the distance between corresponding points. Iterative closest point (ICP) and its variants,
such as Generalized ICP, are representative iterative point cloud registration algorithms.
They are widely used in applications like autonomous vehicle localization, place recognition,
and object classification. Since these applications often require real-time or near-real-time
processing, speed is a critical factor in point cloud registration routines.

**small_gicp** provides efficient and parallel algorithms to create an extremely
fast point cloud registration pipeline. It offers parallel implementations of 
downsampling, nearest neighbor search, local feature extraction, and registration
to accelerate the entire process.
small_gicp is implemented as a header-only C++ library with minimal dependencies
to offer efficiency, portability, and customizability.

# Statement of need

There are several point cloud processing libraries, and PCL [@Rusu] and Open3D
[@Zhou] are the notable ones among them.
Although they offer numerous functionalities, including those required for point cloud
registration, they present several challenges for practical applications and scientific
research.

**Processing speed:**
A typical point cloud registration pipeline includes processes such as downsampling,
nearest neighbor search (e.g., KdTree construction), local feature estimation, and
registration error minimization.
PCL and Open3D support multi-threading only for parts of these processes (feature
estimation and registration error minimization), with the remaining single-threaded
parts often limiting the overall processing speed.
Additionally, the multi-thread implementations in these libraries can have significant
overheads, reducing scalability to many-core CPUs.
These issues make it difficult to meet real-time processing requirements, especially on
low-specification CPUs. It is also difficult to fully utilize the computational power 
of modern high-end CPUs.

**Customizability:**
Customizing the internal workings (e.g., replacing the registration cost function or
changing the correspondence search method) of existing implementations is challenging
due to hard-coded processes. This poses a significant hurdle for research and development,
where testing new cost functions and search algorithms is essential.

**small_gicp:**
To address these issues and accelerate the development of point cloud registration-related systems,
we designed small_gicp with the following features:

- Fully parallelized point cloud preprocessing and registration algorithms with minimal overhead,
  offering up to 2x speed gain in single-threaded scenarios and better scalability in multi-core
  environments.

- A modular and customizable framework using C++ templates, allowing easy customization of the
  algorithm's internal workings while maintaining efficiency.

- A header-only C++ library implementation for easy integration into user projects, with Python bindings
  provided for collaborative use with other libraries (e.g., Open3D).

# Functionalities

**small_gicp** implements several preprocessing algorithms related to point cloud registration, and
ICP variant algorithms (point-to-point ICP, point-to-plane ICP, and Generalized ICP based on
distribution-to-distribution correspondence).

- Downsampling
    - Voxelgrid sampling
    - Random sampling
- Nearest neighbor search and point accumulation structures
    - KdTree
    - Linear iVox (supports incremental points insertion and LRU-cache-based voxel deletion) [@Bai]
    - Gaussian voxelmap (supports incremental points insertion and LRU-cache-based voxel deletion) [@Koide]
- Registration error functions
    - Point-to-point ICP error [@Zhang]
    - Point-to-plane ICP error
    - Generalized ICP error [@Segal]
    - Robust kernels
- Least squares optimizers
    - GaussNewton optimizer
    - LevenbergMarquardt optimizer

# Benchmark results

- Single-threaded and multi-threaded (6 threads) `small_gicp::voxelgrid_sampling` are approximately 1.3x and 3.2x faster than `pcl::VoxelGrid`, respectively.
- Multi-threaded construction of `small_gicp::KdTree` can be up to 6x faster than that of `nanoflann`.
- Single-threaded `small_gicp::GICP` is about 2.4x faster than `pcl::GICP`, with the multi-threaded version showing better scalability.

More details can be found at https://github.com/koide3/small_gicp/blob/master/BENCHMARK.md.

# Acknowledgements

This work was supported in part by JSPS KAKENHI Grant Number 23K16979.

# References