# small_gicp (fast_gicp2)

**small_gicp** is a header-only C++ library that provides efficient and parallelized fine point cloud registration algorithms (ICP, Point-to-Plane ICP, GICP, VGICP, etc.). It is a refined and optimized version of its predecessor, [fast_gicp](https://github.com/SMRT-AIST/fast_gicp), with the following features. 

- **Highly ptimized** : The implementation of the core registration algorithm is further optimized from that in fast_gicp. It can provide up to 2x speed up compared to fast_gicp.
- **All parallerized** : small_gicp provides parallelized implementations of several algorithms in the point cloud registration process (Downsampling, KdTree construction, Normal/covariance estimation). As a parallelism backend, either (or both) of [OpenMP](https://www.openmp.org/) and [Intel TBB](https://github.com/oneapi-src/oneTBB) can be used. 
- **Minimum dependency** : Only [Eigen](https://eigen.tuxfamily.org/) (and bundled [nanoflann](https://github.com/jlblancoc/nanoflann) and [Sophus](https://github.com/strasdat/Sophus)) are required at a minimum. Optionally, it provides the [PCL](https://pointclouds.org/) registration interface so that it can be used as a drop-in replacement in many systems.
- **Customizable** : small_gicp is implemented with the trait mechanism that allows feeding any custom point cloud class to the registration algorithm. Furthermore, the template-based implementation allows customizing the regisration process with your original correspondence estimator and registration factors.
- **Python bindinds** (coming soon) : The isolation from PCL makes the small_gicp's python bindinds more portable and connectable to other libraries seamlessly. 

[![Build](https://github.com/koide3/small_gicp/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/small_gicp/actions/workflows/build.yml)

## Requirements

This library uses some C++20 features. While porting it to older environments should be easy, we have no plan to support environments older than Ubuntu 22.04.

## Installation

### C++

small_gicp is a header-only library. You can just download and drop it in your project directory to use it.

Meanwhile, if you need only basic point cloud registration, you can build and install the helper library as follows.

```bash
sudo apt-get install libeigen3-dev libomp-dev

cd small_gicp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
sudo make install
```

### Python

Coming soon.

## Usage (C++)

The following examples assume `using namespace small_gicp` is placed somewhere.

### Using helper library ([01_basic_resigtration.cpp](https://github.com/koide3/small_gicp/blob/master/src/example/01_basic_registration.cpp))

The helper library (`registration_helper.hpp`) enables easily processing point clouds represented as `std::vector<Eigen::Vector(3|4)(f|d)>`.
<details><summary>Expand</summary>

`small_gicp::align` takes two point clouds (`std::vectors` of `Eigen::Vector(3|4)(f|d)`) and returns a registration result (estimated transformation and some information on the optimization result). This is the easiest way to use **small_gicp** but causes an overhead for duplicated preprocessing.

```cpp
#include <small_gicp/registration/registration_helper.hpp>

std::vector<Eigen::Vector3d> target_points = ...;   // Any of Eigen::Vector(3|4)(f|d) can be used
std::vector<Eigen::Vector3d> source_points = ...;   // 

RegistrationSetting setting;
setting.num_threads = 4;                    // Number of threads to be used
setting.downsampling_resolution = 0.25;     // Downsampling resolution
setting.max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
RegistrationResult result = align(target_points, source_points, init_T_target_source, setting);

Eigen::Isometry3d T = result.T_target_source;  // Estimated transformation
size_t num_inliers = result.num_inliers;       // Number of inlier source points
Eigen::Matrix<double, 6, 6> H = result.H;      // Final Hessian matrix (6x6)
```

There is also a way to perform preprocessing and registration separately. This enables saving the time for preprocessing in case registration is performed several times for a same point cloud (e.g., typical odometry estimation based on scan-to-scan matching).

```cpp
#include <small_gicp/registration/registration_helper.hpp>

std::vector<Eigen::Vector3d> target_points = ...;   // Any of Eigen::Vector(3|4)(f|d) can be used
std::vector<Eigen::Vector3d> source_points = ...;   // 

int num_threads = 4;                    // Number of threads to be used
double downsampling_resolution = 0.25;    // Downsampling resolution
int num_neighbors = 10;                 // Number of neighbor points used for normal and covariance estimation

// std::pair<PointCloud::Ptr, KdTree<PointCloud>::Ptr>
auto [target, target_tree] = preprocess_points(target_points, downsampling_resolution, num_neighbors, num_threads);
auto [source, source_tree] = preprocess_points(source_points, downsampling_resolution, num_neighbors, num_threads);

RegistrationSetting setting;
setting.num_threads = num_threads;
setting.max_correspondence_distance = 1.0;  // Maximum correspondence distance between points (e.g., triming threshold)

Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
RegistrationResult result = align(*target, *source, *target_tree, init_T_target_source, setting);

Eigen::Isometry3d T = result.T_target_source;  // Estimated transformation
size_t num_inliers = result.num_inliers;       // Number of inlier source points
Eigen::Matrix<double, 6, 6> H = result.H;      // Final Hessian matrix (6x6)
```

</details>

### Using with PCL interface ([02_basic_resigtration_pcl.cpp](https://github.com/koide3/small_gicp/blob/master/src/example/02_basic_resigtration_pcl.cpp))

The PCL interface allows using small_gicp as a drop-in replacement for `pcl::GeneralizedIterativeClosestPoint`. It is also possible to directly feed `pcl::PointCloud` to algorithms implemented in small_gicp.

<details><summary>Expand</summary>

```cpp
#include <small_gicp/pcl/pcl_registration.hpp>

pcl::PointCloud<pcl::PointXYZ>::Ptr raw_target = ...;
pcl::PointCloud<pcl::PointXYZ>::Ptr raw_source = ...;

// small_gicp::voxelgrid_downsampling can directly operate on pcl::PointCloud.
pcl::PointCloud<pcl::PointXYZ>::Ptr target = voxelgrid_sampling_omp(*raw_target, 0.25);
pcl::PointCloud<pcl::PointXYZ>::Ptr source = voxelgrid_sampling_omp(*raw_source, 0.25);

// RegistrationPCL is derived from pcl::Registration and has mostly the same interface as pcl::GeneralizedIterativeClosestPoint.
RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> reg;
reg.setNumThreads(4);
reg.setCorrespondenceRandomness(20);
reg.setMaxCorrespondenceDistance(1.0);
reg.setVoxelResolution(1.0);
reg.setRegistrationType("VGICP");  // or "GICP" (default = "GICP")

// Set input point clouds.
reg.setInputTarget(target);
reg.setInputSource(source);

// Align point clouds.
auto aligned = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
reg.align(*aligned);

// Swap source and target and align again.
// This is useful when you want to re-use preprocessed point clouds for successive registrations (e.g., odometry estimation).
reg.swapSourceAndTarget();
reg.align(*aligned);
```

It is also possible to directly feed `pcl::PointCloud` to `small_gicp::Registration`. Because all preprocessed data are exposed in this way, you can easily re-use them to obtain the best efficiency.

```cpp
#include <small_gicp/pcl/pcl_registration.hpp>

pcl::PointCloud<pcl::PointXYZ>::Ptr raw_target = ...;
pcl::PointCloud<pcl::PointXYZ>::Ptr raw_source = ...;

// Downsample points and convert them into pcl::PointCloud<pcl::PointCovariance>.
pcl::PointCloud<pcl::PointCovariance>::Ptr target = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(*raw_target, 0.25);
pcl::PointCloud<pcl::PointCovariance>::Ptr source = voxelgrid_sampling_omp<pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointCovariance>>(*raw_source, 0.25);

// Estimate covariances of points.
const int num_threads = 4;
const int num_neighbors = 20;
estimate_covariances_omp(*target, num_neighbors, num_threads);
estimate_covariances_omp(*source, num_neighbors, num_threads);

// Create KdTree for target and source.
auto target_tree = std::make_shared<KdTreeOMP<pcl::PointCloud<pcl::PointCovariance>>>(target, num_threads);
auto source_tree = std::make_shared<KdTreeOMP<pcl::PointCloud<pcl::PointCovariance>>>(source, num_threads);

Registration<GICPFactor, ParallelReductionOMP> registration;
registration.reduction.num_threads = num_threads;
registration.rejector.max_dist_sq = 1.0;

// Align point clouds. Note that the input point clouds are pcl::PointCloud<pcl::PointCovariance>.
auto result = registration.align(*target, *source, *target_tree, Eigen::Isometry3d::Identity());
```

</details>

### Using `small_gicp::Registration` template (`registration.hpp`)

If you want to fine-control and customize the registration process, use `small_gicp::Registration` template that allows modifying the inner algorithms and parameters.
<details><summary>Expand</summary>

```cpp
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration.hpp>

std::vector<Eigen::Vector3d> target_points = ...;   // Any of Eigen::Vector(3|4)(f|d) can be used
std::vector<Eigen::Vector3d> source_points = ...;   // 

int num_threads = 4;
double downsampling_resolution = 0.25;
int num_neighbors = 10;
double max_correspondence_distance = 1.0;

// Convert to small_gicp::PointCloud
auto target = std::make_shared<PointCloud>(target_points);
auto source = std::make_shared<PointCloud>(source_points);

// Downsampling
target = voxelgrid_downsampling_omp(*target, downsampling_resolution, num_threads);
source = voxelgrid_downsampling_omp(*source, downsampling_resolution, num_threads);

// Create KdTree
auto target_tree = std::make_shared<KdTreeOMP<PointCloud>>(target, num_threads);
auto source_tree = std::make_shared<KdTreeOMP<PointCloud>>(source, num_threads);

// Estimate point covariances
estimate_covariances_omp(*target, *target_tree, num_neighbors, num_threads);
estimate_covariances_omp(*source, *source_tree, num_neighbors, num_threads);

// GICP + OMP-based parallel reduction
Registration<GICPFactor, ParallelReductionOMP> registration;
registration.reduction.num_threads = num_threads;
registration.rejector.max_dist_sq = max_correspondence_distance * max_correspondence_distance;

// Align point clouds
Eigen::Isometry3d init_T_target_source = Eigen::Isometry3d::Identity();
auto result = registration.align(*target, *source, *target_tree, init_T_target_source);

Eigen::Isometry3d T = result.T_target_source;  // Estimated transformation
size_t num_inliers = result.num_inliers;       // Number of inlier source points
Eigen::Matrix<double, 6, 6> H = result.H;      // Final Hessian matrix (6x6)
```

Custom registration example:

```cpp
using PerPointFactor = PointToPlaneICPFactor;       // Point-to-plane ICP
using GeneralFactor = RestrictDoFFactor;            // DoF restriction
using Reduction = ParallelReductionTBB;             // TBB-based parallel reduction
using CorrespondenceRejector = DistanceRejector;    // Distance-based correspondence rejection
using Optimizer = LevenbergMarquardtOptimizer;      // Levenberg marquardt optimizer

Registration<PerPointFactor, Reduction, GeneralFactor, CorrespondenceRejector, Optimizer> registration;
registration.general_factor.set_translation_mask(Eigen::Vector3d(1.0, 1.0, 0.0));   // XY-translation only
registration.general_factor.set_ratation_mask(Eigen::Vector3d(0.0, 0.0, 1.0));      // Z-rotation only
registration.optimizer.init_lambda = 1e-3;                                          // Initial damping scale
```

</details>

## Usage (Python)

Coming soon.

## Benchmark

### Downsampling

- Single-threaded `small_gicp::voxelgrid_sampling` is about 1.3x faster than `pcl::VoxelGrid`.
- Multi-threaded `small_gicp::voxelgrid_sampling_tbb` (6 threads) is about 3.2x faster than `pcl::VoxelGrid`.
- `small_gicp::voxelgrid_sampling` gives accurate downsampling results (almost identical to those of `pcl::VoxelGrid`) while `pcl::ApproximateVoxelGrid` yields spurious points (up to 2x points).
- `small_gicp::voxelgrid_sampling` can process a larger point cloud with a fine voxel resolution compared to `pcl::VoxelGrid` (for a point cloud of 150m width, minimum voxel resolution can be 0.07 mm).

![downsampling_comp](docs/assets/downsampling_comp.png)

### KdTree construction

- Multi-threaded implementation (TBB and OMP) can be up to 4x faster than the single-threaded one (All the implementations are based on nanoflann).
- Basically the processing speed get faster as the number of threads increases, but the speed gain is not monotonic sometimes (because of the scheduling algorithm or some CPU(AMD 5995WX)-specific issues?).
- This benchmark only compares the construction time (query time is not included). 

![kdtree_time](docs/assets/kdtree_time.png)

### Odometry estimation

- Single-thread `small_gicp::GICP` is about 2.4x and 1.9x faster than `pcl::GICP` and `fast_gicp::GICP`, respectively.
- `small_gicp::(GICP|VGICP)` shows a better multi-thread scalability compared to `fast_gicp::(GICP|VGICP)`.
- `small_gicp::GICP` parallelized with [TBB flow graph](https://github.com/koide3/small_gicp/blob/master/src/odometry_benchmark_small_gicp_tbb_flow.cpp) shows an excellent scalablity to many-threads situations (~128 threads) but with a latency degradation.

![odometry_time](docs/assets/odometry_time.png)

## License
This package is released under the MIT license.

If you find this package useful for your project, please consider leaving a comment here. It would help the author gain internal recognition in his organization and keep working on this project.

## Papers
- Kenji Koide, Masashi Yokozuka, Shuji Oishi, and Atsuhiko Banno, Voxelized GICP for Fast and Accurate 3D Point Cloud Registration, ICRA2021

## Contact

[Kenji Koide](https://staff.aist.go.jp/k.koide/), National Institute of Advanced Industrial Science and Technology (AIST)