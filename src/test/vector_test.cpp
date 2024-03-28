#include <random>
#include <small_gicp/util/fast_floor.hpp>
#include <small_gicp/util/vector3i_hash.hpp>

#include <gtest/gtest.h>

using namespace small_gicp;

TEST(VectorTest, HashTest) {
  std::mt19937 mt;

  for (int i = 0; i < 1000; i++) {
    std::uniform_int_distribution<> dist(-1000, 1000);
    const Eigen::Vector3i v(dist(mt), dist(mt), dist(mt));
    EXPECT_EQ(XORVector3iHash::hash(v), XORVector3iHash()(v));
    EXPECT_TRUE(XORVector3iHash::equal(v, v));
  }
}

TEST(VectorTest, FloorTest) {
  std::mt19937 mt;
  for (int i = 0; i < 1000; i++) {
    std::uniform_real_distribution<> dist(-1000.0, 1000.0);
    const Eigen::Vector4d v(dist(mt), dist(mt), dist(mt), 1.0);
    const Eigen::Array4i floor1 = fast_floor(v);
    EXPECT_EQ(floor1[0], std::floor(v[0]));
    EXPECT_EQ(floor1[1], std::floor(v[1]));
    EXPECT_EQ(floor1[2], std::floor(v[2]));
    EXPECT_EQ(floor1[3], std::floor(v[3]));
  }
}