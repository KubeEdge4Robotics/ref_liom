// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once
// std头文件
#include <cmath>
#include <vector>
#include <list>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <string>
// Eigen头文件
#include <Eigen/Eigen>
// ros头文件
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
// TimeSyncronizer headers
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
// pcl 头文件
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
// ceres 相关头文件
#include <ceres/ceres.h>
#include <ceres/rotation.h>
// igl
#include <igl/opengl/glfw/Viewer.h>
#include <igl/AABB.h>
#include <igl/point_mesh_squared_distance.h>

#include "tic_toc.h"

struct PointMeshFactor
{
	PointMeshFactor(const Eigen::Vector3d &lidar_pt_, const Eigen::Vector3d &mesh_pt_)
			: lidar_pt(lidar_pt_), mesh_pt(mesh_pt_) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_1{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_1{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> curr_pt{T(lidar_pt.x()), T(lidar_pt.y()), T(lidar_pt.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_1 * curr_pt + t_1;

		residual[0] = point_w.x() - T(mesh_pt.x());
		residual[1] = point_w.y() - T(mesh_pt.y());
		residual[2] = point_w.z() - T(mesh_pt.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &lidar_pt_, const Eigen::Vector3d &mesh_pt_)
	{
		return (new ceres::AutoDiffCostFunction<
						PointMeshFactor, 3, 4, 3>(
				new PointMeshFactor(lidar_pt_, mesh_pt_)));
	}

	Eigen::Vector3d lidar_pt, mesh_pt;
};

struct Mesh
{
	Eigen::MatrixXd vertices; // vertices : list of vertex positions 顶点集
	Eigen::MatrixXi facets;		// facets : list of mesh primitives(vetex indices of one facet)一个面片包含的顶点索引
														// 一行代表一个面片包含的顶点索引
	Eigen::Vector3f min_pt;		// Mesh bounding box size
	Eigen::Vector3f max_pt;
};
// Oriented Bounding Box
class OBB
{
public:
	Eigen::Vector3f min_pt; // min_(x,y,z)
	Eigen::Vector3f max_pt; // max_(x,y,z)
	// world frame to model frame
	Eigen::Vector3f rot;	 //(roll, pitch, yaw)
	Eigen::Vector3f trans; //(x,y,z)
	// default constructor
	OBB() : min_pt(0, 0, 0), max_pt(0, 0, 0), rot(0, 0, 0), trans(0, 0, 0){};
};
// 先验模型
class PriorModel
{
public:
	OBB bbox;
	std::shared_ptr<igl::AABB<Eigen::MatrixXd, 3>> tree_ptr;
	std::shared_ptr<Mesh> mesh_ptr;
	PriorModel() : bbox(), mesh_ptr(nullptr), tree_ptr(nullptr){};
	PriorModel(const std::shared_ptr<Mesh> &m) : mesh_ptr(m), tree_ptr(new igl::AABB<Eigen::MatrixXd, 3>)
	{
		tree_ptr->init(mesh_ptr->vertices, mesh_ptr->facets);
	};
};

inline double rad2deg(double radians)
{
	return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
	return degrees * M_PI / 180.0;
}

template <typename T>
sensor_msgs::PointCloud2::Ptr publishCloud(const ros::Publisher &thisPub, const T &thisCloud, const ros::Time &thisStamp, const std::string &thisFrame)
{
	sensor_msgs::PointCloud2::Ptr tempCloud(new sensor_msgs::PointCloud2);
	pcl::toROSMsg(*thisCloud, *tempCloud);
	tempCloud->header.stamp = thisStamp;
	tempCloud->header.frame_id = thisFrame;
	if (thisPub.getNumSubscribers() != 0)
		thisPub.publish(tempCloud);
	return tempCloud;
}

inline Eigen::Affine3f poseMsg2Affine3f(const geometry_msgs::Pose &msg)
{
	Eigen::Vector3f t(msg.position.x, msg.position.y, msg.position.z);
	Eigen::Quaternionf q(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
	Eigen::Affine3f aff = Eigen::Affine3f::Identity();
	aff.rotate(q);
	aff.pretranslate(t);
}
inline Eigen::Affine3d poseMsg2Affine3d(const geometry_msgs::Pose &msg)
{
	Eigen::Vector3d t(msg.position.x, msg.position.y, msg.position.z);
	Eigen::Quaterniond q(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
	Eigen::Affine3d aff = Eigen::Affine3d::Identity();
	aff.rotate(q);
	aff.pretranslate(t);
	return aff;
}
inline geometry_msgs::Pose Affine3d2poseMsg(const Eigen::Affine3d &aff)
{
	geometry_msgs::Pose pose;
	Eigen::Quaterniond q(aff.rotation());
	pose.orientation.x = q.x();
	pose.orientation.y = q.y();
	pose.orientation.z = q.z();
	pose.orientation.w = q.w();
	pose.position.x = aff.translation().x();
	pose.position.y = aff.translation().y();
	pose.position.z = aff.translation().z();
	return pose;
}
class Point2meshICP
{
protected:
	double max_correspondence_dist_;
	double fitness_;
	int outlier_num_;
	Eigen::Affine3d transform_;
	double outlier_threshold_;
	bool converged_;
	bool set_rot_constant_;

public:
	// constructor
	Point2meshICP() : max_correspondence_dist_(0.5), fitness_(std::numeric_limits<double>::max()),
										outlier_num_(0), transform_(Eigen::Affine3d::Identity()), outlier_threshold_(1.0),
										converged_(false), set_rot_constant_(false){};
	Point2meshICP(double max_dist) : max_correspondence_dist_(0.5), fitness_(std::numeric_limits<double>::max()),
																	 outlier_num_(0), transform_(Eigen::Affine3d::Identity()), outlier_threshold_(1.0),
																	 converged_(false), set_rot_constant_(false){};
	void setRotCostant()
	{
		set_rot_constant_ = true;
	}
	void setRotVariable()
	{
		set_rot_constant_ = false;
	}
	// set max outliers proportion
	void setOutlierThreshold(double threshold)
	{
		outlier_threshold_ = threshold;
	}
	// set max distance between correspondences
	void setMaxCorrespondenceDist(double max_dist)
	{
		max_correspondence_dist_ = max_dist;
	}
	// solve one point-mesh ICP
	bool operator()(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_model, std::shared_ptr<PriorModel> prior_model, pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud)
	{
		converged_ = false; // 重置收敛标志
		outlier_num_ = 0;		// 重置outlier_num_
		// 搜索虚拟点+点云网格配准
		Eigen::VectorXd sqrD;						// list of squared distances
		Eigen::VectorXi indices_facets; // list of indices into Element of closest mesh primitive
		Eigen::MatrixXd closest_points; // list of closest points
		// ceres ICP
		Eigen::MatrixXd query_points(cloud_segmented_model->size(), 3); // query_points : list of query points
		for (int i = 0; i < cloud_segmented_model->size(); i++)
		{
			query_points(i, 0) = cloud_segmented_model->points[i].x;
			query_points(i, 1) = cloud_segmented_model->points[i].y;
			query_points(i, 2) = cloud_segmented_model->points[i].z;
		}
		// 更新mesh匹配
		mesh_cloud->clear();
		// pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		TicToc t_point_mesh; // 搜索匹配点的时间
		prior_model->tree_ptr->squared_distance(prior_model->mesh_ptr->vertices, prior_model->mesh_ptr->facets, query_points, sqrD, indices_facets, closest_points);
		std::cout << "*******************************" << std::endl;
		std::cout << "prior_lidar_pts size: " << cloud_segmented_model->size() << "\n"
							<< "point-mesh time (ms): " << t_point_mesh.toc() << std::endl;
		uint32_t meshnum = closest_points.rows(); // mesh虚拟点数目
		for (uint32_t i = 0; i < meshnum; i++)
		{
			pcl::PointXYZ pointc; // 匹配最近点
			pointc.x = closest_points(i, 0);
			pointc.y = closest_points(i, 1);
			pointc.z = closest_points(i, 2);
			mesh_cloud->points.push_back(pointc);
		}
		double scale_mesh = 1.0 / meshnum; // point-to-mesh权重系数
		// 构造损失函数
		double parameter[7] = {0, 0, 0, 1, 0, 0, 0}; // 迭代初值
		Eigen::Map<Eigen::Quaterniond> q(parameter);
		Eigen::Map<Eigen::Vector3d> t(parameter + 4);
		ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);//ceres::HuberLoss(0.1); 
		ceres::LossFunction *loss_function_scaled =
				new ceres::ScaledLoss(loss_function, scale_mesh, ceres::Ownership::DO_NOT_TAKE_OWNERSHIP);
		ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
		ceres::Problem::Options problem_options;
		ceres::Problem problem(problem_options);
		problem.AddParameterBlock(parameter, 4, q_parameterization); // 添加参数块，旋转四元数
		problem.AddParameterBlock(parameter + 4, 3);
		if (set_rot_constant_)
		{
			problem.SetParameterBlockConstant(parameter);
		}
		else
		{
			problem.SetParameterBlockVariable(parameter);
		}
		for (int i = 0; i < meshnum; i++)
		{
			pcl::PointXYZ &lidarPt = cloud_segmented_model->points[i];
			pcl::PointXYZ &meshPt = mesh_cloud->points[i];
			Eigen::Vector3d lidar_pt(lidarPt.x, lidarPt.y, lidarPt.z);
			Eigen::Vector3d mesh_pt(meshPt.x, meshPt.y, meshPt.z);
			ceres::CostFunction *cost_function = PointMeshFactor::Create(lidar_pt, mesh_pt);
			problem.AddResidualBlock(cost_function, loss_function_scaled, parameter, parameter + 4);
			if ((lidar_pt - mesh_pt).norm() > max_correspondence_dist_)
			{
				// 异常值 + 1
				outlier_num_++;
			}
		}
		double outlier_proportion = static_cast<double>(outlier_num_) / static_cast<double>(cloud_segmented_model->size());
		// if (outlier_proportion > outlier_threshold_)
		// 	return false;
		TicToc t_solver;
		ceres::Solver::Options options;
		options.minimizer_type = ceres::TRUST_REGION;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // ceres::DENSE_QR;
		options.dense_linear_algebra_library_type = ceres::LAPACK;
		options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
		options.max_num_iterations = 100;
		options.minimizer_progress_to_stdout = false;
		options.check_gradients = false;
		options.gradient_check_relative_precision = 1e-4;
		options.parameter_tolerance = 1e-10;
		options.function_tolerance = 1e-10;
		options.gradient_tolerance = 1e-14;
		options.max_solver_time_in_seconds = 0.5;
		options.num_threads = 8;
		options.initial_trust_region_radius = 1e4;											 // 初始信任区域的大小。
		options.max_trust_region_radius = 1e20;													 // 信任区域半径最大值。
		options.min_trust_region_radius = 1e-32;												 // 信任区域的最小值。当信任区域小于此值，会停止优化。
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // 还有ceres::DOGLEG
		// options.dogleg_type = ceres::SUBSPACE_DOGLEG // 使用DOGLEG方法时指定
		// options.min_relative_decrease = ;//信任域步长(trust region step)相对减少的最小值。
		// options.min_lm_diagonal = ;//LEVENBERG MARQUARDT算法使用对角矩阵来规范（regularize）信任域步长。 这是该对角矩阵的值的下限
		// options.max_lm_diagonal = ;//LEVENBERG MARQUARDT算法使用对角矩阵来规范（regularize）信任域步长。这是该对角矩阵的值的上限。
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		fitness_ = summary.final_cost; // 最终函数值
		// 结果输出
		std::cout << "fitness : " << fitness_ << std::endl;
		std::cout << "temination_type:" << summary.termination_type << std::endl;					 // 0:CONVERGENCE, 1:NO_CONVERGENCE, 3:USER_SUCCESS
		std::cout << "num_successful_steps:" << summary.num_successful_steps << std::endl; // 最终被接受的迭代次数
		std::cout << "*******************************" << std::endl;
		if (summary.termination_type == 1)
		{
			converged_ = false;
			return false;
		}
		else
			converged_ = true;
		// std::cout << summary.BriefReport() << std::endl;
		// 优化结果
		transform_.setIdentity();
		transform_.rotate(q);
		transform_.pretranslate(t);
		return true;
	}
	inline double getFitness()
	{
		return (fitness_);
	}
	inline Eigen::Affine3d getTransform()
	{
		return (transform_);
	}
	inline bool hasConverged()
	{
		return (converged_);
	}
};
