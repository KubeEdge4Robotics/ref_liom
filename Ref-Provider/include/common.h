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
#include <queue>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <limits>
// Eigen头文件
#include <Eigen/Eigen>
// ros头文件
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
// TimeSyncronizer headers
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
// pcl 头文件
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>

#include "tic_toc.h"

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
