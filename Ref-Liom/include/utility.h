#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
// ros msgs headers
#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
// TimeSyncronizer headers
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
// tf headers
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
// pcl headers
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
// std headers
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <unordered_set>

// gtsam headers
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
using namespace std;

typedef pcl::PointXYZI PointType;
/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
 */
struct PointTypePose
{
	double time;
	gtsam::Pose3 pose;
	uint32_t index;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;									// enforce SSE padding for correct memory alignment

enum class SensorType
{
	VELODYNE,
	OUSTER,
	LIVOX
};

class ParamServer
{
public:
	ros::NodeHandle nh;
	// Topics
	string pointCloudTopic;
	string odomTopic;
	string gpsTopic;
	string imuTopic;
	string priorTopic;
	string loopTopic;
	// Frames
	string lidarFrame;
	string bodyFrame;
	string odometryFrame;
	string mapFrame;
	// CPU Params
	int numberOfCores;
	double mappingProcessInterval;
	// Lidar Sensor Configuration
	SensorType sensor;
	// Surrounding map
	float surroundingkeyframeAddingDistThreshold;
	float surroundingkeyframeAddingAngleThreshold;
	float surroundingKeyframeDensity;
	float surroundingKeyframeSearchRadius;
	// Save pcd
	bool savePCD;
	string savePCDDirectory;
	// Loop closure
	bool loopClosureEnableFlag;
	float loopClosureFrequency;
	int surroundingKeyframeSize;
	float historyKeyframeSearchRadius;
	float historyKeyframeSearchTimeDiff;
	int historyKeyframeSearchNum;
	float historyKeyframeFitnessScore;
	// GPS Settings
	bool useGpsElevation;
	float gpsCovThreshold;
	float poseCovThreshold;
	// voxel filter paprams
	float mappingLeafSize;
	// global map visualization radius
	float globalMapVisualizationSearchRadius;
	float globalMapVisualizationPoseDensity;
	float globalMapVisualizationLeafSize;

	int submapFrameNum = 5; // 一个子图包含的关键帧数量

	ParamServer()
	{
		nh.param<std::string>("ref_liom/pointCloudTopic", pointCloudTopic, "points_raw");
		nh.param<std::string>("ref_liom/imuTopic", imuTopic, "imu_correct");
		nh.param<std::string>("ref_liom/odomTopic", odomTopic, "odometry/imu");
		nh.param<std::string>("ref_liom/gpsTopic", gpsTopic, "odometry/gps");
		nh.param<std::string>("ref_liom/loopTopic", loopTopic, "loop_closure_detection");
		nh.param<std::string>("ref_liom/priorTopic", priorTopic, "prior_constraints");

		nh.param<std::string>("ref_liom/bodyFrame", bodyFrame, "base_link");
		nh.param<std::string>("ref_liom/odomFrame", odometryFrame, "odom");
		nh.param<std::string>("ref_liom/mapFrame", mapFrame, "map");

		nh.param<bool>("ref_liom/useGpsElevation", useGpsElevation, false);
		nh.param<float>("ref_liom/gpsCovThreshold", gpsCovThreshold, 2.0);
		nh.param<float>("ref_liom/poseCovThreshold", poseCovThreshold, 25.0);

		nh.param<bool>("ref_liom/savePCD", savePCD, false);
		nh.param<std::string>("ref_liom/savePCDDirectory", savePCDDirectory, "~/ref_liom_outputbao");

		std::string sensorStr;
		nh.param<std::string>("ref_liom/sensor", sensorStr, "");
		if (sensorStr == "ouster")
		{
			sensor = SensorType::OUSTER;
		}
		else if (sensorStr == "livox")
		{
			sensor = SensorType::LIVOX;
		}
		else
		{
			sensor = SensorType::VELODYNE;
		}

		nh.param<float>("ref_liom/mappingLeafSize", mappingLeafSize, 0.2);

		nh.param<int>("ref_liom/numberOfCores", numberOfCores, 2);
		nh.param<double>("ref_liom/mappingProcessInterval", mappingProcessInterval, 0.15);

		nh.param<float>("ref_liom/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
		nh.param<float>("ref_liom/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
		nh.param<float>("ref_liom/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
		nh.param<float>("ref_liom/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

		nh.param<bool>("ref_liom/loopClosureEnableFlag", loopClosureEnableFlag, false);
		nh.param<float>("ref_liom/loopClosureFrequency", loopClosureFrequency, 1.0);
		nh.param<int>("ref_liom/surroundingKeyframeSize", surroundingKeyframeSize, 50);
		nh.param<float>("ref_liom/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
		nh.param<float>("ref_liom/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
		nh.param<int>("ref_liom/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
		nh.param<float>("ref_liom/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

		nh.param<float>("ref_liom/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
		nh.param<float>("ref_liom/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
		nh.param<float>("ref_liom/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);
		nh.param<int>("ref_liom/submapFrameNum", submapFrameNum, 5);

		usleep(100);
	}
};

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

template <typename T>
double ROS_TIME(T msg)
{
	return msg->header.stamp.toSec();
}

template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
	*angular_x = thisImuMsg->angular_velocity.x;
	*angular_y = thisImuMsg->angular_velocity.y;
	*angular_z = thisImuMsg->angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
	*acc_x = thisImuMsg->linear_acceleration.x;
	*acc_y = thisImuMsg->linear_acceleration.y;
	*acc_z = thisImuMsg->linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
	double imuRoll, imuPitch, imuYaw;
	tf::Quaternion orientation;
	tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
	tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

	*rosRoll = imuRoll;
	*rosPitch = imuPitch;
	*rosYaw = imuYaw;
}

float pointDistance(PointType p)
{
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float pointDistance(PointType p1, PointType p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

/*
 * convert Pose formats
 */
inline gtsam::Pose3 trans2gtsamPose(const float transformIn[])
{
	return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
											gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

inline Eigen::Affine3f PointTypePose2Affine3f(const PointTypePose &thisPoint)
{
	return Eigen::Affine3f(thisPoint.pose.matrix().cast<float>());
}
inline Eigen::Affine3d PointTypePose2Affine3d(const PointTypePose &thisPoint)
{
	return Eigen::Affine3d(thisPoint.pose.matrix());
}

inline Eigen::Affine3f trans2Affine3f(const float transformIn[])
{
	return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
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
inline gtsam::Pose3 poseMsg2gtsamPose(const geometry_msgs::Pose &msg)
{
	return gtsam::Pose3(
			gtsam::Rot3::Quaternion(
					msg.orientation.w,
					msg.orientation.x,
					msg.orientation.y,
					msg.orientation.z),
			gtsam::Point3(msg.position.x, msg.position.y, msg.position.z));
}
inline Eigen::Affine3f gtsamPose2Affine3f(const gtsam::Pose3 &pose3)
{
	return Eigen::Affine3f(pose3.matrix().cast<float>());
}
inline gtsam::Pose3 Affine3f2gtsamPose(const Eigen::Affine3f &aff)
{
	return gtsam::Pose3(aff.matrix().cast<double>());
}
inline gtsam::Pose3 Affine3d2gtsamPose(const Eigen::Affine3d &aff)
{
	return gtsam::Pose3(aff.matrix());
}
inline geometry_msgs::Pose gtsamPose2poseMsg(const gtsam::Pose3 &pose3)
{
	geometry_msgs::Pose msg;
	gtsam::Quaternion q = pose3.rotation().toQuaternion();
	msg.orientation.x = q.x();
	msg.orientation.y = q.y();
	msg.orientation.z = q.z();
	msg.orientation.w = q.w();
	msg.position.x = pose3.translation().x();
	msg.position.y = pose3.translation().y();
	msg.position.z = pose3.translation().z();
	return msg;
}
#endif
