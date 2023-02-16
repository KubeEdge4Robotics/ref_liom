#include "utility.h"
#include "ref_liom/save_map.h"

#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class Submap
{ // 子地图信息
public:
	ros::Time time;
	std::vector<uint32_t> keyCloudIndices; // 子图点云:包含原始激光帧的编号
	gtsam::Pose3 pose;										 // 子图位姿
	// 构造函数
	Submap() : time(0), pose(gtsam::Pose3::Identity()){};
	Submap(double msg_time, const gtsam::Pose3 &p) : time(msg_time), pose(p){};
	Submap(ros::Time msg_stamp, const gtsam::Pose3 &p) : time(msg_stamp), pose(p){};
};

class FactorsWithValues
{
public:
	// A shared_ptr to this class
	typedef FactorsWithValues This;
	typedef std::shared_ptr<This> shared_ptr;

	gtsam::Values values;
	std::vector<gtsam::NonlinearFactor::shared_ptr> factors;
	bool inited;
	FactorsWithValues() : inited(false){};
	// 析构函数
	virtual ~FactorsWithValues()
	{
		factors.clear();
	}
};

using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose
using symbol_shorthand::M; // Model Pose3
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
bool holdForLoop = false;	 // 回环约束条件变量

class mapOptimization : public ParamServer
{
public:
	// gtsam
	NonlinearFactorGraph gtSAMgraph;
	Values initialEstimate;
	ISAM2 *isam;
	Values isamCurrentEstimate;
	Eigen::MatrixXd poseCovariance;

	ros::Publisher pubGlobalMap;
	ros::Publisher pubAftPgoOdom;
	ros::Publisher pubKeyPoses;
	ros::Publisher pubAftPgoPath;
	ros::Publisher pubSubmap;

	ros::Publisher pubHistoryKeyFrames;
	ros::Publisher pubIcpKeyFrames;
	ros::Publisher pubLoopConstraintEdge;

	ros::Subscriber subGPS;
	ros::Subscriber subLoop;
	ros::Subscriber subPrior;

	ros::ServiceServer srvSaveMap;

	std::deque<nav_msgs::Odometry> gpsQueue;

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

	pcl::VoxelGrid<PointType> downSizeFilterMap;								 // for map downsampling
	pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

	ros::Time timeLaserInfoStamp;
	double timeLaserInfoCur;

	std::mutex mtxKeyPoses;				// keyPoses互斥
	std::mutex mtxLoopExt;				// 外部回环信息互斥
	std::mutex mtxLoopConstraint; // 回环约束互斥
	std::condition_variable sigLoopConstraint;

	bool isDegenerate = false;

	bool aLoopIsClosed = false;
	map<int, int> loopIndexContainer; // from new to old
	vector<pair<int, int>> loopIndexQueue;
	vector<gtsam::Pose3> loopPoseQueue;
	vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
	deque<std_msgs::Float64MultiArray> loopInfoVec;

	/* 修改的变量 */
	std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> subOdom;
	std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> subCloud;
	typedef message_filters::sync_policies::ExactTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> SyncPolicy;
	std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> odom_cloud_sync;
	pcl::PointCloud<PointType>::Ptr keyPoses3D; // 历史关键帧位置
	std::vector<PointTypePose> keyPoses6D;			// 历史关键帧位姿
	pcl::PointCloud<PointType>::Ptr copy_keyPoses3D;
	std::vector<PointTypePose> copy_keyPoses6D;
	std::deque<pcl::PointCloud<PointType>::Ptr> keyClouds;							 // 索引是关键帧id，值存放的是body系点云
	std::unordered_map<int, std::shared_ptr<Submap>> hash_submap;				 // 索引是子图id，存放子图的信息
	nav_msgs::Path::Ptr globalPath;																			 // 全局校正轨迹
	nav_msgs::Path::Ptr modelPoses;																			 // 优化后模型位姿
	gtsam::Pose3 lastOdomPose;																					 // 上一里程计位姿
	gtsam::Pose3 thisOdomPose;																					 // 最新里程计位姿
	gtsam::Pose3 latestPose6D;																					 // 最新图优化位姿
	ros::Publisher pubModelPoses;																				 // 优化模型位姿发布者
	std::deque<geometry_msgs::PoseWithCovariance> refQueue;							 // 先验信息msg队列
	std::unordered_map<int, FactorsWithValues::shared_ptr> modelInfoSet; // 先验模型的信息
	std::unordered_set<int> modelReady;
	int minModelFactorNum = 3;
	mapOptimization()
	{
		ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.1;
		parameters.relinearizeSkip = 1;
		isam = new ISAM2(parameters);
		globalPath = boost::make_shared<nav_msgs::Path>();
		globalPath->header.frame_id = mapFrame;
		modelPoses = boost::make_shared<nav_msgs::Path>();
		modelPoses->header.frame_id = mapFrame;

		pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/ref_liom/key_poses", 1000);	 // 关键帧位姿点云
		pubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("/ref_liom/global_map", 1000); // 发布优化后的全局地图
		pubAftPgoOdom = nh.advertise<nav_msgs::Odometry>("/ref_liom/aftPgo_odom", 1000);		 // 发布激光里程计
		pubAftPgoPath = nh.advertise<nav_msgs::Path>("/ref_liom/aftPgo_Path", 1000);				 // 发布优化后的全局轨迹
		pubSubmap = nh.advertise<sensor_msgs::PointCloud2>("/ref_liom/submap", 1000);				 // 发布子图点云
		pubModelPoses = nh.advertise<nav_msgs::Path>("/ref_liom/model_poses", 1000);				 // 先验模型优化位姿

		// 订阅最新关键帧信息(里程计位姿、点云)
		subOdom = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(nh, odomTopic, 1000, ros::TransportHints().tcpNoDelay());
		subCloud = std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(nh, pointCloudTopic, 1000, ros::TransportHints().tcpNoDelay());
		odom_cloud_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(1000), *subOdom, *subCloud); // 1000是消息队列长度
		odom_cloud_sync->registerCallback(boost::bind(&mapOptimization::syncedCallback, this, _1, _2));
		// 订阅先验模型约束
		subPrior = nh.subscribe<geometry_msgs::PoseWithCovariance>(priorTopic, 1000, &mapOptimization::refModelHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅GPS里程计
		subGPS = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 1000, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供
		subLoop = nh.subscribe<std_msgs::Float64MultiArray>(loopTopic, 1000, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
		// 发布地图保存服务
		srvSaveMap = nh.advertiseService("ref_liom/save_map", &mapOptimization::saveMapService, this);
		// 发布闭环匹配关键帧局部map
		pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("ref_liom/icp_loop_closure_history_cloud", 1);
		// 发布当前关键帧经过闭环优化后的之后的特征点云
		pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("ref_liom/icp_loop_closure_corrected_cloud", 1);
		// 发布闭环边，rviz表现为闭环帧之间的连线
		pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/ref_liom/loop_closure_constraints", 1);

		downSizeFilterMap.setLeafSize(mappingLeafSize, mappingLeafSize, mappingLeafSize);
		downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

		allocateMemory();
	}

	virtual ~mapOptimization()
	{
		delete[] isam;
	}

	void allocateMemory()
	{
		keyPoses3D.reset(new pcl::PointCloud<PointType>());
		copy_keyPoses3D.reset(new pcl::PointCloud<PointType>());

		kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
		kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
	}
	/**
	 * 订阅当前激光帧点云信息，来自featureExtraction
	 * 1、当前帧位姿初始化
	 *   1) 如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
	 *   2) 后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
	 * 2、提取局部角点、平面点云集合，加入局部map
	 *   1) 对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
	 *   2) 对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
	 * 3、当前激光帧角点、平面点集合降采样
	 * 4、scan-to-map优化当前帧位姿
	 *   (1) 要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
	 *   (2) 迭代30次（上限）优化
	 *      1) 当前激光帧角点寻找局部map匹配点
	 *          a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
	 *          b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
	 *      2) 当前激光帧平面点寻找局部map匹配点
	 *          a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
	 *          b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
	 *      3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
	 *      4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存latestPose6D
	 *   (3)用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
	 * 5、设置当前帧为关键帧并执行因子图优化
	 *   1) 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
	 *   2) 添加激光里程计因子、GPS因子、闭环因子
	 *   3) 执行因子图优化
	 *   4) 得到当前帧优化后位姿，位姿协方差
	 *   5) 添加keyPoses3D，keyPoses6D，更新latestPose6D，添加当前关键帧的角点、平面点集合
	 * 6、更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
	 * 7、发布激光里程计
	 * 8、发布里程计、点云、轨迹
	 */

	void syncedCallback(const nav_msgs::Odometry::ConstPtr &msgOdom, const sensor_msgs::PointCloud2::ConstPtr &msgCloud)
	{
		// extract time stamp 当前激光帧时间戳
		timeLaserInfoStamp = msgOdom->header.stamp;
		timeLaserInfoCur = msgOdom->header.stamp.toSec();

		thisOdomPose = poseMsg2gtsamPose(msgOdom->pose.pose); // 最新里程计位姿
		// 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
		if (saveFrame() == false)
			return;
		// keyClouds加入关键帧点云
		pcl::PointCloud<PointType>::Ptr cloud_ptr(new pcl::PointCloud<PointType>);
		pcl::fromROSMsg(*msgCloud, *cloud_ptr);
		keyClouds.push_back(cloud_ptr);
		// 设置当前帧为关键帧并执行因子图优化
		// 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
		// 2、添加激光里程计因子、GPS因子、闭环因子
		// 3、执行因子图优化
		// 4、得到当前帧优化后位姿，位姿协方差
		// 5、添加keyPoses3D，keyPoses6D，更新latestPose6D，添加当前关键帧的角点、平面点集合
		saveKeyFramesAndFactor();
		// 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
		correctPoses();
		// 发布里程计、点云、轨迹
		// 1、发布历史关键帧位姿集合
		// 2、发布局部map的降采样平面点集合
		// 3、发布历史帧（累加的）的角点、平面点降采样集合
		// 4、发布里程计轨迹
		publishFrames();

		/* std::unique_lock<std::mutex> ulock(mtxKeyPoses);
		// mapping执行频率控制
		static double timeLastProcessing = -1;
		if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
		{
				timeLastProcessing = timeLaserInfoCur;
				// 当前帧位姿初始化
				// 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
				// 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
				// updateInitialGuess();
				// 提取局部角点、平面点云集合，加入局部map
				// 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
				// 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
				extractSurroundingKeyFrames();
				// 当前激光帧角点、平面点集合降采样
				downsampleCurrentScan();
				// scan-to-map优化当前帧位姿
				// 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
				// 2、迭代30次（上限）优化
				//    1) 当前激光帧角点寻找局部map匹配点
				//       a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
				//       b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
				//    2) 当前激光帧平面点寻找局部map匹配点
				//       a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
				//       b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
				//    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
				//    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存latestPose6D
				// 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
				// scan2MapOptimization();
		} */
	}

	void gpsHandler(const nav_msgs::Odometry::ConstPtr &gpsMsg)
	{
		gpsQueue.push_back(*gpsMsg);
	}

	pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, const gtsam::Pose3 &transformIn)
	{
		pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

		int cloudSize = cloudIn->size();
		cloudOut->resize(cloudSize);

		Eigen::Affine3f transCur = gtsamPose2Affine3f(transformIn);

#pragma omp parallel for num_threads(numberOfCores)
		for (int i = 0; i < cloudSize; ++i)
		{
			const auto &pointFrom = cloudIn->points[i];
			cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
			cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
			cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
			cloudOut->points[i].intensity = pointFrom.intensity;
		}
		return cloudOut;
	}

	bool saveMapService(ref_liom::save_mapRequest &req, ref_liom::save_mapResponse &res)
	{
		string saveMapDirectory;
		cout << "****************************************************" << endl;
		cout << "Saving map to pcd files ..." << endl;
		if (req.destination.empty())
			saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
		else
			saveMapDirectory = std::getenv("HOME") + req.destination;
		cout << "Save destination: " << saveMapDirectory << endl;
		// create directory and remove old files;
		int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
		unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
		// save key frame transformations
		pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *keyPoses3D);
		// pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *keyPoses6D);
		// extract global point cloud map
		pcl::PointCloud<PointType>::Ptr globalCloud(new pcl::PointCloud<PointType>());
		for (size_t i = 0; i < keyPoses3D->size(); i++)
		{
			*globalCloud += *transformPointCloud(keyClouds[i], keyPoses6D[i].pose);
			cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << keyPoses6D.size() << " ...";
		}
		// downsample global map
		downSizeFilterMap.setInputCloud(globalCloud);
		downSizeFilterMap.filter(*globalCloud);
		int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalCloud);
		res.success = ret == 0;
		cout << "****************************************************" << endl;
		cout << "Saving map to pcd files completed\n"
				 << endl;

		return true;
	}

	void visualizeGlobalMapThread()
	{
		ros::Rate rate(0.2);
		while (ros::ok())
		{
			rate.sleep();
			publishGlobalMap();
		}

		if (savePCD == false)
			return;

		ref_liom::save_mapRequest req;
		ref_liom::save_mapResponse res;

		if (!saveMapService(req, res))
		{
			cout << "Fail to save map" << endl;
		}
	}

	void publishGlobalMap()
	{
		if (pubGlobalMap.getNumSubscribers() == 0)
			return;

		if (keyPoses3D->points.empty() == true)
			return;

		pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
		;
		pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

		// kd-tree to find near key frames to visualize
		std::vector<int> pointSearchIndGlobalMap;
		std::vector<float> pointSearchSqDisGlobalMap;
		// search near key frames to visualize
		{
			std::unique_lock<std::mutex> ulock(mtxKeyPoses);
			kdtreeGlobalMap->setInputCloud(keyPoses3D);
			kdtreeGlobalMap->radiusSearch(keyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
		}

		for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
			globalMapKeyPoses->push_back(keyPoses3D->points[pointSearchIndGlobalMap[i]]);
		// downsample near selected key frames
		pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;																																														// for global map visualization
		downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
		downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
		downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
		for (auto &pt : globalMapKeyPosesDS->points)
		{
			kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
			pt.intensity = keyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
		}

		// extract visualized and downsampled key frames
		for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
		{
			if (pointDistance(globalMapKeyPosesDS->points[i], keyPoses3D->back()) > globalMapVisualizationSearchRadius)
				continue;
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
			*globalMapKeyFrames += *transformPointCloud(keyClouds[thisKeyInd], keyPoses6D[thisKeyInd].pose);
		}
		// downsample visualized points
		pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;																																										// for global map visualization
		downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
		downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
		downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
		publishCloud(pubGlobalMap, globalMapKeyFramesDS, timeLaserInfoStamp, mapFrame);
	}
	/**
	 * 闭环线程
	 * 1、闭环scan-to-map，icp优化位姿
	 *   1) 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
	 *   2) 提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
	 *   3) 执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
	 * 2、rviz展示闭环边
	 */
	void loopClosureThread()
	{
		if (loopClosureEnableFlag == false)
			return;

		ros::Rate rate(loopClosureFrequency);
		while (ros::ok())
		{
			rate.sleep();
			// 闭环scan-to-map，icp优化位姿
			// 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
			// 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
			// 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
			// 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
			performLoopClosure();
			// rviz展示闭环边
			visualizeLoopClosure();
		}
	}

	void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr &loopMsg)
	{
		std::unique_lock<std::mutex> lock(mtxLoopExt);
		if (loopMsg->data.size() != 2)
			return;

		loopInfoVec.push_back(*loopMsg);

		while (loopInfoVec.size() > 5)
			loopInfoVec.pop_front();
	}
	/**
	 * 闭环scan-to-map，icp优化位姿
	 * 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
	 * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
	 * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
	 * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
	 */
	void performLoopClosure()
	{
		// 如果关键帧集合为空，则返回
		if (keyPoses3D->points.empty() == true)
			return;

		{ // 复制关键帧信息
			std::unique_lock<std::mutex> ulock(mtxKeyPoses);
			*copy_keyPoses3D = *keyPoses3D;
			copy_keyPoses6D = keyPoses6D;
		}

		// find keys
		int loopKeyCur; // 当前关键帧索引
		int loopKeyPre; // 候选闭环匹配帧索引
		// not-used
		if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
			// 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
			if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
				return;

		// extract cloud
		pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
		{
			// 提取当前关键帧点云集合，降采样
			loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
			// 提取闭环匹配关键帧前后相邻若干帧的关键帧点云集合，降采样
			loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
			// 如果特征点较少，则返回
			if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
				return;
			// 发布闭环匹配关键帧局部map
			if (pubHistoryKeyFrames.getNumSubscribers() != 0)
				publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, mapFrame);
		}

		// ICP Settings
		static pcl::IterativeClosestPoint<PointType, PointType> icp;
		icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius);
		icp.setMaximumIterations(100);
		icp.setTransformationEpsilon(1e-6);
		icp.setEuclideanFitnessEpsilon(1e-6);
		icp.setRANSACIterations(0);

		// Align clouds
		icp.setInputSource(cureKeyframeCloud);
		icp.setInputTarget(prevKeyframeCloud);
		pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
		icp.align(*unused_result);
		// 未收敛，或者匹配不够好
		if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
			return;

		// publish corrected cloud 发布当前关键帧经过闭环优化后的特征点云
		if (pubIcpKeyFrames.getNumSubscribers() != 0)
		{
			pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
			pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
			publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, mapFrame);
		}

		// Get pose transformation
		gtsam::Pose3 poseBetween(icp.getFinalTransformation().cast<double>());
		gtsam::Vector Vector6(6);
		float noiseScore = icp.getFitnessScore();
		Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
		noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
		// Add pose constraint 添加闭环因子需要的数据
		{ // 加锁
			std::unique_lock<std::mutex> ulock(mtxLoopConstraint);
			sigLoopConstraint.wait(ulock, []
														 { return !holdForLoop; });
			holdForLoop = true;
		}
		loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
		loopPoseQueue.push_back(poseBetween);
		loopNoiseQueue.push_back(constraintNoise);
		// add loop constriant
		loopIndexContainer[loopKeyCur] = loopKeyPre;
		{ // 解锁
			std::unique_lock<std::mutex> ulock(mtxLoopConstraint);
			holdForLoop = false;
			sigLoopConstraint.notify_all();
		}
	}
	/**
	 * 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
	 */
	bool detectLoopClosureDistance(int *latestID, int *closestID)
	{
		// 当前关键帧
		int loopKeyCur = copy_keyPoses3D->size() - 1;
		int loopKeyPre = -1;

		// check loop constraint added before
		auto it = loopIndexContainer.find(loopKeyCur);
		// 当前关键帧已经添加过闭环对应关系，不再继续添加
		if (it != loopIndexContainer.end())
			return false;

		// find the closest history key frame 在历史关键帧中寻找与当前关键帧距离最近的关键帧集合
		std::vector<int> pointSearchIndLoop;
		std::vector<float> pointSearchSqDisLoop;
		kdtreeHistoryKeyPoses->setInputCloud(copy_keyPoses3D);
		kdtreeHistoryKeyPoses->radiusSearch(copy_keyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
		// 在候选关键帧集合中，找到与当前关键帧时间相隔较远的帧，设为候选匹配帧
		for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
		{
			int id = pointSearchIndLoop[i];
			if (abs(copy_keyPoses6D[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
			{
				loopKeyPre = id;
				break;
			}
		}

		if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
			return false;

		*latestID = loopKeyCur;
		*closestID = loopKeyPre;

		return true;
	}

	bool detectLoopClosureExternal(int *latestID, int *closestID)
	{
		// this function is not used yet, please ignore it
		int loopKeyCur = -1;
		int loopKeyPre = -1;

		std::unique_lock<std::mutex> lock(mtxLoopExt);
		if (loopInfoVec.empty())
			return false;

		double loopTimeCur = loopInfoVec.front().data[0];
		double loopTimePre = loopInfoVec.front().data[1];
		loopInfoVec.pop_front();

		if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
			return false;

		int cloudSize = copy_keyPoses6D.size();
		if (cloudSize < 2)
			return false;

		// latest key
		loopKeyCur = cloudSize - 1;
		for (int i = cloudSize - 1; i >= 0; --i)
		{
			if (copy_keyPoses6D[i].time >= loopTimeCur)
				loopKeyCur = round(copy_keyPoses6D[i].index);
			else
				break;
		}

		// previous key
		loopKeyPre = 0;
		for (int i = 0; i < cloudSize; ++i)
		{
			if (copy_keyPoses6D[i].time <= loopTimePre)
				loopKeyPre = round(copy_keyPoses6D[i].index);
			else
				break;
		}

		if (loopKeyCur == loopKeyPre)
			return false;

		auto it = loopIndexContainer.find(loopKeyCur);
		if (it != loopIndexContainer.end())
			return false;

		*latestID = loopKeyCur;
		*closestID = loopKeyPre;

		return true;
	}

	void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
	{
		// extract near keyframes
		nearKeyframes->clear();
		int cloudSize = copy_keyPoses6D.size();
		for (int i = -searchNum; i <= searchNum; ++i)
		{
			int keyNear = key + i;
			if (keyNear < 0 || keyNear >= cloudSize)
				continue;
			*nearKeyframes += *transformPointCloud(keyClouds[keyNear], copy_keyPoses6D[keyNear].pose);
		}

		if (nearKeyframes->empty())
			return;

		// downsample near keyframes
		downSizeFilterMap.setInputCloud(nearKeyframes);
		downSizeFilterMap.filter(*nearKeyframes);
	}

	void visualizeLoopClosure()
	{
		if (loopIndexContainer.empty())
			return;

		visualization_msgs::MarkerArray markerArray;
		// loop nodes
		visualization_msgs::Marker markerNode;
		markerNode.header.frame_id = mapFrame;
		markerNode.header.stamp = timeLaserInfoStamp;
		markerNode.action = visualization_msgs::Marker::ADD;
		markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
		markerNode.ns = "loop_nodes";
		markerNode.id = 0;
		markerNode.pose.orientation.w = 1;
		markerNode.scale.x = 0.3;
		markerNode.scale.y = 0.3;
		markerNode.scale.z = 0.3;
		markerNode.color.r = 0;
		markerNode.color.g = 0.8;
		markerNode.color.b = 1;
		markerNode.color.a = 1;
		// loop edges
		visualization_msgs::Marker markerEdge;
		markerEdge.header.frame_id = mapFrame;
		markerEdge.header.stamp = timeLaserInfoStamp;
		markerEdge.action = visualization_msgs::Marker::ADD;
		markerEdge.type = visualization_msgs::Marker::LINE_LIST;
		markerEdge.ns = "loop_edges";
		markerEdge.id = 1;
		markerEdge.pose.orientation.w = 1;
		markerEdge.scale.x = 0.1;
		markerEdge.color.r = 0.9;
		markerEdge.color.g = 0.9;
		markerEdge.color.b = 0;
		markerEdge.color.a = 1;

		for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
		{
			int key_cur = it->first;
			int key_pre = it->second;
			geometry_msgs::Point p;
			p.x = copy_keyPoses6D[key_cur].pose.translation().x();
			p.y = copy_keyPoses6D[key_cur].pose.translation().y();
			p.z = copy_keyPoses6D[key_cur].pose.translation().z();
			markerNode.points.push_back(p);
			markerEdge.points.push_back(p);
			p.x = copy_keyPoses6D[key_pre].pose.translation().x();
			p.y = copy_keyPoses6D[key_pre].pose.translation().y();
			p.z = copy_keyPoses6D[key_pre].pose.translation().z();
			markerNode.points.push_back(p);
			markerEdge.points.push_back(p);
		}

		markerArray.markers.push_back(markerNode);
		markerArray.markers.push_back(markerEdge);
		pubLoopConstraintEdge.publish(markerArray);
	}

	// 根据位姿增量判断当前帧是否为关键帧
	bool saveFrame()
	{
		if (keyPoses3D->points.empty())
			return true;

		if (sensor == SensorType::LIVOX)
		{
			if (timeLaserInfoCur - keyPoses6D.back().time > 1.0)
				return true;
		}
		Eigen::Vector3d rotBetween = thisOdomPose.rotation().localCoordinates(keyPoses6D.back().pose.rotation());
		gtsam::Point3 transBetween = thisOdomPose.translation() - keyPoses6D.back().pose.translation();
		if (rotBetween.norm() < surroundingkeyframeAddingAngleThreshold &&
				transBetween.norm() < surroundingkeyframeAddingDistThreshold)
			return false;

		return true;
	}
	// 加入里程计因子
	void addOdomFactor()
	{
		gtsam::Pose3 poseTo = thisOdomPose;
		if (keyPoses3D->points.empty()) // 还没有历史关键帧，第一个位姿
		{
			noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad*rad, meter*meter
			gtSAMgraph.add(PriorFactor<Pose3>(X(0), poseTo, priorNoise));
			initialEstimate.insert(X(0), poseTo);
		}
		else
		{
			noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
			// gtsam::Pose3 poseFrom = keyPoses6D.back().pose;
			gtsam::Pose3 poseFrom = lastOdomPose;
			gtSAMgraph.add(BetweenFactor<Pose3>(X(keyPoses3D->size() - 1), X(keyPoses3D->size()), poseFrom.between(poseTo), odometryNoise));
			initialEstimate.insert(X(keyPoses3D->size()), keyPoses6D.back().pose * lastOdomPose.inverse() * thisOdomPose);
		}
		lastOdomPose = poseTo;
	}
	// 加入GPS因子
	void addGPSFactor()
	{
		if (gpsQueue.empty())
			return;

		// wait for system initialized and settles down
		if (keyPoses3D->points.empty())
			return;
		else
		{
			if (pointDistance(keyPoses3D->front(), keyPoses3D->back()) < 5.0)
				return;
		}

		// pose covariance small, no need to correct
		if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold)
			return;

		// last gps position
		static PointType lastGPSPoint;

		while (!gpsQueue.empty())
		{
			if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
			{
				// message too old
				gpsQueue.pop_front();
			}
			else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
			{
				// message too new
				break;
			}
			else
			{
				nav_msgs::Odometry thisGPS = gpsQueue.front();
				gpsQueue.pop_front();

				// GPS too noisy, skip
				float noise_x = thisGPS.pose.covariance[0];
				float noise_y = thisGPS.pose.covariance[7];
				float noise_z = thisGPS.pose.covariance[14];
				if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
					continue;

				float gps_x = thisGPS.pose.pose.position.x;
				float gps_y = thisGPS.pose.pose.position.y;
				float gps_z = thisGPS.pose.pose.position.z;
				if (!useGpsElevation)
				{
					gps_z = latestPose6D.translation().z();
					noise_z = 0.01;
				}

				// GPS not properly initialized (0,0,0)
				if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
					continue;

				// Add GPS every a few meters
				PointType curGPSPoint;
				curGPSPoint.x = gps_x;
				curGPSPoint.y = gps_y;
				curGPSPoint.z = gps_z;
				if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
					continue;
				else
					lastGPSPoint = curGPSPoint;

				gtsam::Vector Vector3(3);
				Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
				noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
				gtsam::GPSFactor gps_factor(X(keyPoses3D->size()), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
				gtSAMgraph.add(gps_factor);

				aLoopIsClosed = true;
				break;
			}
		}
	}
	// 加入闭环因子
	void addLoopFactor()
	{
		{ // 加锁
			std::unique_lock<std::mutex> ulock(mtxLoopConstraint);
			sigLoopConstraint.wait(ulock, []
														 { return !holdForLoop; });
			holdForLoop = true;
		}
		if (loopIndexQueue.empty())
		{
			{ // 解锁
				std::unique_lock<std::mutex> ulock(mtxLoopConstraint);
				holdForLoop = false;
				sigLoopConstraint.notify_all();
			}
			return;
		}

		for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
		{
			int indexFrom = loopIndexQueue[i].first;
			int indexTo = loopIndexQueue[i].second;
			gtsam::Pose3 poseBetween = loopPoseQueue[i];
			gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
			gtSAMgraph.add(BetweenFactor<Pose3>(X(indexFrom), X(indexTo), poseBetween, noiseBetween));
		}

		loopIndexQueue.clear();
		loopPoseQueue.clear();
		loopNoiseQueue.clear();
		aLoopIsClosed = true;
		{ // 解锁
			std::unique_lock<std::mutex> ulock(mtxLoopConstraint);
			holdForLoop = false;
			sigLoopConstraint.notify_all();
		}
	}
	// 处理先验信息的回调函数
	void refModelHandler(const geometry_msgs::PoseWithCovariance::ConstPtr &msg)
	{
		refQueue.push_back(*msg);
	}
	// 加入先验模型因子
	void addRefModelFactor()
	{
		while (!refQueue.empty())
		{
			geometry_msgs::PoseWithCovariance &refMsg = refQueue.front();
			gtsam::Pose3 poseBetween = poseMsg2gtsamPose(refMsg.pose);					// model->body
			uint32_t keyposeIdx = static_cast<uint32_t>(refMsg.covariance[0]); // 关键帧id
			int modelIdx = static_cast<int>(refMsg.covariance[1]);							// 模型id
			gtsam::Pose3 poseFrom = keyPoses6D[keyposeIdx].pose;								// 关键帧位姿
			gtsam::Pose3 poseTo = poseFrom * poseBetween;
			double var_trans = 1000 * refMsg.covariance[2];	 // 位移协方差
			double var_rot = 10000 * refMsg.covariance[3]; // 旋转协方差
			cout << "var_trans: " << var_trans << "var_rot:" << var_rot << endl;
			// If this is the first iteration, add a prior on the first pose to set the
			// coordinate frame and a prior on the first landmark to set the scale Also,
			// as iSAM solves incrementally, we must wait until each is observed at
			// least twice before adding it to iSAM.
			if (modelReady.count(modelIdx) == 0)
			{
				if (modelInfoSet.count(modelIdx) == 0)
				{
					modelInfoSet.emplace(modelIdx, std::make_shared<FactorsWithValues>());
					(modelInfoSet[modelIdx]->values).insert(M(modelIdx), poseTo);																																														 // 加入初值
					noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << Vector3::Constant(1e-2), Vector3::Constant(1e-2)).finished()); // rad*rad, meter*meter
					noiseModel::Diagonal::shared_ptr refNoise = noiseModel::Diagonal::Variances((Vector(6) << Vector3::Constant(var_trans), Vector3::Constant(var_rot)).finished());
					(modelInfoSet[modelIdx]->factors).push_back(gtsam::NonlinearFactor::shared_ptr(new PriorFactor<Pose3>(M(modelIdx), poseTo, priorNoise)));
					(modelInfoSet[modelIdx]->factors).push_back(BetweenFactor<Pose3>::shared_ptr(new BetweenFactor<Pose3>(X(keyposeIdx), M(modelIdx), poseBetween, refNoise)));
				}
				else
				{
					noiseModel::Diagonal::shared_ptr refNoise = noiseModel::Diagonal::Variances((Vector(6) << Vector3::Constant(var_trans), Vector3::Constant(var_rot)).finished());
					BetweenFactor<Pose3>::shared_ptr refFactor(new BetweenFactor<Pose3>(X(keyposeIdx), M(modelIdx), poseBetween, refNoise));
					(modelInfoSet[modelIdx]->factors).push_back(refFactor);// 加入模型因子
					if (modelInfoSet[modelIdx]->factors.size() > minModelFactorNum){
						initialEstimate.insert(modelInfoSet[modelIdx]->values);
						gtSAMgraph.add_factors(modelInfoSet[modelIdx]->factors);
						modelInfoSet.erase(modelIdx);//删除模型因子与初值信息
						modelReady.emplace(modelIdx);
						aLoopIsClosed = true;
					}			
				}
			}
			else
			{
				noiseModel::Diagonal::shared_ptr refNoise = noiseModel::Diagonal::Variances((Vector(6) << Vector3::Constant(var_trans), Vector3::Constant(var_rot)).finished());
				BetweenFactor<Pose3>::shared_ptr refFactor(new BetweenFactor<Pose3>(X(keyposeIdx), M(modelIdx), poseBetween, refNoise));
				gtSAMgraph.add(refFactor); // 加入模型因子
				aLoopIsClosed = true;
			}

			refQueue.pop_front();
		}
	}
	/**
	 * 设置当前帧为关键帧并执行因子图优化
	 * 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
	 * 2、添加激光里程计因子、GPS因子、闭环因子
	 * 3、执行因子图优化
	 * 4、得到当前帧优化后位姿，位姿协方差
	 * 5、添加keyPoses3D，keyPoses6D，更新latestPose6D，添加当前关键帧的角点、平面点集合
	 */
	void saveKeyFramesAndFactor()
	{
		// odom factor
		addOdomFactor();
		// gps factor
		addGPSFactor();
		// loop factor
		addLoopFactor();
		// reference model factor
		addRefModelFactor();
		// cout << "****************************************************" << endl;
		// gtSAMgraph.print("GTSAM Graph:\n");

		// update iSAM
		isam->update(gtSAMgraph, initialEstimate);
		isam->update();

		if (aLoopIsClosed == true)
		{
			isam->update();
			isam->update();
			isam->update();
			isam->update();
			isam->update();
		}
		// update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
		gtSAMgraph.resize(0);
		initialEstimate.clear();

		// save key poses
		PointType thisPose3D;
		PointTypePose thisPose6D;
		// 优化结果
		isamCurrentEstimate = isam->calculateEstimate();
		// 当前帧位姿结果
		latestPose6D = isamCurrentEstimate.at<Pose3>(X(keyPoses3D->size()));
		// cout << "****************************************************" << endl;
		// isamCurrentEstimate.print("Current estimate: ");

		// keyPose3D加入当前关键帧位置
		thisPose3D.x = latestPose6D.translation().x();
		thisPose3D.y = latestPose6D.translation().y();
		thisPose3D.z = latestPose6D.translation().z();
		thisPose3D.intensity = keyPoses3D->size(); // this can be used as index
		keyPoses3D->push_back(thisPose3D);
		// keyPoses6D加入当前帧位姿
		thisPose6D.pose = latestPose6D;
		thisPose6D.time = timeLaserInfoCur;
		thisPose6D.index = thisPose3D.intensity; // this can be used as index
		keyPoses6D.push_back(thisPose6D);

		static int subframe_idx = 0;					// 子图中的关键帧索引
		static int submap_idx = 0;						// 子图索引
		static uint32_t submap_start_idx = 0; // 子图第一个关键帧索引
		static uint32_t submap_end_idx = 0;		// 子图最后一个关键帧索引
		subframe_idx++;
		// 积累n个关键帧作为一个子图
		if (subframe_idx == 1)
		{
			// if(subframe_idx < submap_frame_num){
			submap_start_idx = keyPoses3D->size() - 1;
			hash_submap[submap_idx] = std::make_shared<Submap>();
			hash_submap[submap_idx]->keyCloudIndices.push_back(keyPoses3D->size() - 1); // 累积点云索引
		}
		else if (subframe_idx == submapFrameNum)
		{
			submap_end_idx = keyPoses3D->size() - 1;
			hash_submap[submap_idx]->pose = thisPose6D.pose;														// 子图包含的最后一帧关键帧位姿作为子图位姿
			hash_submap[submap_idx]->keyCloudIndices.push_back(keyPoses3D->size() - 1); // 累积点云索引
			pcl::PointCloud<PointType>::Ptr submap_cloud(new pcl::PointCloud<PointType>);
			// pcl::PointCloud<PointType>::Ptr transformed_cloud(new pcl::PointCloud<PointType>);
			for (size_t i = submap_start_idx; i <= submap_end_idx; i++)
			{
				// for(auto i : hash_submap[submap_idx]->keyCloudIndices){
				// Eigen::Matrix4f transform = (keyPoses6D[submap_end_idx].pose.inverse() * keyPoses6D[i].pose).matrix();
				if (i >= keyClouds.size())
					cout << "error" << endl;
				if (keyClouds[i]->points.size() == 0)
					cout << "error2" << endl;
				// 将关键帧点云都投影到最后一帧的坐标系中
				*submap_cloud += *transformPointCloud(keyClouds[i], keyPoses6D[submap_end_idx].pose.inverse() * keyPoses6D[i].pose);
			}
			publishCloud(pubSubmap, submap_cloud, ros::Time(keyPoses6D[submap_end_idx].time), bodyFrame); // 发送子图点云
			// 为下一子图做准备
			submap_idx++;			// 子图索引+1
			subframe_idx = 0; // 子图内关键帧索引归零
		}
		else
		{
			hash_submap[submap_idx]->keyCloudIndices.push_back(keyPoses3D->size() - 1); // 累积点云索引
		}
	}

	void correctPoses()
	{

		if (keyPoses3D->points.empty())
			return;

		if (aLoopIsClosed == true) // 一旦有新的Prior Factor, GPS Factor 或 Loop Factor加入，则更新历史关键帧位姿
		{
			// 清空模型位姿容器
			modelPoses->poses.clear();
			for (auto &ele : modelInfoSet)
			{
				geometry_msgs::PoseStamped pose_stamped;
				pose_stamped.header.seq = ele.first;
				pose_stamped.header.frame_id = mapFrame;
				pose_stamped.pose = gtsamPose2poseMsg(isamCurrentEstimate.at<Pose3>(M(ele.first)));
				modelPoses->poses.push_back(pose_stamped);
				modelPoses->header.stamp = timeLaserInfoStamp;
				pubModelPoses.publish(modelPoses);
			}
			// clear path 清空里程计轨迹
			globalPath->poses.clear();
			// update key poses 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
			int numPoses = keyPoses3D->size();
			for (int i = 0; i < numPoses; ++i)
			{
				keyPoses6D[i].pose = isamCurrentEstimate.at<Pose3>(X(i));

				keyPoses3D->points[i].x = keyPoses6D[i].pose.translation().x();
				keyPoses3D->points[i].y = keyPoses6D[i].pose.translation().y();
				keyPoses3D->points[i].z = keyPoses6D[i].pose.translation().z();

				// 更新里程计轨迹
				updatePath(keyPoses6D[i]);
			}
			// publish path
			if (pubAftPgoPath.getNumSubscribers() != 0)
			{
				globalPath->header.stamp = timeLaserInfoStamp;

				pubAftPgoPath.publish(globalPath);
			}
			aLoopIsClosed = false;
		}
	}

	void updatePath(const PointTypePose &pose_in)
	{
		geometry_msgs::PoseStamped pose_stamped;
		pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
		pose_stamped.header.frame_id = mapFrame;
		pose_stamped.pose.position.x = pose_in.pose.translation().x();
		pose_stamped.pose.position.y = pose_in.pose.translation().y();
		pose_stamped.pose.position.z = pose_in.pose.translation().z();
		gtsam::Quaternion q = pose_in.pose.rotation().toQuaternion();
		pose_stamped.pose.orientation.x = q.x();
		pose_stamped.pose.orientation.y = q.y();
		pose_stamped.pose.orientation.z = q.z();
		pose_stamped.pose.orientation.w = q.w();

		globalPath->poses.push_back(pose_stamped);
	}

	void publishOdometry()
	{
		// Publish odometry for ROS (global)
		nav_msgs::Odometry::Ptr laserOdometryROS(new nav_msgs::Odometry);
		laserOdometryROS->header.stamp = timeLaserInfoStamp;
		laserOdometryROS->header.frame_id = mapFrame;
		laserOdometryROS->child_frame_id = bodyFrame;
		gtsam::Point3 latestTrans = latestPose6D.translation();
		laserOdometryROS->pose.pose.position.x = latestTrans.x();
		laserOdometryROS->pose.pose.position.y = latestTrans.y();
		laserOdometryROS->pose.pose.position.z = latestTrans.z();
		gtsam::Quaternion latestQuat = latestPose6D.rotation().toQuaternion();
		laserOdometryROS->pose.pose.orientation.x = latestQuat.x();
		laserOdometryROS->pose.pose.orientation.y = latestQuat.y();
		laserOdometryROS->pose.pose.orientation.z = latestQuat.z();
		laserOdometryROS->pose.pose.orientation.w = latestQuat.w();

		pubAftPgoOdom.publish(laserOdometryROS);

		// Publish TF
		// static tf::TransformBroadcaster br;
		// tf::Transform t_odom_to_lidar = tf::Transform(tf::Quaternion(latestQuat.w(), latestQuat.x(), latestQuat.y(), latestQuat.z()),
		// 																							tf::Vector3(latestTrans.x(), latestTrans.y(), latestTrans.z()));
		// tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, mapFrame, bodyFrame);
		// br.sendTransform(trans_odom_to_lidar);
	}

	void publishFrames()
	{
		if (keyPoses3D->points.empty())
			return;
		// publish After Global Optimizition Odometry
		publishOdometry();
		// publish key poses
		publishCloud(pubKeyPoses, keyPoses3D, timeLaserInfoStamp, mapFrame);
	}
};

int main(int argc, char **argv)
{
	ros::init(argc, argv, "r_liom");

	mapOptimization MO;

	ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

	std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
	std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);
	ros::spin();

	loopthread.join();
	visualizeMapThread.join();
	return 0;
}
