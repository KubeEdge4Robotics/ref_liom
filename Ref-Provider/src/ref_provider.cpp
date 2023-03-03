#include "mesh.hpp"
#include <ros/package.h>
#include <ref_liom/header_idx.h>
#include "json/json.h"
#include <igl/read_triangle_mesh.h>

using namespace std;

ros::Publisher pub_prior_constraint;
ros::Publisher pub_prior_lidar_pts;
ros::Publisher pub_prior_mesh_pts_aligned;
ros::Publisher pub_prior_track_bbox;
ros::Publisher pub_prior_constraint_edge;
ros::Publisher pub_model_cloud_normal;
ros::Publisher pub_ground_cloud;

ros::Publisher pub_model_pose;
std::mutex mtx_models;					// 模型容器互斥量
double track_bbox_scale;				// 模型bbox的缩放因子
int segment_num_threshold;			// 分割得到的模型点云的最少数目
int max_opti_num;								// ICP迭代次数
float search_radius;						// Define the circular range for search
double fitness_threshold;				// point-mesh-icp评估阈值
double max_correspondence_dist; // point-mesh匹配最大距离
double overlap_threshold;				// 重合度阈值
double voxel_size;							// 体素尺寸
string map_frame;
string body_frame;
string point_cloud_topic;
string odometry_topic;
string keyframe_idx_topic;
string model_poses_topic;

// 先验模型
std::vector<std::shared_ptr<PriorModel>> prior_models;																	 // 先验模型容器
pcl::PointCloud<pcl::PointXYZ>::Ptr model_pos_cloud(new pcl::PointCloud<pcl::PointXYZ>); // 模型位置点云
// Construct kd-tree object
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_model_pos(new pcl::KdTreeFLANN<pcl::PointXYZ>()); // 模型位置点云kdtree
std::shared_ptr<Recognizer> recognizer;
std::shared_ptr<PriorModel> modelRecognition(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented)
{
	std::shared_ptr<PriorModel> prior_model(new PriorModel);
	Eigen::Vector4f min_pt(-10, -10.5, -3, 1);
	Eigen::Vector4f max_pt(0, 10.5, 10, 1);
	pcl::CropBox<pcl::PointXYZ> crop_filter;
	crop_filter.setInputCloud(cloud_in);
	crop_filter.setMin(min_pt);
	crop_filter.setMax(max_pt);
	crop_filter.filter(*cloud_segmented);
	return prior_model;
}

void estimateOBB(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::shared_ptr<PriorModel> prior_model)
{
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud_in, centroid);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_model(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>); // mesh虚拟匹配点
	// 世界系 -> model自身坐标系
	Eigen::Affine3d T_w2m = Eigen::Affine3d::Identity();
	T_w2m.pretranslate(Eigen::Vector3d(-centroid(0), -centroid(1), -centroid(2)));
	Point2meshICP point2model_icp;
	for (int i = 1; i <= max_opti_num; i++)
	{
		// 更新原始点云world_to_model
		pcl::transformPointCloud(*cloud_in, *cloud_segmented_model, T_w2m); // 更新之后的模型系点云
		point2model_icp(cloud_segmented_model, prior_model, model_cloud);
		Eigen::Affine3d T_inc = point2model_icp.getTransform(); // 位姿增量
		T_w2m = T_inc * T_w2m;																	// 更新T_w2m
	}
	double x, y, z, roll, pitch, yaw;
	pcl::getTranslationAndEulerAngles(T_w2m, x, y, z, roll, pitch, yaw);
	prior_model->bbox.trans << x, y, z;
	prior_model->bbox.rot << roll, pitch, yaw;
	// 将模型位置加入model_pos_cloud中
	model_pos_cloud->push_back(pcl::PointXYZ(x, y, z));
	// 估计bbox的尺寸
	// 将矩阵转换为pcl点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	// 遍历矩阵
	Eigen::MatrixXd &vertex_matrix = prior_model->mesh_ptr->vertices;
	uint32_t vertex_num = vertex_matrix.rows();
	for (uint32_t i = 0; i < vertex_num; i++)
	{
		// 将点添加到点云中
		cloud->points.push_back(pcl::PointXYZ(vertex_matrix(i, 0), vertex_matrix(i, 1), vertex_matrix(i, 2)));
	}
	pcl::PointXYZ min_pt, max_pt;
	pcl::getMinMax3D(*cloud, min_pt, max_pt);
	prior_model->bbox.min_pt << min_pt.x, min_pt.y, min_pt.z;
	prior_model->bbox.max_pt << max_pt.x, max_pt.y, max_pt.z;
}

void visualizePriorConstraint(const ros::Time &time_stamp, const Eigen::Vector3d &trans_b2w, const std::vector<Eigen::Vector3d> &model_pos)
{
	visualization_msgs::MarkerArray markerArray;
	// prior nodes
	static visualization_msgs::Marker markerNode;
	markerNode.header.frame_id = map_frame;
	markerNode.header.stamp = time_stamp;
	markerNode.action = visualization_msgs::Marker::ADD;
	markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
	markerNode.ns = "prior_nodes";
	markerNode.id = 2;
	markerNode.pose.orientation.w = 1;
	markerNode.scale.x = 0.3;
	markerNode.scale.y = 0.3;
	markerNode.scale.z = 0.3;
	markerNode.color.r = 0;
	markerNode.color.g = 0;
	markerNode.color.b = 1;
	markerNode.color.a = 1;
	// prior edges
	static visualization_msgs::Marker markerEdge;
	markerEdge.header.frame_id = map_frame;
	markerEdge.header.stamp = time_stamp;
	markerEdge.action = visualization_msgs::Marker::ADD;
	markerEdge.type = visualization_msgs::Marker::LINE_LIST;
	markerEdge.ns = "prior_edges";
	markerEdge.id = 3;
	markerEdge.pose.orientation.w = 1;
	markerEdge.scale.x = 0.1;
	markerEdge.color.r = 0;
	markerEdge.color.g = 0.9;
	markerEdge.color.b = 0;
	markerEdge.color.a = 1;

	for (auto &trans_m2w : model_pos)
	{
		geometry_msgs::Point p;
		p.x = trans_b2w.x();
		p.y = trans_b2w.y();
		p.z = trans_b2w.z();
		markerNode.points.push_back(p);
		markerEdge.points.push_back(p);
		p.x = trans_m2w.x();
		p.y = trans_m2w.y();
		p.z = trans_m2w.z();
		markerNode.points.push_back(p);
		markerEdge.points.push_back(p);
		/* // 模型绝对位姿
		p.x = -prior_models.front()->bbox.trans.x();
		p.y = -prior_models.front()->bbox.trans.y();
		p.z = -prior_models.front()->bbox.trans.z();
		markerNode.points.push_back(p); */
	}

	markerArray.markers.push_back(markerNode);
	markerArray.markers.push_back(markerEdge);
	pub_prior_constraint_edge.publish(markerArray);
}
void syncedCallback(const nav_msgs::Odometry::ConstPtr &msg_odom, const sensor_msgs::PointCloud2ConstPtr &msg_cloud, const ref_liom::header_idx::ConstPtr &msg_idx)
{
	ROS_INFO("\033[1;31m xxxxsyncedCallbackStart, %f  \033[0m", ros::Time::now().toSec());
	// 可视化用的
	std::vector<Eigen::Vector3d> model_pos;
	// PCL
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_body(new pcl::PointCloud<pcl::PointXYZ>); // pcl_cloud_msg要转化成的点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_world(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_model(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_body(new pcl::PointCloud<pcl::PointXYZ>);	 // 分割算法得到的点云(body系)
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_segmented_world(new pcl::PointCloud<pcl::PointXYZ>); // 分割得到的点云(世界系)
	uint32_t keypose_idx = msg_idx->idx;																													 // 关键帧id
	// 转化：ROS 点云 -> PCL
	pcl::fromROSMsg(*msg_cloud, *pcl_cloud_body);
	pcl::IndicesPtr indices_plane(new pcl::Indices);
	// 分割剔除地面点
	segmentPlane<pcl::PointXYZ>(pcl_cloud_body, *indices_plane);
	pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
	// 提取非平面点云
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	// 提取平面点
	extract.setInputCloud(pcl_cloud_body);
	extract.setIndices(indices_plane);
	extract.setNegative(false); // 设置为剔除索引提供的点云
	extract.filter(*ground_points);
	// 提取非平面点
	extract.setInputCloud(pcl_cloud_body);
	extract.setIndices(indices_plane);
	extract.setNegative(true); // 设置为剔除索引提供的点云
	extract.filter(*pcl_cloud_body);
	// 发布地面点云
	// publishCloud(pub_ground_cloud, ground_points, msg_odom->header.stamp, body_frame);
	// 坐标转换body_to_world
	Eigen::Affine3d T_b2w = poseMsg2Affine3d(msg_odom->pose.pose);
	pcl::transformPointCloud(*pcl_cloud_body, *pcl_cloud_world, T_b2w.matrix());
	// pcl_cloud_body.reset();
	publishCloud(pub_ground_cloud, pcl_cloud_world, msg_odom->header.stamp, map_frame);

	// 估计法向量
	// pcl::PointCloud<pcl::Normal>::Ptr cloud_world_normals = computeNormals<pcl::PointXYZ, pcl::Normal>(pcl_cloud_world, 0, 2.0);
	// cout << "cloud_world_normals.size():" << cloud_world_normals->size() << endl;
	// 检测
	// if (recognizer)
	// 	recognizer->recognize(pcl_cloud_world, cloud_world_normals, Recognizer::ObjRecMode::FULL_MODE);

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	pcl::PointXYZ query_pos; // 查询位置
	query_pos.x = msg_odom->pose.pose.position.x;
	query_pos.y = msg_odom->pose.pose.position.y;
	query_pos.z = msg_odom->pose.pose.position.z;
	// Search the kd-tree
	kdtree_model_pos->setInputCloud(model_pos_cloud);
	unsigned int results = kdtree_model_pos->radiusSearch(query_pos, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
	// 对于每一个可能提取到的先验模型
	for (size_t i = 0; i < results; i++)
	{
		int model_idx = pointIdxRadiusSearch[i]; // 模型的索引
		std::shared_ptr<PriorModel> &prior_model = prior_models[model_idx];
		Eigen::Affine3d T_w2m; // 世界系->模型系
		{
			std::unique_lock<std::mutex> ulock(mtx_models);
			pcl::getTransformation(prior_model->bbox.trans(0), prior_model->bbox.trans(1), prior_model->bbox.trans(2),
														 prior_model->bbox.rot(0), prior_model->bbox.rot(1), prior_model->bbox.rot(2), T_w2m);
		}
		Eigen::Affine3d T_b2m = T_w2m * T_b2w;
		pcl::transformPointCloud(*pcl_cloud_body, *pcl_cloud_model, T_b2m); // 分割点云由世界系转换到模型系下
		pcl::IndicesPtr segmented_indices(new std::vector<int>);						// 分割点云的索引
		// 设置跟踪bounding box
		Eigen::Vector4f min_pt = Eigen::Vector4f::Ones();
		Eigen::Vector4f max_pt = Eigen::Vector4f::Ones();
		Eigen::Vector3f exponsion = track_bbox_scale * (prior_model->bbox.max_pt - prior_model->bbox.min_pt);
		min_pt.head<3>() = prior_model->bbox.min_pt - exponsion;
		max_pt.head<3>() = prior_model->bbox.max_pt + exponsion;
		// Create the crop box filter and apply it
		pcl::CropBox<pcl::PointXYZ> crop_filter;
		crop_filter.setInputCloud(pcl_cloud_model);
		crop_filter.setMin(min_pt);
		crop_filter.setMax(max_pt);
		crop_filter.filter(*segmented_indices);
		// 如果分割点云过少，则跳过当前循环
		if (segmented_indices->size() < segment_num_threshold)
		{
			cout << "too few segmented points: " << segmented_indices->size() << endl;
			continue;
		}
		// 分割后点云(world系)
		pcl::copyPointCloud(*pcl_cloud_world, *segmented_indices, *cloud_segmented_world);
		// 分割后点云(body系)
		pcl::copyPointCloud(*pcl_cloud_body, *segmented_indices, *cloud_segmented_body);
		publishCloud(pub_prior_track_bbox, cloud_segmented_world, ros::Time::now(), map_frame);
		// 提取剩余点云
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(pcl_cloud_body);
		extract.setIndices(segmented_indices);
		extract.setNegative(true); // 设置为排除索引提供的点云
		extract.filter(*pcl_cloud_body);
		pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud(new pcl::PointCloud<pcl::PointXYZ>); // mesh匹配虚拟点
		Point2meshICP point2model_icp;																											 // 点到mesh ICP
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_others(new pcl::PointCloud<pcl::PointXYZ>);
		if (!modelRegister(prior_model, point2model_icp, T_b2m,
											 cloud_segmented_body, cloud_others,
											 model_cloud, segment_num_threshold,
											 fitness_threshold, max_correspondence_dist,
											 voxel_size, overlap_threshold, max_opti_num))
		{
			break;
		}
		*pcl_cloud_body += *cloud_others; // 剩余点云放回pcl_cloud_body
		cout << "modelRegister finished" << endl;
		T_w2m = T_b2m * T_b2w.inverse();
		Eigen::Affine3d T_m2w = T_w2m.inverse();
		if (point2model_icp.hasConverged() && point2model_icp.getFitness() < 0.01 * fitness_threshold)
		{
			float overlap_score = calculateOverlapScore<pcl::PointXYZ>(model_cloud, prior_model->hash_map3d, voxel_size, 0.5);
			cout << "two stage overlap score: " << overlap_score << endl;
			if (overlap_score < overlap_threshold)
			{
				cout << "overlap score too low" << endl;
			}
			else
			{
				// 发送模型与雷达帧的约束
				geometry_msgs::PoseWithCovarianceStamped msg_constraint;
				msg_constraint.header.stamp = msg_odom->header.stamp;
				msg_constraint.header.frame_id = body_frame;
				Eigen::Affine3d T_m2b = T_b2m.inverse(); // model->body约束
				msg_constraint.pose.pose = Affine3d2poseMsg(T_m2b);
				msg_constraint.pose.covariance[0] = keypose_idx; // 关键帧id
				msg_constraint.pose.covariance[1] = model_idx;	 // 模型id
				double cov = point2model_icp.getFitness();
				overlap_score = overlap_score * overlap_score;
				msg_constraint.pose.covariance[2] = cov / overlap_score; // 位移协方差
				msg_constraint.pose.covariance[3] = cov / overlap_score; // 旋转协方差
				pub_prior_constraint.publish(msg_constraint);
				// 可视化用的
				model_pos.push_back(T_m2w.translation());
				// 发布模型位姿
				nav_msgs::Odometry::Ptr msg(new nav_msgs::Odometry);
				msg->header.stamp = ros::Time::now();
				msg->header.frame_id = map_frame;
				msg->child_frame_id = "model";
				msg->pose.pose = Affine3d2poseMsg(T_m2w);
				pub_model_pose.publish(msg);
			}
		}
		// 分割后的点云（世界坐标系）、迭代前的虚拟点云、迭代后的虚拟点云
		pcl::transformPointCloud(*cloud_segmented_body, *cloud_segmented_world, T_m2w * T_b2m); // 更新之后的模型系点云
		publishCloud(pub_prior_lidar_pts, cloud_segmented_world, msg_odom->header.stamp, map_frame);
		// 迭代后的虚拟点云转换会world系发布
		pcl::transformPointCloud(*model_cloud, *model_cloud, T_m2w); // 更新之后的模型系点云
		publishCloud(pub_prior_mesh_pts_aligned, model_cloud, msg_odom->header.stamp, map_frame);
	}
	// 可视化先验约束
	if (model_pos.size() > 0)
		visualizePriorConstraint(msg_odom->header.stamp, T_b2w.translation(), model_pos);
	// // 对于剩下的点云进行分割，得到在world系下模型扫描点云模型
	// std::shared_ptr<PriorModel> prior_model = modelRecognition(pcl_cloud_world, cloud_segmented_world);
	// // 将模型放入容器
	// prior_models.push_back(prior_model);
	// estimateOBB(cloud_segmented_world, prior_model);
}

void modelPosesCallback(const nav_msgs::Path::ConstPtr &msg)
{
	// 校正模型位姿
	std::unique_lock<std::mutex> ulock(mtx_models);
	for (auto &pose : msg->poses)
	{
		double x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(poseMsg2Affine3d(pose.pose).inverse(), x, y, z, roll, pitch, yaw);
		prior_models[pose.header.seq]->bbox.trans << x, y, z;
		prior_models[pose.header.seq]->bbox.rot << roll, pitch, yaw;
		model_pos_cloud->points[pose.header.seq].x = x;
		model_pos_cloud->points[pose.header.seq].y = y;
		model_pos_cloud->points[pose.header.seq].z = z;
	}
}
// 从文件中读取先验信息
bool modelsInit(const string &path)
{
	// read json from stream
	std::ifstream ifs;
	ifs.open(path);
	if (!ifs.is_open())
		std::cerr << "ifstream open error!" << std::endl;
	Json::Value root;
	Json::CharReaderBuilder builder;
	builder["collectComments"] = true;
	JSONCPP_STRING errs;
	if (!parseFromStream(builder, ifs, &root, &errs))
	{
		std::cerr << errs << std::endl;
		return false;
	}
	TicToc t_model_init; // 模型初始化的时间
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_points_all(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::UniformSampling<pcl::PointXYZ> uniform_sampling; // 下采样滤波模型
	// 分割先验模型点云
	// recognizer.reset(new Recognizer(40.0, 5.0, 5));
	for (auto &m : root["models"])
	{
		std::shared_ptr<Mesh> mesh_ptr(new Mesh);
		igl::read_triangle_mesh(m["path"].asString(), mesh_ptr->vertices, mesh_ptr->facets);
		Eigen::Affine3d T_m2w;
		pcl::getTransformation(m["trans"][0].asDouble(), m["trans"][1].asDouble(), m["trans"][2].asDouble(),
													 m["rot"][0].asDouble(), m["rot"][1].asDouble(), m["rot"][2].asDouble(), T_m2w);
		Eigen::Vector3f min_pt(m["min_pt"][0].asFloat(), m["min_pt"][1].asFloat(), m["min_pt"][2].asFloat());
		Eigen::Vector3f max_pt(m["max_pt"][0].asFloat(), m["max_pt"][1].asFloat(), m["max_pt"][2].asFloat());
		std::shared_ptr<PriorModel> model_ptr = modelInit(m["name"].asString(), mesh_ptr, T_m2w.inverse(), voxel_size);

		// mesh to point cloud with normal vector
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_points(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(new pcl::PointCloud<pcl::Normal>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_points_ds(new pcl::PointCloud<pcl::PointXYZ>);
		uniform_sampling.setInputCloud(model_ptr->cloud_ptr); // 模型点云
		uniform_sampling.setRadiusSearch(voxel_size);					// 模型点云下采样滤波搜索半径
		uniform_sampling.filter(*cloud_points_ds);						// 下采样得到的关键点
		// pcl::PointCloud<pcl::Normal>::Ptr model_normals = computeNormals<pcl::PointXYZ, pcl::Normal>(cloud_points_ds, int(10), 0.0, model_ptr->cloud_ptr);
		// recognizer->addModel(cloud_points_ds, model_normals, model_ptr->name);
		pcl::transformPointCloud(*(model_ptr->cloud_ptr), *cloud_points, T_m2w.matrix());
		*cloud_points_all += *cloud_points;
		// model pose加入点云中
		model_pos_cloud->push_back(pcl::PointXYZ(model_ptr->bbox.trans(0), model_ptr->bbox.trans(1), model_ptr->bbox.trans(2)));
		prior_models.push_back(model_ptr); // 将模型加入容器中
	}
	cout << "models have inited. cost time(ms):" << t_model_init.toc() << endl;

	ros::Rate rate(10);
	while (ros::ok())
	{
		rate.sleep();
		publishCloud(pub_model_cloud_normal, cloud_points_all, ros::Time::now(), map_frame);
	}
	return true;
}
int main(int argc, char **argv)
{
	ros::init(argc, argv, "ref_provider");
	ros::NodeHandle nh;
	ROS_INFO("\033[1;31m **********Point-Mesh Thread Started********** \033[0m");

	nh.param<double>("ref_provider/track_bbox_scale", track_bbox_scale, 1.0);
	nh.param<int>("ref_provider/segment_num_threshold", segment_num_threshold, 500);
	nh.param<int>("ref_provider/max_opti_num", max_opti_num, 5);
	nh.param<float>("ref_provider/search_radius", search_radius, 100.0);
	nh.param<double>("ref_provider/fitness_threshold", fitness_threshold, 0.02);
	nh.param<double>("ref_provider/max_correspondence_dist", max_correspondence_dist, 1);
	nh.param<double>("ref_provider/overlap_threshold", overlap_threshold, 0.5);
	nh.param<double>("ref_provider/voxel_size", voxel_size, 0.1);
	nh.param<string>("ref_provider/map_frame", map_frame, "map");
	nh.param<string>("ref_provider/body_frame", body_frame, "body");
	nh.param<string>("ref_provider/point_cloud_topic", point_cloud_topic, "submap");
	nh.param<string>("ref_provider/odometry_topic", odometry_topic, "aftPgo_odom");
	nh.param<string>("ref_provider/keyframe_idx_topic", keyframe_idx_topic, "submap_key_idx");
	nh.param<string>("ref_provider/model_poses_topic", model_poses_topic, "model_poses");
	string json_path = ros::package::getPath("ref_provider") + "/config/params.json";
	std::thread m_init(modelsInit, json_path);
	// if (!modelsInit(json_path))
	// 	std::cerr << "model init error" << std::endl;
	std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> subOdom;
	std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> subCloud;
	std::shared_ptr<message_filters::Subscriber<ref_liom::header_idx>> subIdx;

	typedef message_filters::sync_policies::ExactTime<nav_msgs::Odometry, sensor_msgs::PointCloud2, ref_liom::header_idx> SyncPolicy;
	std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> odom_cloud_sync;
	subOdom = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(nh, odometry_topic, 1000);
	subCloud = std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(nh, point_cloud_topic, 1000);
	subIdx = std::make_shared<message_filters::Subscriber<ref_liom::header_idx>>(nh, keyframe_idx_topic, 1000);
	odom_cloud_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(1000), *subOdom, *subCloud, *subIdx); // 1000是消息队列长度
	odom_cloud_sync->registerCallback(boost::bind(&syncedCallback, _1, _2, _3));
	ros::Subscriber subModelPoses = nh.subscribe<nav_msgs::Path>(model_poses_topic, 1000, modelPosesCallback); // 先验模型优化位姿

	pub_prior_constraint = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("prior_constraints", 1000); // 先验模型约束
	pub_prior_lidar_pts = nh.advertise<sensor_msgs::PointCloud2>("prior_lidar_pts", 1000);
	pub_prior_mesh_pts_aligned = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_pts_aligned", 1000);
	pub_prior_track_bbox = nh.advertise<sensor_msgs::PointCloud2>("prior_tracking_bbox", 1000);
	pub_prior_constraint_edge = nh.advertise<visualization_msgs::MarkerArray>("prior_constraints_marker", 1000);
	pub_model_pose = nh.advertise<nav_msgs::Odometry>("prior_model_pose", 1000);
	pub_model_cloud_normal = nh.advertise<sensor_msgs::PointCloud2>("prior_mesh_cloud_normal", 1000);
	pub_ground_cloud = nh.advertise<sensor_msgs::PointCloud2>("ground_plane_cloud", 1000);

	sleep(3);
	ros::spin();
	m_init.join();
	return 0;
}
